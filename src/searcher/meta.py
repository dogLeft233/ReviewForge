"""MetaSearcher 主协调器"""

import asyncio
import logging
from typing import Optional

from src.searcher.adapters.router import AdapterRouter
from src.searcher.llm import LLMQueryGenerator
from src.searcher.models import DomainProfile, QueryConfig, fallback_query_config
from src.searcher.result import SearchContext, SearchResult
from src.searcher.rankers import apply_boost, deduplicate, rrf_fusion, rerank as do_rerank

logger = logging.getLogger(__name__)


class MetaSearcher:
    """多源联网搜索 + 多阶段排序模块

    完整流程：Phase 0 LLM查询生成 → Phase 1 并行检索 → Phase 2 RRF融合
           → Phase 3 URL去重 → Phase 4 Rerank → Phase 5 Boost
    """

    def __init__(self) -> None:
        self._router = AdapterRouter()
        self._llm = LLMQueryGenerator()

    async def search(
        self,
        query: str,
        domain_profile: Optional[DomainProfile] = None,
        max_results: int = 10,
    ) -> SearchContext:
        """完整流程：LLM查询生成 → 并行检索 → RRF融合 → 去重 → Rerank → Boost

        Args:
            query: 用户输入的搜索主题
            domain_profile: Explorer 输出的领域知识（可选，为 None 时使用简单查询）
            max_results: 每类返回的最大结果数

        Returns:
            SearchContext: 包含 papers/resources/news 分类结果
        """

        # Phase 0: LLM 查询生成（如有 domain_profile）
        if domain_profile is not None:
            logger.info("[Phase 0] LLM 生成优化查询: topic='%s'", query)
            query_config = await self._llm.generate_queries(query, domain_profile)
        else:
            logger.info("[Phase 0] 无 DomainProfile，使用降级查询")
            query_config = fallback_query_config(query)

        logger.debug(
            "[Phase 0] query_config: %d sources, rerank_query='%s'",
            len(query_config.queries),
            query_config.rerank_query,
        )

        # Phase 1: 并行检索（使用 QueryConfig）
        papers_raw, resources_raw, news_raw = await self._parallel_search(query_config, max_results)

        logger.debug(
            "[Phase 1] 原始结果 — papers: %s, resources: %s, news: %s",
            [len(r) for r in papers_raw],
            [len(r) for r in resources_raw],
            [len(r) for r in news_raw],
        )

        # Phase 2: RRF 融合（分类内）
        logger.info("[Phase 2] RRF 融合")
        papers_fused = rrf_fusion({
            src: results
            for src, results in [("arxiv", papers_raw[0] if len(papers_raw) > 0 else []), ("serper", papers_raw[1] if len(papers_raw) > 1 else [])]
            if results
        }) if papers_raw else []
        resources_fused = rrf_fusion({
            src: results
            for src, results in [("github", resources_raw[0] if len(resources_raw) > 0 else []), ("huggingface", resources_raw[1] if len(resources_raw) > 1 else [])]
            if results
        }) if resources_raw else []
        news_fused = rrf_fusion({
            src: results
            for src, results in [("bocha", news_raw[0] if len(news_raw) > 0 else []), ("hackernews", news_raw[1] if len(news_raw) > 1 else [])]
            if results
        }) if news_raw else []

        logger.debug(
            "[Phase 2] 融合后 — papers: %d, resources: %d, news: %d",
            len(papers_fused),
            len(resources_fused),
            len(news_fused),
        )

        # Phase 3: URL 去重
        logger.info("[Phase 3] URL 去重")
        papers_dedup = deduplicate(papers_fused)
        resources_dedup = deduplicate(resources_fused)
        news_dedup = deduplicate(news_fused)

        logger.debug(
            "[Phase 3] 去重后 — papers: %d, resources: %d, news: %d",
            len(papers_dedup),
            len(resources_dedup),
            len(news_dedup),
        )

        # Phase 4: SiliconFlow Rerank（使用 LLM 优化的 rerank_query）
        logger.info("[Phase 4] SiliconFlow Rerank (query='%s')", query_config.rerank_query)
        papers_reranked = do_rerank(query_config.rerank_query, papers_dedup)
        resources_reranked = do_rerank(query_config.rerank_query, resources_dedup)
        news_reranked = do_rerank(query_config.rerank_query, news_dedup)

        # Phase 5: Boost（经典论文 + 高引用加权）
        logger.info("[Phase 5] Boost")
        papers_boosted = apply_boost(papers_reranked, query_config.boost)
        resources_boosted = apply_boost(resources_reranked, query_config.boost)
        news_boosted = apply_boost(news_reranked, query_config.boost)

        logger.info(
            "[Done] topic='%s' — papers: %d, resources: %d, news: %d",
            query,
            len(papers_boosted),
            len(resources_boosted),
            len(news_boosted),
        )

        return SearchContext(
            topic=query,
            papers=papers_boosted,
            resources=resources_boosted,
            news=news_boosted,
        )

    async def _parallel_search(
        self, query_config: QueryConfig, max_results: int
    ) -> tuple[list[list[SearchResult]], list[list[SearchResult]], list[list[SearchResult]]]:
        """三类适配器并行检索，支持 QueryConfig 多查询变体"""

        async def run_adapter(adapter, source_name: str) -> tuple[str, list[SearchResult]]:
            try:
                # 优先使用 variant 查询，否则用 topic
                variant_query = query_config.get_query(source_name, "primary")
                if not variant_query:
                    variant_query = query_config.topic
                results = await adapter.search(variant_query, max_results)
                logger.debug("Adapter '%s' returned %d results for query='%s'", source_name, len(results), variant_query)
                return adapter.name, results
            except Exception as e:
                logger.warning("Adapter '%s' failed: %s", source_name, e)
                return adapter.name, []

        papers_adapters = self._router.route(query_config.topic, "papers")
        resources_adapters = self._router.route(query_config.topic, "resources")
        news_adapters = self._router.route(query_config.topic, "news")

        all_tasks = [
            run_adapter(a, a.name)
            for a in papers_adapters + resources_adapters + news_adapters
        ]

        results_map: dict[str, list[SearchResult]] = {}
        for name, res in await asyncio.gather(*all_tasks):
            results_map[name] = res

        papers_results = [] if not papers_adapters else [results_map.get(a.name, []) for a in papers_adapters]
        resources_results = [] if not resources_adapters else [results_map.get(a.name, []) for a in resources_adapters]
        news_results = [] if not news_adapters else [results_map.get(a.name, []) for a in news_adapters]

        return papers_results, resources_results, news_results