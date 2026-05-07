"""MetaSearcher 主协调器"""

import asyncio
import logging
import math
from typing import Optional

from src.searcher.adapters.router import AdapterRouter
from src.searcher.llm import LLMQueryGenerator
from src.searcher.models import DomainProfile, QueryConfig
from src.searcher.result import SearchContext, SearchResult
from src.searcher.rankers import apply_boost, deduplicate, rrf_fusion, rerank as do_rerank

logger = logging.getLogger(__name__)


def _build_results_preview(
    papers_raw: list[list[SearchResult]],
    resources_raw: list[list[SearchResult]],
    news_raw: list[list[SearchResult]],
    max_per: int = 5,
) -> str:
    """从原始检索结果构建文本预览（用于 LLM Rerank 优化）"""
    lines = []
    for label, source_list in [("Papers", papers_raw), ("Resources", resources_raw), ("News", news_raw)]:
        count = 0
        for results in source_list:
            for r in results[:max_per]:
                title = r.title[:80] if r.title else "(no title)"
                lines.append(f"[{label}] {title}")
                lines.append(f"       URL: {r.url}")
                if r.snippet:
                    lines.append(f"       Snippet: {r.snippet[:100]}")
                count += 1
        if count == 0:
            lines.append(f"[{label}] (no results)")
    return "\n".join(lines)


class MetaSearcher:
    """多源联网搜索 + 多阶段排序模块

    完整流程：Phase 0 LLM查询生成 → Phase 1 并行检索 → Phase 1.5 Rerank查询优化
           → Phase 2 RRF融合 → Phase 3 URL去重 → Phase 4 Rerank → Phase 5 Boost
    """

    def __init__(self) -> None:
        self._router = AdapterRouter()
        self._llm = LLMQueryGenerator()

    async def search(
        self,
        query: str,
        domain_profile: Optional[DomainProfile] = None,
        max_papers: int = 30,
        max_resources: int = 30,
        max_news: int = 30,
    ) -> SearchContext:
        """完整流程：LLM查询生成 → 并行检索 → Rerank优化 → RRF融合 → 去重 → Rerank → Boost

        Args:
            query: 用户输入的搜索主题
            domain_profile: Explorer 输出的领域知识（可选，为 None 时使用空 Profile）
            max_papers: 论文最大返回数（默认 30）
            max_resources: 资源最大返回数（默认 30）
            max_news: 新闻最大返回数（默认 30）

        Returns:
            SearchContext: 包含 papers/resources/news 分类结果
        """

        # Phase 0: LLM 查询生成（始终尝试 LLM，无 domain_profile 时用空 Profile）
        profile = domain_profile if domain_profile is not None else DomainProfile.empty(query)
        logger.info("[Phase 0] LLM 生成优化查询: topic='%s'", query)
        query_config = await self._llm.generate_queries(query, profile)

        logger.debug(
            "[Phase 0] query_config: %d sources, rerank_query='%s'",
            len(query_config.queries),
            query_config.rerank_query,
        )

        # Phase 1: 并行检索
        papers_raw, resources_raw, news_raw = await self._parallel_search(
            query_config, max_papers, max_resources, max_news
        )

        logger.debug(
            "[Phase 1] 原始结果 — papers: %s, resources: %s, news: %s",
            [len(r) for r in papers_raw],
            [len(r) for r in resources_raw],
            [len(r) for r in news_raw],
        )

        # Phase 1.5: 根据检索结果优化 Rerank 查询 + 获取额外加权建议
        logger.info("[Phase 1.5] 根据检索结果优化 Rerank 查询")
        preview = _build_results_preview(papers_raw, resources_raw, news_raw)
        rerank_result = await self._llm.generate_rerank_query(
            original_query=query,
            results_preview=preview,
            domain_profile=profile,
        )
        optimized_rerank_query = rerank_result.query or query_config.rerank_query
        boost_override = rerank_result.boost_override or None
        logger.debug(
            "[Phase 1.5] 优化后 rerank_query='%s', boost_hint=%d urls",
            optimized_rerank_query,
            len(rerank_result.urls_to_boost),
        )

        # Phase 2: RRF 融合（分类内）
        logger.info("[Phase 2] RRF 融合")
        papers_fused = rrf_fusion({
            src: results
            for src, results in [
                ("arxiv", papers_raw[0] if len(papers_raw) > 0 else []),
                ("semantic_scholar", papers_raw[1] if len(papers_raw) > 1 else []),
                ("serper", papers_raw[2] if len(papers_raw) > 2 else []),
            ]
            if results
        }) if papers_raw else []
        resources_fused = rrf_fusion({
            src: results
            for src, results in [
                ("github", resources_raw[0] if len(resources_raw) > 0 else []),
                ("huggingface", resources_raw[1] if len(resources_raw) > 1 else []),
            ]
            if results
        }) if resources_raw else []
        news_fused = rrf_fusion({
            src: results
            for src, results in [
                ("bocha", news_raw[0] if len(news_raw) > 0 else []),
                ("hackernews", news_raw[1] if len(news_raw) > 1 else []),
            ]
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

        # Phase 4: SiliconFlow Rerank（使用动态优化的 rerank_query）
        logger.info("[Phase 4] SiliconFlow Rerank (query='%s')", optimized_rerank_query)
        papers_reranked = do_rerank(optimized_rerank_query, papers_dedup)
        resources_reranked = do_rerank(optimized_rerank_query, resources_dedup)
        news_reranked = do_rerank(optimized_rerank_query, news_dedup)

        # Phase 5: Boost（经典论文 + 高引用加权 + LLM 加权建议）
        logger.info("[Phase 5] Boost")
        papers_boosted = apply_boost(papers_reranked, query_config.boost, override=boost_override)
        resources_boosted = apply_boost(resources_reranked, query_config.boost, override=boost_override)
        news_boosted = apply_boost(news_reranked, query_config.boost, override=boost_override)

        # Phase 6: 最终截断
        papers_final = papers_boosted[:max_papers]
        resources_final = resources_boosted[:max_resources]
        news_final = news_boosted[:max_news]

        logger.info(
            "[Done] topic='%s' — papers: %d, resources: %d, news: %d",
            query,
            len(papers_final),
            len(resources_final),
            len(news_final),
        )

        return SearchContext(
            topic=query,
            papers=papers_final,
            resources=resources_final,
            news=news_final,
        )

    async def _parallel_search(
        self,
        query_config: QueryConfig,
        max_papers: int,
        max_resources: int,
        max_news: int,
    ) -> tuple[list[list[SearchResult]], list[list[SearchResult]], list[list[SearchResult]]]:
        """三类适配器并行检索，支持 QueryConfig 多查询变体"""

        async def run_adapter(
            adapter, source_name: str, per_adapter_max: int
        ) -> tuple[str, list[SearchResult]]:
            try:
                variant_query = query_config.get_query(source_name, "primary")
                if not variant_query:
                    variant_query = query_config.topic
                results = await adapter.search(variant_query, per_adapter_max)
                logger.debug(
                    "Adapter '%s' returned %d results for query='%s'",
                    source_name, len(results), variant_query
                )
                return adapter.name, results
            except Exception as e:
                logger.warning("Adapter '%s' failed: %s", source_name, e)
                return adapter.name, []

        papers_adapters = self._router.route(query_config.topic, "papers")
        resources_adapters = self._router.route(query_config.topic, "resources")
        news_adapters = self._router.route(query_config.topic, "news")

        n_papers = len(papers_adapters)
        n_resources = len(resources_adapters)
        n_news = len(news_adapters)
        per_paper = max(math.ceil(max_papers / n_papers), 5) if n_papers else 5
        per_resource = max(math.ceil(max_resources / n_resources), 5) if n_resources else 5
        per_news = max(math.ceil(max_news / n_news), 5) if n_news else 5

        all_tasks = (
            [run_adapter(a, a.name, per_paper) for a in papers_adapters]
            + [run_adapter(a, a.name, per_resource) for a in resources_adapters]
            + [run_adapter(a, a.name, per_news) for a in news_adapters]
        )

        results_map: dict[str, list[SearchResult]] = {}
        for name, res in await asyncio.gather(*all_tasks):
            results_map[name] = res

        papers_results = (
            [] if not papers_adapters
            else [results_map.get(a.name, []) for a in papers_adapters]
        )
        resources_results = (
            [] if not resources_adapters
            else [results_map.get(a.name, []) for a in resources_adapters]
        )
        news_results = (
            [] if not news_adapters
            else [results_map.get(a.name, []) for a in news_adapters]
        )
        return papers_results, resources_results, news_results
