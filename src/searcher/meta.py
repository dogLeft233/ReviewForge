"""MetaSearcher 主协调器"""

import asyncio
import logging
from typing import Any

from src.searcher.adapters.router import AdapterRouter
from src.searcher.result import SearchContext, SearchResult
from src.searcher.rankers import rrf_fusion, deduplicate, rerank as do_rerank

logger = logging.getLogger(__name__)


class MetaSearcher:
    """多源联网搜索 + 多阶段排序模块"""

    def __init__(self) -> None:
        self._router = AdapterRouter()

    async def search(self, query: str, max_results: int = 10) -> SearchContext:
        """完整流程：并行检索 → RRF 融合 → 去重 → Rerank → 分类结果"""

        logger.info("[Phase 1] 并行检索: query='%s', max_results=%d", query, max_results)

        # Phase 1: 并行检索
        papers_raw, resources_raw, news_raw = await self._parallel_search(query, max_results)

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

        # Phase 3: 去重
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

        # Phase 4: Rerank（SiliconFlow API）
        logger.info("[Phase 4] SiliconFlow Rerank")
        papers_reranked = do_rerank(query, papers_dedup)
        resources_reranked = do_rerank(query, resources_dedup)
        news_reranked = do_rerank(query, news_dedup)

        logger.info(
            "[Done] topic='%s' — papers: %d, resources: %d, news: %d",
            query,
            len(papers_reranked),
            len(resources_reranked),
            len(news_reranked),
        )

        return SearchContext(
            topic=query,
            papers=papers_reranked,
            resources=resources_reranked,
            news=news_reranked,
        )

    async def _parallel_search(
        self, query: str, max_results: int
    ) -> tuple[list[list[SearchResult]], list[list[SearchResult]], list[list[SearchResult]]]:
        """三类适配器并行检索"""

        async def run_adapter(adapter: Any) -> tuple[str, list[SearchResult]]:
            try:
                results = await adapter.search(query, max_results)
                return adapter.name, results
            except Exception:
                return adapter.name, []

        # papers 类
        papers_adapters = self._router.route(query, "papers")
        # resources 类
        resources_adapters = self._router.route(query, "resources")
        # news 类
        news_adapters = self._router.route(query, "news")

        all_tasks = [
            run_adapter(a) for a in papers_adapters + resources_adapters + news_adapters
        ]

        results_map: dict[str, list[SearchResult]] = {}
        for name, res in await asyncio.gather(*all_tasks):
            results_map[name] = res

        papers_results = [] if not papers_adapters else [results_map.get(a.name, []) for a in papers_adapters]
        resources_results = [] if not resources_adapters else [results_map.get(a.name, []) for a in resources_adapters]
        news_results = [] if not news_adapters else [results_map.get(a.name, []) for a in news_adapters]

        return papers_results, resources_results, news_results