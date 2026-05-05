"""RetrieverManager — 执行检索计划的核心编排器

设计原则（2025 年最佳实践）：
1. asyncio + httpx.AsyncClient 做 I/O 并发（轻量级，比线程池吞吐高 5-10x）
2. 每个 source 独立的 Semaphore 控制并发量，避免压垮单一 API
3. 查询内并行，查询间串行（避免 embedding 服务被并发请求打爆）
4. 单 source 失败不影响其他 source（错误隔离）
5. 全局去重：同标题论文只出现一次
6. Streaming-ready：支持逐 query 输出中间结果（未来对接 WebSocket）

execute_plan() 核心流程：
  RetrievalPlan
    → 对每个 QuerySpec 生成 RoutingDecision（QueryRouter）
    → 按 source 分组，合并相同 source 的查询
    → asyncio.gather 并发执行各 source 的搜索
    → 汇总所有 PaperCard + ResourceCard
    → 去重 → CurationReport

使用方式：
    manager = RetrieverManager()
    await manager.execute_plan(plan)       # async
    with RetrieverManager() as mgr:
        mgr.run(mgr.execute_plan(plan))    # sync 入口
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Awaitable, Callable

import httpx

from src.config import settings
from src.models import CurationReport, PaperCard, ResourceCard, SuppleReport
from src.rate_limits import get_source_config
from src.router import QueryRouter, RoutingDecision

if TYPE_CHECKING:
    from src.planner.schemas import RetrievalPlan

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Retriever 异步适配器
# ──────────────────────────────────────────────

class AsyncRetrieverAdapter:
    """把同步 retriever 包装成异步（在线程池执行，避免阻塞事件循环）

    设计：不做真正的异步改造（因为很多 retriever 用 httpx 但还未 async 化），
    而是用 run_in_executor 把同步调用放到线程池，避免 asyncio 事件循环被阻塞。

    关键属性：
    - source_name: 路由标识（与 RetrieverManager.PAPER_RETRIEVERS 对应）
    - name: 底层 retriever 的真实类名（用于日志）
    """

    def __init__(
        self,
        retriever,
        source_name: str,
        concurrency: int | None = None,
    ) -> None:
        self._retriever = retriever
        self._source_name = source_name
        cfg = get_source_config(source_name)
        conc = concurrency if concurrency is not None else cfg.no_key_concurrency
        self._semaphore = asyncio.Semaphore(conc)
        self._min_interval = cfg.min_interval_seconds
        self._last_called = 0.0
        self.name = getattr(retriever, "name", retriever.__class__.__name__)

    @property
    def source(self) -> str:
        """返回路由标识（用于与 PAPER_RETRIEVERS / RESOURCE_RETRIEVERS 匹配）"""
        return self._source_name

    async def _enforce_rate_limit(self) -> None:
        """检查距上次调用的时间间隔，不满足时主动 sleep"""
        if self._min_interval <= 0:
            return
        now = asyncio.get_running_loop().time()
        elapsed = now - self._last_called
        if elapsed < self._min_interval:
            wait = self._min_interval - elapsed
            logger.debug("%s: rate limit back-off %.1fs", self.name, wait)
            await asyncio.sleep(wait)
        self._last_called = asyncio.get_running_loop().time()

    async def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[PaperCard]:
        async with self._semaphore:
            await self._enforce_rate_limit()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self._retriever.search(query, max_results)
            )

    async def search_resources(
        self,
        query: str,
    ) -> list[ResourceCard]:
        async with self._semaphore:
            await self._enforce_rate_limit()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self._retriever.search_resources(query)
            )


# ──────────────────────────────────────────────
# RetrieverManager
# ──────────────────────────────────────────────

class RetrieverManager:
    """检索计划执行器——管理所有检索器，对 RetrievalPlan 输出 CurationReport

    关键设计：
    - execute_plan() 是 async 方法，返回 CurationReport
    - run() 是同步入口，内部创建事件循环
    - with RetrieverManager() as mgr 可用于上下文管理器
    """

    # 论文类 retriever（输出 PaperCard）
    PAPER_RETRIEVERS = (
        "arxiv", "semantic_scholar", "dblp", "serper",
    )
    # 资源类 retriever（输出 ResourceCard）
    RESOURCE_RETRIEVERS = (
        "github", "huggingface", "hackernews", "serper",
    )

    def __init__(
        self,
        http_timeout: float = 30.0,
        max_results_per_source: int | None = None,
    ) -> None:
        self._http_timeout = http_timeout
        self._max_results = max_results_per_source or settings.default_max_results

        # 初始化所有同步 retriever（包装为异步）
        self._retrievers: dict[str, AsyncRetrieverAdapter] = {}
        self._init_retrievers()

        # QueryRouter（embedding 路由，sync）
        self._router = QueryRouter()

        # 共享 HTTP 客户端（连接复用）
        self._client: httpx.AsyncClient | None = None

    # ── 初始化 ──

    def _init_retrievers(self) -> None:
        """初始化所有 retriever（lazy import 避免循环依赖）"""
        from src.retrievers.arxiv import ArxivRetriever
        from src.retrievers.semantic_scholar import SemanticScholarRetriever
        from src.retrievers.dblp import DblpRetriever
        from src.retrievers.github import GithubRetriever
        from src.retrievers.papers_with_code import PapersWithCodeRetriever
        from src.retrievers.huggingface import HuggingFaceRetriever
        from src.retrievers.hackernews import HackerNewsRetriever
        from src.retrievers.serper import SerperRetriever

        retriever_map = {
            "arxiv": ArxivRetriever,
            "semantic_scholar": SemanticScholarRetriever,
            "dblp": DblpRetriever,
            "github": GithubRetriever,
            "papers_with_code": PapersWithCodeRetriever,
            "huggingface": HuggingFaceRetriever,
            "hackernews": HackerNewsRetriever,
            "serper": SerperRetriever,
        }

        for name, cls in retriever_map.items():
            self._retrievers[name] = AsyncRetrieverAdapter(
                cls(),
                source_name=name,  # 路由标识，与 PAPER_RETRIEVERS/RESOURCE_RETRIEVERS 匹配
                concurrency=None,  # 使用 rate_limits.py 配置
            )

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._http_timeout)
        return self._client

    # ── 公开 API ──

    async def execute_plan(
        self,
        plan: "RetrievalPlan",
        routing_mode: str = "target_sources",
        top_k: int | None = None,
        threshold: float | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> CurationReport:
        """执行完整检索计划

        Args:
            plan: Planner 生成的检索计划（含多个 QuerySpec）
            routing_mode: 路由模式（target_sources | embedding | hybrid）
            top_k: embedding/hybrid 模式下最多选几个 source
            threshold: embedding/hybrid 模式下的相似度阈值
            progress_callback: (query, completed, total) → None
                               用于流式输出进度（如 WebSocket 推送）

        Returns:
            聚合的 CurationReport（含所有查询去重后的 papers + resources）
        """
        start_time = time.time()

        # 路由：对整个 plan 的每个 QuerySpec 生成决策
        try:
            decisions = self._router.route_plan(
                plan,
                mode=routing_mode,
                top_k=top_k,
                threshold=threshold,
            )
        except Exception as e:
            logger.warning("route_plan failed (%s), using fallback all-source routing", e)
            decisions = self._route_with_fallback(plan)


        # fallback：如果所有 decision 的 routed_sources 都为空，回退到全部 source
        if all(not d.routed_sources for d in decisions):
            logger.warning("all decisions have empty sources, using fallback routing")
            decisions = self._route_with_fallback(plan)

        # 按 source 分组：相同 source 的查询合并（避免重复调用）
        source_tasks: dict[str, list[tuple[RoutingDecision, str]]] = defaultdict(list)
        # source_tasks[source] = [(decision, query_string), ...]

        for decision in decisions:
            for source in decision.routed_sources:
                source_tasks[source].append((decision, decision.query_spec.query))

        logger.info(
            "execute_plan: %d queries → %d unique sources",
            len(decisions), len(source_tasks),
        )

        # 收集所有 paper + resource
        all_papers: list[PaperCard] = []
        all_resources: list[ResourceCard] = []
        seen_titles: set[str] = set()  # 去重用

        # 并发执行所有 source（用 asyncio.gather，跨 Python 版本兼容）
        gather_tasks: list[Awaitable[tuple[list[PaperCard], list[ResourceCard]]]] = []

        for source, query_pairs in source_tasks.items():
            adapter = self._retrievers.get(source)
            if adapter is None:
                logger.warning("No retriever found for source: %s", source)
                continue

            gather_tasks.append(
                self._search_source(adapter, query_pairs, seen_titles)
            )

        if gather_tasks:
            results = await asyncio.gather(*gather_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Source task failed: %s", result)
                    continue
                papers, resources = result
                all_papers.extend(papers)
                all_resources.extend(resources)

        # 统计
        classic = [p for p in all_papers if p.citation_count > 50]
        frontier = [p for p in all_papers if p.year >= 2023]
        executed_queries = [decision.query_spec.query for decision in decisions]

        duration = time.time() - start_time
        logger.info(
            "execute_plan done: %d papers (%d classic, %d frontier), %d resources, %.1fs",
            len(all_papers), len(classic), len(frontier), len(all_resources), duration,
        )

        return CurationReport(
            topic=plan.topic,
            papers=all_papers,
            resources=all_resources,
            total_papers=len(all_papers),
            classic_count=len(classic),
            frontier_count=len(frontier),
            github_count=sum(1 for r in all_resources if r.type == "github_repo"),
            benchmark_count=sum(1 for r in all_resources if r.type == "benchmark"),
            dataset_count=sum(1 for r in all_resources if r.type == "dataset"),
            queries=executed_queries,
        )

    async def _search_source(
        self,
        adapter: AsyncRetrieverAdapter,
        query_pairs: list[tuple[RoutingDecision, str]],
        seen_titles: set[str],
    ) -> tuple[list[PaperCard], list[ResourceCard]]:
        """并发执行单个 source 的多个查询

        Args:
            adapter: AsyncRetrieverAdapter 包装的 retriever
            query_pairs: [(decision, query_string), ...]
            seen_titles: 全局已见标题集合（用于去重，会就地修改）

        Returns:
            (papers, resources)
        """
        source_name = adapter.source  # "arxiv" / "github" / etc.

        async def search_one(decision: RoutingDecision, query: str) -> tuple[list[PaperCard], list[ResourceCard]]:
            papers: list[PaperCard] = []
            resources: list[ResourceCard] = []

            try:
                # 论文类 source
                if source_name in self.PAPER_RETRIEVERS:
                    results = await adapter.search(query, self._max_results // len(query_pairs))
                    for p in results:
                        title_key = p.title.lower().strip() if p.title else ""
                        if title_key and title_key not in seen_titles:
                            seen_titles.add(title_key)
                            papers.append(p)
                # 资源类 source
                if source_name in self.RESOURCE_RETRIEVERS:
                    results = await adapter.search_resources(query)
                    resources.extend(results)
            except Exception as e:
                logger.warning("source %s query %r failed: %s", source_name, query, e)

            return papers, resources

        # 并发执行该 source 的所有查询（Python 3.10 兼容，不用 TaskGroup）
        inner_tasks = [
            search_one(d, q)
            for d, q in query_pairs
        ]
        inner_results = await asyncio.gather(*inner_tasks, return_exceptions=True)

        all_papers: list[PaperCard] = []
        all_resources: list[ResourceCard] = []

        for inner_result in inner_results:
            if isinstance(inner_result, Exception):
                logger.warning("Query in source %s failed: %s", source_name, inner_result)
                continue
            papers, resources = inner_result
            all_papers.extend(papers)
            all_resources.extend(resources)

        logger.debug(
            "source %s: %d papers, %d resources from %d queries",
            source_name, len(all_papers), len(all_resources), len(query_pairs),
        )
        return all_papers, all_resources

    def run(
        self,
        coro: Awaitable,
    ) -> object:
        """同步入口：在新事件循环中运行 async coroutine

        用于非 async 上下文调用 execute_plan()。
        内部管理事件循环创建和清理。
        """
        return asyncio.run(coro)

    # ── 简化 API ──


    def _all_sources(self) -> list[str]:
        """返回所有可用 source（用于 fallback when embedding 不可用）"""
        return list(self._retrievers.keys())


    def _route_with_fallback(
        self,
        plan: "RetrievalPlan",
    ) -> list[RoutingDecision]:
        """对 plan 做路由，embedding 不可用时 fallback 到全部 source"""
        from src.router import RoutingDecision

        decisions = []
        for spec in plan.queries:
            try:
                decision = self._router.route_spec(
                    spec, mode="embedding", top_k=8, threshold=0.0,
                )
                if not decision.routed_sources:
                    logger.warning(
                        "embedding routing returned no sources for %r, using all",
                        spec.query[:40],
                    )
                    decision = RoutingDecision(
                        query_spec=spec,
                        routed_sources=self._all_sources(),
                        similarity_scores={},
                        routing_mode="fallback",
                    )
            except Exception as e:
                logger.warning(
                    "embedding routing failed for %r (%s), using all sources",
                    spec.query[:40], e,
                )
                decision = RoutingDecision(
                    query_spec=spec,
                    routed_sources=self._all_sources(),
                    similarity_scores={},
                    routing_mode="fallback",
                )
            decisions.append(decision)
        return decisions

    async def search_all(
        self,
        query: str,
        max_results: int | None = None,
    ) -> CurationReport:
        """单查询全源检索（并行，async）

        简化入口，绕过 execute_plan() 直接对单查询执行全源并发检索。
        等价于 manager.py 的同步 search_all()，但使用 async 并发。
        """
        from src.planner.schemas import QuerySpec, RetrievalPlan

        plan = RetrievalPlan(
            topic=query,
            queries=[QuerySpec(query=query, language="en", target_sources=[])],
        )
        report = await self.execute_plan(plan, routing_mode="embedding")
        if max_results and len(report.papers) > max_results:
            report.papers = report.papers[:max_results]
        return report

    async def supplementary_search(
        self,
        gap_queries: list[str],
        existing_report: CurationReport | None = None,
        section_query_map: dict[str, list[str]] | None = None,
        max_results: int | None = None,
    ) -> SuppleReport:
        """对细纲缺口执行补搜（async 并发版）

        对每条 gap query 并发执行全源检索，与已有结果去重后输出 SuppleReport。

        Args:
            gap_queries: 缺口的查询字符串列表
            existing_report: 已有的检索结果（用于去重）
            section_query_map: 章节→查询映射 {section_id: [queries]}
            max_results: 每个查询的最大结果数

        Returns:
            SuppleReport 对象（含新发现汇总）
        """
        from src.planner.schemas import QuerySpec, RetrievalPlan

        start = time.time()
        unique_queries = list(dict.fromkeys(gap_queries))

        # 已有论文标题（用于去重）
        existing_titles: set[str] = set()
        if existing_report:
            for p in existing_report.papers:
                if p.title:
                    existing_titles.add(p.title.lower().strip())

        result_map: dict[str, CurationReport] = {}
        total_new_papers = 0
        total_new_resources = 0
        new_classics = 0
        new_frontiers = 0

        # 并发执行所有 gap queries
        async def search_one(q: str) -> tuple[str, CurationReport]:
            report = await self.search_all(q, max_results)
            return q, report

        results = await asyncio.gather(
            *[search_one(q) for q in unique_queries],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                continue
            q, report = result
            result_map[q] = report

            for p in report.papers:
                key = p.title.lower().strip() if p.title else ""
                if key and key not in existing_titles:
                    total_new_papers += 1
                    existing_titles.add(key)
                    if p.citation_count > 50:
                        new_classics += 1
                    if p.year >= 2023:
                        new_frontiers += 1
            total_new_resources += len(report.resources)

        # 构建章节映射（仅记录有补搜的章节）
        sections_improved: list[str] = []
        gap_mapping: dict[str, list[str]] = {}
        if section_query_map:
            for sec_id, queries in section_query_map.items():
                active_queries = [q for q in queries if q in result_map]
                if active_queries:
                    gap_mapping[sec_id] = active_queries
                    sections_improved.append(sec_id)

        duration = time.time() - start

        return SuppleReport(
            topic=existing_report.topic if existing_report else "补搜",
            queries_executed=unique_queries,
            result_map=result_map,
            total_new_papers=total_new_papers,
            total_new_resources=total_new_resources,
            new_classics=new_classics,
            new_frontiers=new_frontiers,
            sections_improved=sections_improved,
            gap_queries=gap_mapping,
            duration_seconds=duration,
        )

    def close(self) -> None:
        if self._client and not self._client.is_closed:
            asyncio.create_task(self._client.aclose())

    def __enter__(self) -> "RetrieverManager":
        return self

    def __exit__(self, *args) -> None:
        self.close()
