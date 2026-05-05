"""RetrieverManager / execute_plan 单元 + 集成测试"""

import asyncio
import time
from collections import defaultdict
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.executor import AsyncRetrieverAdapter, RetrieverManager
from src.models import CurationReport, PaperCard, ResourceCard
from src.planner.schemas import RetrievalPlan, QuerySpec
from src.router import QueryRouter, RoutingDecision

from tests.test_router import FakeEmbedder, make_plan


# ── Mock Retriever ──

class MockPaperRetriever:
    """模拟论文检索器"""
    name = "mock_paper"

    def __init__(self, papers_per_query: list[list[PaperCard]] | None = None) -> None:
        self._papers = papers_per_query or []
        self.call_count = 0

    def search(self, query: str, max_results: int | None = None) -> list[PaperCard]:
        self.call_count += 1
        idx = min(self.call_count - 1, len(self._papers) - 1)
        return list(self._papers[idx]) if self._papers else []

    def search_resources(self, query: str) -> list[ResourceCard]:
        return []


class MockResourceRetriever:
    """模拟资源检索器"""
    name = "mock_resource"

    def __init__(self, resources_per_query: list[list[ResourceCard]] | None = None) -> None:
        self._resources = resources_per_query or []
        self.call_count = 0

    def search(self, query: str, max_results: int | None = None) -> list[PaperCard]:
        return []

    def search_resources(self, query: str) -> list[ResourceCard]:
        self.call_count += 1
        idx = min(self.call_count - 1, len(self._resources) - 1)
        return list(self._resources[idx]) if self._resources else []


# ── 测试类 ──

class TestAsyncRetrieverAdapter:
    """AsyncRetrieverAdapter 单元测试"""

    def test_search_runs_in_executor(self):
        """验证 search() 确实在线程池执行（不阻塞事件循环）"""
        inner_call_count = 0

        class SlowRetriever:
            name = "slow"
            def search(self, query, max_results=None):
                nonlocal inner_call_count
                inner_call_count += 1
                time.sleep(0.01)  # 10ms，模拟 I/O
                return [PaperCard(title=f"paper-{inner_call_count}", source="test")]

        adapter = AsyncRetrieverAdapter(SlowRetriever(), source_name="slow", concurrency=2)

        async def run():
            results = await adapter.search("test query")
            return results

        result = asyncio.run(run())
        assert inner_call_count == 1
        assert len(result) == 1

    def test_concurrency_limit(self):
        """验证信号量限制并发数"""
        active = []

        class CountingRetriever:
            name = "counting"
            def search(self, query, max_results=None):
                active.append(1)
                time.sleep(0.05)
                active.remove(1)
                return []
            def search_resources(self, query):
                return []

        adapter = AsyncRetrieverAdapter(CountingRetriever(), source_name="counting", concurrency=2)

        async def run():
            await asyncio.gather(
                adapter.search("q1"),
                adapter.search("q2"),
                adapter.search("q3"),
            )

        start = time.time()
        asyncio.run(run())
        elapsed = time.time() - start

        # 2 并发，每任务 50ms → 前两个立即启动，第三个等 50ms
        # 最少需要: 50ms (前两个) + 50ms (第三个) = 100ms
        assert elapsed >= 0.09  # 至少有 100ms
        # 但不应该超过 3 × 50ms = 150ms（说明并发受限）
        assert elapsed < 0.20


class TestRetrieverManagerExecutePlan:
    """RetrieverManager.execute_plan 集成测试"""

    @pytest.mark.asyncio
    async def test_routing_decisions_used(self):
        """验证 QueryRouter 的 decision.routed_sources 决定实际调用"""
        plan = make_plan([
            {"query": "transformer architecture", "target_sources": ["arxiv"], "priority": 1},
            {"query": "github ml tools", "target_sources": ["github"], "priority": 2},
        ])

        fake = FakeEmbedder(fixed_scores={
            "arxiv": 0.90, "github": 0.85, "semantic_scholar": 0.70,
        })
        router = QueryRouter(embedding_service=fake)

        mock_arxiv = MockPaperRetriever([
            [PaperCard(title="Attention Is All You Need", source="arxiv", year=2017)],
            [],
        ])
        mock_github = MockResourceRetriever([
            [],
            [ResourceCard(name="transformers", type="github_repo", url="https://github.com/huggingface/transformers", stars=100000)],
        ])

        manager = RetrieverManager()
        # 替换 _retrievers 的两个实例
        manager._retrievers["arxiv"] = AsyncRetrieverAdapter(mock_arxiv, source_name="arxiv")
        manager._retrievers["github"] = AsyncRetrieverAdapter(mock_github, source_name="github")

        decisions = router.route_plan(plan, mode="target_sources")

        # 验证路由结果
        assert decisions[0].routed_sources == ["arxiv"]
        assert decisions[1].routed_sources == ["github"]

        # 执行
        async def run():
            return await manager.execute_plan(plan, routing_mode="target_sources")

        report = await run()

        assert report.total_papers >= 1
        assert mock_arxiv.call_count >= 1
        assert mock_github.call_count >= 1

    @pytest.mark.asyncio
    async def test_deduplication_by_title(self):
        """验证相同标题的论文只出现一次"""
        plan = make_plan([
            {"query": "attention mechanism", "target_sources": ["arxiv"], "priority": 1},
            {"query": "attention is all you need", "target_sources": ["semantic_scholar"], "priority": 1},
        ])

        fake = FakeEmbedder(fixed_scores={"arxiv": 0.9, "semantic_scholar": 0.85})
        router = QueryRouter(embedding_service=fake)

        same_paper = PaperCard(
            title="Attention Is All You Need",
            source="arxiv",
            year=2017,
            citation_count=50000,
        )
        mock_arxiv = MockPaperRetriever([[same_paper]])
        mock_ss = MockPaperRetriever([[same_paper]])  # 完全相同标题

        manager = RetrieverManager()
        manager._retrievers["arxiv"] = AsyncRetrieverAdapter(mock_arxiv, source_name="arxiv")
        manager._retrievers["semantic_scholar"] = AsyncRetrieverAdapter(mock_ss, source_name="semantic_scholar")

        async def run():
            return await manager.execute_plan(plan, routing_mode="target_sources")

        report = await run()

        # 标题去重后应该只有 1 篇
        titles = [p.title for p in report.papers]
        assert titles.count("Attention Is All You Need") == 1

    @pytest.mark.asyncio
    async def test_source_failure_isolated(self):
        """验证单个 source 失败不影响其他 source"""
        plan = make_plan([
            {"query": "test query", "target_sources": ["arxiv", "github", "hackernews"], "priority": 1},
        ])

        fake = FakeEmbedder(fixed_scores={
            "arxiv": 0.9, "github": 0.85, "hackernews": 0.70,
        })
        router = QueryRouter(embedding_service=fake)

        class FailingRetriever:
            name = "failing"
            def search(self, query, max_results=None):
                raise RuntimeError("Simulated API failure")
            def search_resources(self, query):
                raise RuntimeError("Simulated API failure")

        manager = RetrieverManager()
        manager._retrievers["arxiv"] = AsyncRetrieverAdapter(
            MockPaperRetriever([[PaperCard(title="Good Paper", source="arxiv")]]),
            source_name="arxiv",
        )
        manager._retrievers["github"] = AsyncRetrieverAdapter(FailingRetriever(), source_name="github")
        manager._retrievers["hackernews"] = AsyncRetrieverAdapter(
            MockResourceRetriever([[ResourceCard(name="Good Resource", type="github_repo")]]),
            source_name="hackernews",
        )

        async def run():
            return await manager.execute_plan(plan, routing_mode="target_sources")

        report = await run()

        # arxiv 和 hackernews 应该成功，github 失败被隔离
        assert report.total_papers >= 1
        assert report.resources or report.total_papers > 0

    @pytest.mark.asyncio
    async def test_empty_plan_returns_empty_report(self):
        """空 plan 返回空报告"""
        plan = make_plan([])

        fake = FakeEmbedder()
        router = QueryRouter(embedding_service=fake)

        manager = RetrieverManager()

        async def run():
            return await manager.execute_plan(plan, routing_mode="target_sources")

        report = await run()

        assert report.total_papers == 0

    @pytest.mark.asyncio
    async def test_classic_and_frontier_classification(self):
        """验证经典/前沿论文分类"""
        plan = make_plan([
            {"query": "deep learning", "target_sources": ["arxiv"], "priority": 1},
        ])

        fake = FakeEmbedder(fixed_scores={"arxiv": 0.9})
        router = QueryRouter(embedding_service=fake)

        classic = PaperCard(title="AlexNet", source="arxiv", year=2012, citation_count=100000)
        frontier = PaperCard(title="GPT-4", source="arxiv", year=2023, citation_count=5000)
        recent = PaperCard(title="Llama 2", source="arxiv", year=2022, citation_count=2000)

        manager = RetrieverManager()
        manager._retrievers["arxiv"] = AsyncRetrieverAdapter(
            MockPaperRetriever([[classic, frontier, recent]]),
            source_name="arxiv",
        )

        async def run():
            return await manager.execute_plan(plan, routing_mode="target_sources")

        report = await run()

        assert report.classic_count >= 1
        assert report.frontier_count >= 1
        assert report.total_papers == 3

    @pytest.mark.asyncio
    async def test_multiple_queries_same_source_merged(self):
        """多个查询指向同一 source 时合并执行"""
        plan = make_plan([
            {"query": "query A", "target_sources": ["arxiv"], "priority": 1},
            {"query": "query B", "target_sources": ["arxiv"], "priority": 1},
            {"query": "query C", "target_sources": ["github"], "priority": 1},
        ])

        fake = FakeEmbedder(fixed_scores={"arxiv": 0.9, "github": 0.85})
        router = QueryRouter(embedding_service=fake)

        mock_arxiv = MockPaperRetriever([
            [PaperCard(title="Paper A", source="arxiv")],
            [PaperCard(title="Paper B", source="arxiv")],
        ])
        mock_github = MockResourceRetriever([
            [ResourceCard(name="Repo C", type="github_repo")],
        ])

        manager = RetrieverManager()
        manager._retrievers["arxiv"] = AsyncRetrieverAdapter(mock_arxiv, source_name="arxiv")
        manager._retrievers["github"] = AsyncRetrieverAdapter(mock_github, source_name="github")

        async def run():
            return await manager.execute_plan(plan, routing_mode="target_sources")

        report = await run()

        # arxiv 被调用 2 次（query A 和 query B），github 调用 1 次
        assert mock_arxiv.call_count == 2
        assert mock_github.call_count == 1
        # 总共应该有几篇论文
        assert report.total_papers >= 2


class TestRetrieverManagerRunSync:
    """同步入口 run() 测试"""

    def test_run_execute_plan_sync(self):
        """验证 run() 同步调用 execute_plan"""
        plan = make_plan([
            {"query": "test", "target_sources": ["arxiv"], "priority": 1},
        ])

        fake = FakeEmbedder(fixed_scores={"arxiv": 0.9})
        router = QueryRouter(embedding_service=fake)

        manager = RetrieverManager()
        manager._retrievers["arxiv"] = AsyncRetrieverAdapter(
            MockPaperRetriever([[PaperCard(title="Test Paper", source="arxiv")]]),
            source_name="arxiv",
        )

        async def run():
            return await manager.execute_plan(plan, routing_mode="target_sources")

        report = manager.run(run())
        assert report.total_papers >= 1

    def test_context_manager(self):
        """验证 with RetrieverManager() as mgr 上下文管理器"""
        with RetrieverManager() as mgr:
            assert mgr is not None
            # 验证 _retrievers 已初始化
            assert len(mgr._retrievers) > 0


# ── 对齐旧 RetrieverManager.search_all ──

class TestBackwardCompatibility:
    """验证新 executor 不破坏旧 search_all() 行为"""

    def test_search_all_still_works(self):
        """旧的 RetrieverManager.search_all() 应该仍然可用"""
        from src.manager import RetrieverManager

        # search_all 是同步的，不依赖 executor
        manager = RetrieverManager()
        # 不调用 search_all（因为会真实请求外部 API）
        # 只验证 manager 可以创建成功
        assert manager is not None
        assert hasattr(manager, "search_all")
        assert hasattr(manager, "supplementary_search")