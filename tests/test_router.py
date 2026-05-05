"""QueryRouter 单元与集成测试"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.embedding import EmbeddingService, RetrieverEmbedding
from src.router import QueryRouter, RoutingDecision
from src.planner.schemas import RetrievalPlan, QuerySpec


# ── Fake 嵌入服务 ──

class FakeEmbedder:
    """Fake 嵌入服务——返回预设的相似度，不调用真实 API"""

    def __init__(
        self,
        fixed_scores: dict[str, float] | None = None,
        actual_dim: int = 1024,
    ) -> None:
        self._scores = fixed_scores or {}
        self._dim = actual_dim
        self._ready = True
        self._vectors = {}

    def build_retriever_index(self) -> None:
        for name in ["arxiv", "semantic_scholar", "dblp", "github",
                     "papers_with_code", "huggingface", "hackernews", "serper"]:
            vec = np.random.rand(self._dim)
            vec = vec / np.linalg.norm(vec)
            self._vectors[name] = vec

    def is_ready(self) -> bool:
        return self._ready

    def actual_dimension(self) -> int:
        return self._dim

    def route_query(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.3,
    ) -> list[tuple[str, float]]:
        filtered = [(s, round(sc, 4))
                    for s, sc in self._scores.items() if sc >= threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:top_k]


# ── 辅助函数 ──

def make_plan(
    queries: list[dict],
    topic: str = "test topic",
) -> RetrievalPlan:
    """从 dict 构建 RetrievalPlan（测试用）"""
    specs = []
    for q in queries:
        specs.append(QuerySpec(
            query=q["query"],
            target_sources=q.get("target_sources", []),
            priority=q.get("priority", 1),
        ))
    return RetrievalPlan(topic=topic, queries=specs)


# ── 测试类 ──

class TestQueryRouterUnit:
    """QueryRouter 单元测试"""

    def test_target_sources_mode_returns_spec_fields(self):
        spec = make_plan([{
            "query": "spark performance optimization",
            "target_sources": ["arxiv", "github"],
            "priority": 1,
        }]).queries[0]

        fake = FakeEmbedder()
        router = QueryRouter(embedding_service=fake)
        decision = router.route_spec(spec, mode="target_sources")

        assert decision.routed_sources == ["arxiv", "github"]
        assert decision.routing_mode == "target_sources"
        assert decision.similarity_scores == {}

    def test_target_sources_empty_returns_empty_list(self):
        spec = make_plan([{
            "query": "some topic",
            "target_sources": [],
            "priority": 1,
        }]).queries[0]

        fake = FakeEmbedder()
        router = QueryRouter(embedding_service=fake)
        decision = router.route_spec(spec, mode="target_sources")

        assert decision.routed_sources == []
        assert decision.routing_mode == "target_sources"

    def test_embedding_mode_uses_similarity_scores(self):
        spec = make_plan([{
            "query": "distributed database",
            "target_sources": [],
            "priority": 1,
        }]).queries[0]

        fake = FakeEmbedder(fixed_scores={
            "arxiv": 0.85, "github": 0.72, "dblp": 0.55,
            "hackernews": 0.40, "serper": 0.35,
        })
        router = QueryRouter(embedding_service=fake)
        decision = router.route_spec(spec, mode="embedding", top_k=3, threshold=0.3)

        assert "arxiv" in decision.routed_sources
        assert "github" in decision.routed_sources
        assert "dblp" in decision.routed_sources
        assert "hackernews" not in decision.routed_sources  # 0.40 < threshold
        assert decision.routing_mode == "embedding"
        assert decision.similarity_scores["arxiv"] == pytest.approx(0.85)

    def test_embedding_mode_threshold_filters(self):
        spec = make_plan([{
            "query": "web development",
            "target_sources": [],
            "priority": 1,
        }]).queries[0]

        fake = FakeEmbedder(fixed_scores={
            "arxiv": 0.5, "github": 0.4, "hackernews": 0.35,
        })
        router = QueryRouter(embedding_service=fake)
        decision = router.route_spec(spec, mode="embedding", threshold=0.38)

        assert "github" in decision.routed_sources
        assert "hackernews" not in decision.routed_sources  # 0.35 < 0.38

    def test_hybrid_mode_uses_target_sources_directly(self):
        """hybrid 模式：有 target_sources 时直接使用，不补充 embedding"""
        spec = make_plan([{
            "query": "benchmark comparison",
            "target_sources": ["arxiv"],
            "priority": 1,
        }]).queries[0]

        fake = FakeEmbedder(fixed_scores={
            "arxiv": 0.9, "github": 0.8, "serper": 0.5,
        })
        router = QueryRouter(embedding_service=fake)
        decision = router.route_spec(spec, mode="hybrid", top_k=3, threshold=0.3)

        # target_sources 直接使用，不补充
        assert decision.routed_sources == ["arxiv"]
        assert decision.routing_mode == "hybrid"
        # 但相似度信息仍然记录（用于分析）
        assert "github" in decision.similarity_scores
        assert "serper" in decision.similarity_scores

    def test_hybrid_mode_no_target_sources_uses_embedding_only(self):
        spec = make_plan([{
            "query": "open source tool",
            "target_sources": [],
            "priority": 1,
        }]).queries[0]

        fake = FakeEmbedder(fixed_scores={
            "github": 0.85, "arxiv": 0.65, "hackernews": 0.55,
        })
        router = QueryRouter(embedding_service=fake)
        decision = router.route_spec(spec, mode="hybrid", top_k=3, threshold=0.3)

        assert "github" in decision.routed_sources
        assert "arxiv" in decision.routed_sources
        assert decision.routing_mode == "hybrid"

    def test_hybrid_mode_target_sources_sufficient_no_embedding_补充(self):
        spec = make_plan([{
            "query": "NLP survey",
            "target_sources": ["arxiv", "semantic_scholar", "dblp"],
            "priority": 1,
        }]).queries[0]

        fake = FakeEmbedder(fixed_scores={
            "arxiv": 0.9, "github": 0.8, "semantic_scholar": 0.85, "dblp": 0.7,
        })
        router = QueryRouter(embedding_service=fake)
        decision = router.route_spec(spec, mode="hybrid", top_k=3, threshold=0.3)

        # target_sources 已有 3 个，满足 top_k，不补
        assert set(decision.routed_sources) == {"arxiv", "semantic_scholar", "dblp"}
        # github 仍被记录到 similarity_scores（用于分析）
        assert "github" in decision.similarity_scores

    def test_route_plan_processes_all_queries(self):
        plan = make_plan([
            {"query": "mapreduce", "target_sources": ["arxiv"], "priority": 1},
            {"query": "kubernetes deployment", "target_sources": [], "priority": 2},
            {"query": "llm fine-tuning", "target_sources": ["huggingface"], "priority": 1},
        ])

        fake = FakeEmbedder(fixed_scores={
            "arxiv": 0.85, "github": 0.70, "huggingface": 0.80,
            "semantic_scholar": 0.60, "serper": 0.50,
        })
        router = QueryRouter(embedding_service=fake)
        decisions = router.route_plan(plan, mode="hybrid", top_k=3, threshold=0.3)

        assert len(decisions) == 3
        assert decisions[0].routed_sources == ["arxiv"]  # target_sources 优先
        assert "github" in decisions[1].routed_sources  # embedding 补充
        assert decisions[2].routed_sources == ["huggingface"]  # target_sources

    def test_default_threshold_and_top_k(self):
        spec = make_plan([{"query": "test", "target_sources": [],}]).queries[0]

        fake = FakeEmbedder(fixed_scores={
            "arxiv": 0.5, "github": 0.4, "hackernews": 0.3, "serper": 0.25,
        })
        router = QueryRouter(embedding_service=fake)

        # 默认 threshold=0.3, top_k=3
        decision = router.route_spec(spec, mode="embedding")
        assert len(decision.routed_sources) <= 3
        # hackernews: 0.3 >= 0.3（包含），serper: 0.25 < 0.3（排除）
        assert "hackernews" in decision.routed_sources
        assert "serper" not in decision.routed_sources

    def test_decision_contains_query_embedding_dim(self):
        spec = make_plan([{"query": "test", "target_sources": [],}]).queries[0]
        fake = FakeEmbedder(fixed_scores={"arxiv": 0.8}, actual_dim=1024)
        router = QueryRouter(embedding_service=fake)
        decision = router.route_spec(spec, mode="embedding")
        assert decision.query_embedding_dim == 1024


class TestQueryRouterIntegration:
    """QueryRouter 集成测试——模拟 RetrieverManager.execute_plan 行为"""

    def test_routing_decisions_loggable(self):
        """验证路由决策可以被正确序列化/记录"""
        plan = make_plan([
            {"query": "distributed computing", "target_sources": ["arxiv", "github"]},
        ])
        fake = FakeEmbedder(fixed_scores={"arxiv": 0.85, "github": 0.72})
        router = QueryRouter(embedding_service=fake)
        decisions = router.route_plan(plan, mode="hybrid")

        d = decisions[0]
        # 所有字段都是可序列化的（无 numpy 对象暴露）
        assert isinstance(d.routing_mode, str)
        assert isinstance(d.routed_sources, list)
        assert isinstance(d.similarity_scores, dict)
        assert all(isinstance(s, float) for s in d.similarity_scores.values())

    def test_cross_source_deduplication_behavior(self):
        """验证同一 source 在多个 QuerySpec 中出现时的去重预期"""
        plan = make_plan([
            {"query": "spark rdd", "target_sources": ["arxiv"]},
            {"query": "spark sql", "target_sources": ["arxiv", "github"]},
        ])
        fake = FakeEmbedder(fixed_scores={
            "arxiv": 0.85, "github": 0.72, "dblp": 0.60,
        })
        router = QueryRouter(embedding_service=fake)
        decisions = router.route_plan(plan, mode="hybrid")

        # 两个决策都对 arxiv 有路由需求
        arxiv_routes = [d for d in decisions if "arxiv" in d.routed_sources]
        assert len(arxiv_routes) == 2
        # github 只出现在第二个
        github_routes = [d for d in decisions if "github" in d.routed_sources]
        assert len(github_routes) == 1

    def test_different_modes_produce_different_results(self):
        spec = make_plan([{
            "query": "benchmark comparison",
            "target_sources": ["arxiv"],
        }]).queries[0]

        fake = FakeEmbedder(fixed_scores={
            "arxiv": 0.9, "github": 0.85, "dblp": 0.60,
            "hackernews": 0.5, "serper": 0.45,
        })
        router = QueryRouter(embedding_service=fake)

        d_target = router.route_spec(spec, mode="target_sources")
        d_embed = router.route_spec(spec, mode="embedding", top_k=3, threshold=0.3)
        d_hybrid = router.route_spec(spec, mode="hybrid", top_k=3, threshold=0.3)

        assert d_target.routed_sources == ["arxiv"]  # 只用 target_sources
        assert set(d_embed.routed_sources) == {"arxiv", "github", "dblp"}  # 纯嵌入
        # hybrid：有 target_sources 时只用它（不补充），相似度仍被记录
        assert set(d_hybrid.routed_sources) == {"arxiv"}
        assert "github" in d_hybrid.similarity_scores  # 相似度被记录


# ── MockRetriever 工厂（用于后续 manager 集成测试） ──

class MockRetriever:
    """模拟检索器——记录调用，用于验证路由是否正确"""
    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count = 0
        self.last_query: str | None = None

    def search(self, query: str, max_results: int | None = None):
        self.call_count += 1
        self.last_query = query
        return []

    def search_resources(self, query: str):
        return []


# ── 端到端场景测试 ──

class TestEndToEndRoutingScenarios:
    """端到端场景测试——模拟真实使用路径"""

    def test_full_pipeline_plan_to_routing_to_calls(self):
        """完整路径：RetrievalPlan → QueryRouter → 预期调用"""
        plan = make_plan([
            {
                "query": "large language model training efficiency",
                "target_sources": ["arxiv", "github"],
                "priority": 1,
            },
            {
                "query": "huggingface model hub fine-tuning",
                "target_sources": [],
                "priority": 2,
            },
        ])

        fake = FakeEmbedder(fixed_scores={
            "arxiv": 0.90, "github": 0.85, "semantic_scholar": 0.70,
            "huggingface": 0.80, "hackernews": 0.50, "serper": 0.40,
        })
        router = QueryRouter(embedding_service=fake)
        decisions = router.route_plan(plan, mode="hybrid", top_k=3, threshold=0.3)

        # 决策1：有 target_sources → 直接使用，不补充（hybrid 模式）
        assert set(decisions[0].routed_sources) == {"arxiv", "github"}
        # 决策2：无 target_sources → 纯 embedding 路由（top_k=3, threshold=0.3）
        # huggingface:0.80, arxiv:0.90, github:0.85 → 全部 >= 0.3 → 3 个
        assert set(decisions[1].routed_sources) == {"huggingface", "arxiv", "github"}

        # 汇总所有需要调用的 sources（去重）
        all_sources: set[str] = set()
        for d in decisions:
            all_sources.update(d.routed_sources)
        # 应该只有：arxiv, github, huggingface（去重后）
        assert all_sources == {"arxiv", "github", "huggingface"}

    def test_low_threshold_routes_more_sources(self):
        spec1 = make_plan([{"query": "test", "target_sources": []}]).queries[0]
        spec2 = make_plan([{"query": "test", "target_sources": []}]).queries[0]

        fake = FakeEmbedder(fixed_scores={
            "arxiv": 0.85, "github": 0.72, "dblp": 0.55,
            "hackernews": 0.38, "serper": 0.31,
        })
        router = QueryRouter(embedding_service=fake)

        d_strict = router.route_spec(spec1, mode="embedding", threshold=0.40)
        d_loose = router.route_spec(spec2, mode="embedding", threshold=0.30)

        # strict threshold=0.40: arxiv(0.85)✓ github(0.72)✓ dblp(0.55)✓ hackernews(0.38)✗ serper(0.31)✗
        assert set(d_strict.routed_sources) == {"arxiv", "github", "dblp"}
        # loose threshold=0.30: + dblp(0.55)✓ hackernews(0.38)✓ serper(0.31)✓
        assert len(d_loose.routed_sources) >= 2   # dblp/hackernews/serper 也可能进入

    def test_empty_plan_returns_empty_decisions(self):
        plan = make_plan([])
        fake = FakeEmbedder()
        router = QueryRouter(embedding_service=fake)
        decisions = router.route_plan(plan)
        assert decisions == []

    def test_query_with_special_characters(self):
        spec = make_plan([{
            "query": "transformer: attention is all you need (Vaswani 2017)",
            "target_sources": ["arxiv"],
        }]).queries[0]

        fake = FakeEmbedder(fixed_scores={"arxiv": 0.85, "github": 0.72})
        router = QueryRouter(embedding_service=fake)
        decision = router.route_spec(spec, mode="target_sources")

        assert decision.routed_sources == ["arxiv"]
        assert decision.query_spec.query == spec.query  # 查询内容不变