"""查询路由器——基于嵌入的智能检索器路由

支持三种路由模式：
1. target_sources：直接使用 Planner 生成 QuerySpec.target_sources 字段
2. embedding：纯嵌入相似度路由（query embedding vs retriever domain embeddings）
3. hybrid：target_sources 优先（Planner 规划），缺失时用 embedding 补充

使用方式：
    router = QueryRouter(embedding_service=svc)
    decisions = router.route_plan(plan, mode="hybrid")
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.planner.schemas import RetrievalPlan, QuerySpec

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """单次路由决策"""
    query_spec: "QuerySpec"
    routed_sources: list[str]
    similarity_scores: dict[str, float]
    routing_mode: str  # "target_sources" | "embedding" | "hybrid"
    query_embedding: np.ndarray | None = None
    query_embedding_dim: int = 0


class QueryRouter:
    """查询路由器

    Attributes:
        DEFAULT_THRESHOLD: 相似度阈值，低于此值的路由被过滤
        DEFAULT_TOP_K: embedding/hybrid 模式下最多返回的 source 数
    """

    DEFAULT_THRESHOLD = 0.30
    DEFAULT_TOP_K = 3

    def __init__(self, embedding_service: "EmbeddingService | None" = None) -> None:
        self._embedder = embedding_service
        self._embedder_ready = False

    @property
    def embedder(self) -> "EmbeddingService":
        if self._embedder is None:
            from src.embedding import EmbeddingService
            self._embedder = EmbeddingService()
            self._embedder.build_retriever_index()
            self._embedder_ready = True
        return self._embedder

    # ── 公开 API ──

    def route_spec(
        self,
        spec: "QuerySpec",
        mode: str = "hybrid",
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> RoutingDecision:
        """对一个 QuerySpec 做路由决策

        Args:
            spec: 来自 RetrievalPlan 的 QuerySpec
            mode: "target_sources" | "embedding" | "hybrid"
            top_k: embedding 模式下返回前几名
            threshold: embedding 模式下的最低相似度阈值

        Returns:
            RoutingDecision，含实际要调用的 retriever 列表
        """
        tk = top_k if top_k is not None else self.DEFAULT_TOP_K
        th = threshold if threshold is not None else self.DEFAULT_THRESHOLD

        if mode == "target_sources":
            routed = list(spec.target_sources) if spec.target_sources else []
            return RoutingDecision(
                query_spec=spec,
                routed_sources=routed,
                similarity_scores={},
                routing_mode="target_sources",
            )

        if mode == "embedding":
            routed, scores = self._embed_route(spec.query, tk, th)
            return RoutingDecision(
                query_spec=spec,
                routed_sources=routed,
                similarity_scores=scores,
                routing_mode="embedding",
                query_embedding_dim=self.embedder.actual_dimension() or 0,
            )

        # hybrid 模式：target_sources 优先，embedding 仅在未指定时补充
        decided_sources = list(spec.target_sources) if spec.target_sources else []

        if not decided_sources:
            # 没有 target_sources，全部用 embedding 路由
            routed, scores = self._embed_route(spec.query, tk, th)
            decided_sources = routed
        else:
            # 有 target_sources 时直接使用，不补充 embedding（避免冗余）
            _, scores = self._embed_route(spec.query, tk, th)

        return RoutingDecision(
            query_spec=spec,
            routed_sources=decided_sources,
            similarity_scores=scores,
            routing_mode="hybrid",
            query_embedding_dim=self.embedder.actual_dimension() or 0,
        )

    def route_plan(
        self,
        plan: "RetrievalPlan",
        mode: str = "hybrid",
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[RoutingDecision]:
        """对一个完整 RetrievalPlan 的所有 QuerySpec 做路由决策

        Args:
            plan: Planner 生成的检索规划
            mode: 路由模式
            top_k: 每次路由最多选几个 source
            threshold: 最低相似度阈值

        Returns:
            路由决策列表，与 plan.queries 一一对应
        """
        decisions = []
        for spec in plan.queries:
            decision = self.route_spec(spec, mode=mode, top_k=top_k, threshold=threshold)
            decisions.append(decision)
            logger.debug(
                "[%s] query=%r routed=%s scores=%s",
                decision.routing_mode,
                spec.query[:60],
                decision.routed_sources,
                {k: f"{v:.3f}" for k, v in decision.similarity_scores.items()},
            )
        return decisions

    # ── 内部方法 ──

    def _embed_route(
        self,
        query: str,
        top_k: int,
        threshold: float,
    ) -> tuple[list[str], dict[str, float]]:
        """纯嵌入路由：返回 source 列表 + 相似度分数"""
        results = self.embedder.route_query(query, top_k=top_k, threshold=threshold)
        sources = [name for name, _ in results]
        scores = dict(results)
        return sources, scores


def align_query_to_retriever(
    query_emb: np.ndarray,
    retriever_emb: np.ndarray,
) -> np.ndarray:
    """对齐查询向量到检索器向量的维度（公开工具函数）"""
    return EmbeddingService._align_embedding(query_emb, retriever_emb)  # type: ignore[attr-defined]