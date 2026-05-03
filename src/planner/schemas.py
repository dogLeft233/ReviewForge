"""Planner 数据模型——检索规划与覆盖评估的结构化输出"""

from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════
# 检索规划
# ═══════════════════════════════════════════════

@dataclass(slots=True)
class QuerySpec:
    """单个检索查询定义"""

    query: str
    language: str = "en"  # "en" | "zh"
    target_sources: list[str] = field(default_factory=list)
    rationale: str = ""
    priority: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuerySpec":
        return cls(
            query=d.get("query", ""),
            language=d.get("language", "en"),
            target_sources=d.get("target_sources", []),
            rationale=d.get("rationale", ""),
            priority=d.get("priority", 1),
        )


@dataclass(slots=True)
class RetrievalPlan:
    """检索规划——Planner 的核心产出"""

    topic: str
    topic_analysis: dict[str, Any] = field(default_factory=dict)
    queries: list[QuerySpec] = field(default_factory=list)
    classic_search_strategy: str = ""
    frontier_search_strategy: str = ""
    resource_types_needed: list[str] = field(default_factory=list)
    estimated_coverage: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, topic: str, d: dict[str, Any]) -> "RetrievalPlan":
        queries_raw = d.get("queries", [])
        queries = [QuerySpec.from_dict(q) for q in queries_raw]
        return cls(
            topic=topic,
            topic_analysis=d.get("topic_analysis", {}),
            queries=queries,
            classic_search_strategy=d.get("classic_search_strategy", ""),
            frontier_search_strategy=d.get("frontier_search_strategy", ""),
            resource_types_needed=d.get("resource_types_needed", []),
            estimated_coverage=d.get("estimated_coverage", []),
        )


# ═══════════════════════════════════════════════
# 覆盖评估
# ═══════════════════════════════════════════════

@dataclass(slots=True)
class SupplementaryQuery:
    """补搜查询定义"""

    query: str
    target_sources: list[str] = field(default_factory=list)
    rationale: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SupplementaryQuery":
        return cls(
            query=d.get("query", ""),
            target_sources=d.get("target_sources", []),
            rationale=d.get("rationale", ""),
        )


@dataclass(slots=True)
class CoverageEvaluation:
    """覆盖度评估结果"""

    overall_assessment: str = "insufficient"  # "adequate" | "insufficient" | "critical_gaps"
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    missing_classics: list[str] = field(default_factory=list)
    missing_frontiers: list[str] = field(default_factory=list)
    supplementary_queries: list[SupplementaryQuery] = field(default_factory=list)
    resource_gaps: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def needs_supplement(self) -> bool:
        """是否需要补充检索"""
        return self.overall_assessment in ("insufficient", "critical_gaps") or bool(
            self.supplementary_queries
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CoverageEvaluation":
        sq_raw = d.get("supplementary_queries", [])
        sq = [SupplementaryQuery.from_dict(q) for q in sq_raw]
        return cls(
            overall_assessment=d.get("overall_assessment", "insufficient"),
            strengths=d.get("strengths", []),
            gaps=d.get("gaps", []),
            missing_classics=d.get("missing_classics", []),
            missing_frontiers=d.get("missing_frontiers", []),
            supplementary_queries=sq,
            resource_gaps=d.get("resource_gaps", []),
            confidence=d.get("confidence", 0.0),
        )


# ═══════════════════════════════════════════════
# 洞察综合
# ═══════════════════════════════════════════════

@dataclass(slots=True)
class EvolutionPath:
    """方法演进路线"""

    path_name: str = ""
    papers: list[str] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvolutionPath":
        return cls(
            path_name=d.get("path_name", ""),
            papers=d.get("papers", []),
            description=d.get("description", ""),
        )


@dataclass(slots=True)
class WritingRecommendation:
    """写作建议"""

    section: str = ""
    suggested_papers: list[str] = field(default_factory=list)
    key_message: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WritingRecommendation":
        return cls(
            section=d.get("section", ""),
            suggested_papers=d.get("suggested_papers", []),
            key_message=d.get("key_message", ""),
        )


@dataclass(slots=True)
class SynthesisResult:
    """洞察综合结果"""

    evolution_paths: list[EvolutionPath] = field(default_factory=list)
    hot_topics: list[str] = field(default_factory=list)
    open_problems: list[str] = field(default_factory=list)
    cross_references: list[dict[str, str]] = field(default_factory=list)
    writing_recommendations: list[WritingRecommendation] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SynthesisResult":
        evo_raw = d.get("evolution_paths", [])
        evo = [EvolutionPath.from_dict(e) for e in evo_raw]
        wr_raw = d.get("writing_recommendations", [])
        wr = [WritingRecommendation.from_dict(w) for w in wr_raw]
        return cls(
            evolution_paths=evo,
            hot_topics=d.get("hot_topics", []),
            open_problems=d.get("open_problems", []),
            cross_references=d.get("cross_references", []),
            writing_recommendations=wr,
        )
