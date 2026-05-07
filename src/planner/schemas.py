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




def _parse_sq_list(raw: list[Any]) -> list[SupplementaryQuery]:
    """解析 supplementary_queries 列表，兼容字符串和字典两种格式

    LLM 可能输出:
    - ["query string"] (纯字符串，旧格式)
    - [{"query": "...", "target_sources": [...]}] (字典，新格式)
    """
    result: list[SupplementaryQuery] = []
    for item in raw:
        if isinstance(item, str):
            result.append(SupplementaryQuery(query=item))
        elif isinstance(item, dict):
            result.append(SupplementaryQuery.from_dict(item))
    return result


@dataclass(slots=True)
class GapQuery:
    """带来源策略指引的补搜查询，用于 supplementary_search_detailed"""
    query: str
    target_sources: list[str] = field(default_factory=list)
    section_id: str = ""
    section_title: str = ""
    language: str = "en"

    def to_display(self) -> str:
        srcs = ', '.join(self.target_sources) if self.target_sources else '全源'
        return f"  [{self.section_id}] {self.query} -> {srcs}"

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


# ═══════════════════════════════════════════════
# 综述细纲
# ═══════════════════════════════════════════════


@dataclass(slots=True)
class OutlineSection:
    """细纲的一个章节节点"""

    id: str = ""  # "2.1"
    title: str = ""  # "批处理框架：Hadoop→Spark→Flink"
    level: int = 1  # 1/2/3
    description: str = ""  # 本节要回答的核心问题
    evidence_required: list[str] = field(default_factory=list)  # ["经典文献", "前沿论文", "benchmark"]
    coverage_status: str = "unknown"  # "sufficient" | "partial" | "insufficient" | "unknown"
    coverage_rationale: str = ""  # 为什么是这个状态
    supplementary_queries: list[SupplementaryQuery] = field(default_factory=list)  # 针对该节的补搜
    child_sections: list["OutlineSection"] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OutlineSection":
        children_raw = d.get("child_sections", [])
        children = [OutlineSection.from_dict(c) for c in children_raw]
        return cls(
            id=d.get("id", ""),
            title=d.get("title", ""),
            level=d.get("level", 1),
            description=d.get("description", ""),
            evidence_required=d.get("evidence_required", []),
            coverage_status=d.get("coverage_status", "unknown"),
            coverage_rationale=d.get("coverage_rationale", ""),
            supplementary_queries=_parse_sq_list(d.get("supplementary_queries", [])),
            child_sections=children,
        )

    def to_text(self, indent: int = 0) -> str:
        """格式化为文本，用于渲染或调试"""
        prefix = "  " * indent
        status_icon = {
            "sufficient": "✅",
            "partial": "⚠️",
            "insufficient": "❌",
            "unknown": "❓",
        }.get(self.coverage_status, "❓")
        lines = [f"{prefix}{status_icon} {self.id} {self.title}"]
        if self.description:
            lines.append(f"{prefix}   └ {self.description}")
        if self.coverage_rationale:
            lines.append(f"{prefix}   └ 评估: {self.coverage_rationale}")
        if self.supplementary_queries:
            for sq in self.supplementary_queries:
                q_text = sq.query if hasattr(sq, 'query') else str(sq)
                lines.append(f"{prefix}   └ 补搜: {q_text}")
        for child in self.child_sections:
            lines.append(child.to_text(indent + 1))
        return "\n".join(lines)


@dataclass(slots=True)
class Outline:
    """综述细纲——generate_outline() 的核心产出

    既是「检索完备性的测试集」，又是「下游 Writer 的生产蓝图」。
    """

    topic: str = ""
    abstract: str = ""  # 摘要草稿
    sections: list[OutlineSection] = field(default_factory=list)
    overall_coverage: float = 0.0  # 0.0 ~ 1.0
    gap_summary: str = ""  # 整体缺漏概况
    supplementary_queries: list[SupplementaryQuery] = field(default_factory=list)  # 全局补搜建议

    @classmethod
    def from_dict(cls, topic: str, d: dict[str, Any]) -> "Outline":
        sections_raw = d.get("sections", [])
        sections = [OutlineSection.from_dict(s) for s in sections_raw]
        return cls(
            topic=topic,
            abstract=d.get("abstract", ""),
            sections=sections,
            overall_coverage=d.get("overall_coverage", 0.0),
            gap_summary=d.get("gap_summary", ""),
            supplementary_queries=_parse_sq_list(d.get("supplementary_queries", [])),
        )

    @property
    def needs_supplement(self) -> bool:
        """是否有任何章节需要补充检索"""
        if self.supplementary_queries:
            return True
        return any(
            self._section_needs_supplement(s)
            for s in self.sections
        )

    def collect_gap_queries(self) -> tuple[list[str], dict[str, list[str]]]:
        """提取所有缺口查询

        递归收集全局 + 各章节的 supplementary_queries。

        Returns:
            (all_queries, section_map)
            - all_queries: 去重后的所有查询字符串
            - section_map: {section_id: [queries]} 标记每个查询来自哪个章节
        """
        all_queries: list[str] = []
        section_map: dict[str, list[str]] = {}

        # 全局查询
        for sq in self.supplementary_queries:
            q = sq.query if hasattr(sq, 'query') else str(sq)
            if q not in all_queries:
                all_queries.append(q)

        # 逐章节收集（递归）
        self._collect_section_queries(self.sections, all_queries, section_map)

        return all_queries, section_map

    def collect_gap_queries_detailed(self) -> tuple[list[GapQuery], dict[str, list[GapQuery]]]:
        """提取所有缺口查询（详细版，带来源策略和语言）

        Returns:
            (all_gap_queries, section_map)
            - all_gap_queries: 去重后的 GapQuery 列表
            - section_map: {section_id: [GapQuery]}
        """
        all_gap_queries: list[GapQuery] = []
        seen_queries: set[str] = set()
        section_map: dict[str, list[GapQuery]] = {}

        # 全局查询
        for sq in self.supplementary_queries:
            if sq.query not in seen_queries:
                seen_queries.add(sq.query)
                gap = GapQuery(
                    query=sq.query,
                    target_sources=sq.target_sources,
                    section_id="全局",
                    section_title="全局",
                    language="en",
                )
                all_gap_queries.append(gap)

        # 逐章节收集（递归）
        self._collect_section_gap_queries(self.sections, seen_queries, all_gap_queries, section_map)

        return all_gap_queries, section_map

    @staticmethod
    def _collect_section_gap_queries(
        sections: list[OutlineSection],
        seen_queries: set[str],
        all_gap_queries: list[GapQuery],
        section_map: dict[str, list[GapQuery]],
    ) -> None:
        for sec in sections:
            if sec.supplementary_queries:
                gap_queries: list[GapQuery] = []
                for sq in sec.supplementary_queries:
                    if sq.query not in seen_queries:
                        seen_queries.add(sq.query)
                        gap = GapQuery(
                            query=sq.query,
                            target_sources=sq.target_sources,
                            section_id=sec.id,
                            section_title=sec.title,
                            language="en",
                        )
                        all_gap_queries.append(gap)
                        gap_queries.append(gap)
                if gap_queries:
                    section_map[sec.id] = gap_queries
            if sec.child_sections:
                Outline._collect_section_gap_queries(sec.child_sections, seen_queries, all_gap_queries, section_map)

    @staticmethod
    def _collect_section_queries(
        sections: list[OutlineSection],
        all_queries: list[str],
        section_map: dict[str, list[str]],
    ) -> None:
        for sec in sections:
            if sec.supplementary_queries:
                section_map[sec.id] = [(sq.query if hasattr(sq, 'query') else str(sq)) for sq in sec.supplementary_queries]
                for sq in sec.supplementary_queries:
                    q = sq.query if hasattr(sq, 'query') else str(sq)
                    if q not in all_queries:
                        all_queries.append(q)
            if sec.child_sections:
                Outline._collect_section_queries(sec.child_sections, all_queries, section_map)

    @staticmethod
    def _section_needs_supplement(section: OutlineSection) -> bool:
        if section.coverage_status in ("insufficient", "partial"):
            return True
        return any(
            Outline._section_needs_supplement(c)
            for c in section.child_sections
        )

    def to_full_text(self) -> str:
        """格式化为完整文本，可直接用于预览"""
        lines = [
            f"# {self.topic}",
            f"",
            f"**摘要**: {self.abstract}",
            f"",
            f"**整体覆盖度**: {self.overall_coverage:.0%}",
        ]
        if self.gap_summary:
            lines.append(f"**缺漏概况**: {self.gap_summary}")
        if self.supplementary_queries:
            lines.append(f"**需要补搜**:")
            for sq in self.supplementary_queries:
                if hasattr(sq, 'query'):
                    src_tag = f" [{', '.join(sq.target_sources)}]" if sq.target_sources else ""
                    lines.append(f"  - {sq.query}{src_tag}")
                else:
                    lines.append(f"  - {sq}")
        lines.append("")
        for section in self.sections:
            lines.append(section.to_text())
        return "\n".join(lines)


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
