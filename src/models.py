"""数据模型定义"""

from dataclasses import dataclass, field
from datetime import datetime


# ──────────────────────────────────────────────
# 论文卡片
# ──────────────────────────────────────────────


@dataclass(slots=True)
class PaperCard:
    """论文卡片——从检索→策展的完整信息载体"""

    # ── 元数据（来自检索） ──
    title: str
    authors: list[str] = field(default_factory=list)
    year: int = 0
    venue: str = ""
    abstract: str = ""
    citation_count: int = 0
    url: str = ""
    source: str = ""  # "arxiv" | "semantic_scholar" | "dblp"

    # ── 策展字段（来自 LLM 分析） ──
    method_category: str = ""
    key_contribution: str = ""
    limitations: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    section_tag: str = ""  # "classic" | "frontier" | "background"
    talking_point: str = ""
    benchmark_mentioned: str = ""

    # ── 代码资源 ──
    github_repo: str = ""
    github_stars: int = 0

    # ── 关系标注 ──
    builds_on: str = ""
    compared_to: str = ""
    complements: str = ""

    # ── 编排信息 ──
    recommended_use: str = ""
    section_placement_reasoning: str = ""
    uncertainty: str = ""

    # ── 来源追踪 ──
    retrieved_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ──────────────────────────────────────────────
# 资源卡片
# ──────────────────────────────────────────────


@dataclass(slots=True)
class ResourceCard:
    """资源卡片——GitHub / Dataset / Benchmark / Model"""

    name: str
    type: str  # "github_repo" | "dataset" | "benchmark" | "model"
    url: str = ""
    stars: int = 0
    description: str = ""
    owner: str = ""

    # ── 策展字段 ──
    quality_indicators: list[str] = field(default_factory=list)
    related_papers: list[str] = field(default_factory=list)
    recommended_use: str = ""
    demo_scenario: str = ""
    limitations: str = ""

    # ── 来源追踪 ──
    source: str = ""
    retrieved_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ──────────────────────────────────────────────
# 策展报告
# ──────────────────────────────────────────────


@dataclass(slots=True)
class CurationReport:
    """检索策展报告——下游综述写作的输入"""

    topic: str
    papers: list[PaperCard] = field(default_factory=list)
    resources: list[ResourceCard] = field(default_factory=list)

    # 概要统计
    total_papers: int = 0
    classic_count: int = 0
    frontier_count: int = 0
    github_count: int = 0
    benchmark_count: int = 0
    dataset_count: int = 0

    # 分类体系
    categories: dict[str, list[PaperCard]] = field(default_factory=dict)

    # 关键洞察
    insights: list[str] = field(default_factory=list)
    open_problems: list[str] = field(default_factory=list)

    # 质量自检
    missing_directions: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    # 写作支撑映射
    writing_support: dict[str, dict] = field(default_factory=dict)

    # 元数据
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    queries: list[str] = field(default_factory=list)

    def to_papers_summary(self) -> str:
        """论文列表的文本摘要，用于 LLM 输入"""
        if not self.papers:
            return "暂无论文数据"
        lines = []
        for p in self.papers:
            venue_str = p.venue or "-"
            lines.append(
                f"{p.title} | {p.year} | {p.citation_count} | "
                f"{venue_str} | {p.source}"
            )
        return "\n".join(lines)

    def to_resources_summary(self) -> str:
        """资源列表的文本摘要，用于 LLM 输入"""
        if not self.resources:
            return "暂无资源数据"
        lines = []
        for r in self.resources:
            desc = r.description or "-"
            lines.append(f"{r.name} | {r.type} | {desc}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 补搜报告
# ──────────────────────────────────────────────


@dataclass(slots=True)
class SuppleReport:
    """补搜报告——supplementary_search() 的产出

    记录针对细纲缺口执行的所有补搜的结果。
    """

    topic: str
    queries_executed: list[str] = field(default_factory=list)
    """实际执行的查询"""
    result_map: dict[str, CurationReport] = field(default_factory=dict)
    """查询 → 检索报告"""

    # 汇总统计
    total_new_papers: int = 0
    total_new_resources: int = 0
    new_classics: int = 0
    new_frontiers: int = 0
    sections_improved: list[str] = field(default_factory=list)
    """哪些章节因此得到了补充"""

    # 原始缺口信息
    gap_queries: dict[str, list[str]] = field(default_factory=dict)
    """section_id → [gap queries]（仅记录有补搜的章节）"""

    # 元数据
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0

    @property
    def total_queries(self) -> int:
        return len(self.queries_executed)

    @property
    def has_new_findings(self) -> bool:
        return self.total_new_papers > 0 or self.total_new_resources > 0

    def to_summary(self) -> str:
        """摘要文本"""
        lines = [
            f"补搜报告: {self.topic}",
            f"  执行查询: {self.total_queries} 个",
            f"  新增论文: {self.total_new_papers} 篇",
            f"    其中经典: {self.new_classics} | 前沿: {self.new_frontiers}",
            f"  新增资源: {self.total_new_resources} 个",
            f"  补充章节: {len(self.sections_improved)} 个",
            f"  耗时: {self.duration_seconds:.1f}s",
        ]
        if self.gap_queries:
            lines.append(f"  查询分布:")
            for sec_id, queries in self.gap_queries.items():
                lines.append(f"    {sec_id}: {', '.join(queries)}")
        return "\n".join(lines)
