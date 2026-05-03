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
