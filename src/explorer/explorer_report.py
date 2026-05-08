"""Explorer Report — 三阶段探索报告"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class BenchmarkEntry:
    """单个 Benchmark/Leaderboard 条目"""
    name: str
    url: str
    description: str = ""
    metric: str = ""  # e.g. "accuracy", "BLEU", "WER"
    dataset: str = ""  # e.g. "LibriSpeech", "Switchboard"
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    # leaderboard 每项: {"model": str, "score": float, "date": str, "note": str}


@dataclass(slots=True)
class ClassicWork:
    """经典工作"""
    title: str
    year: int
    authors: str = ""
    venue: str = ""  # conference/journal
    key_idea: str = ""
    impact: str = ""


@dataclass(slots=True)
class ExplorerReport:
    """Explorer 的完整探索报告"""

    # 主题
    topic: str

    # ── Stage 1: 领域概况 ────────────────────────────────
    stage1_overview: str = ""       # 领域定义、核心问题、主流方法
    stage1_concepts: list[str] = field(default_factory=list)
    stage1_search_results: str = ""

    # ── Stage 2: 经典工作与历史阶段 ─────────────────────
    stage2_classics: list[ClassicWork] = field(default_factory=list)
    stage2_timeline: str = ""     # 历史演进线
    stage2_search_results: str = ""

    # ── Stage 3: 前沿 ──────────────────────────────────
    stage3_benchmarks: list[BenchmarkEntry] = field(default_factory=list)
    stage3_state_of_art: str = ""  # 当前 SOTA 方法
    stage3_trends: list[str] = field(default_factory=list)
    stage3_search_results: str = ""

    # ── 综合报告（供下游模型使用）──────────────────────
    downstream_report: str = ""   # 最终总结报告

    # ── 元数据 ─────────────────────────────────────────
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_queries: int = 0

    def get_all_benchmark_urls(self) -> list[str]:
        """提取所有 benchmark URL"""
        return [b.url for b in self.stage3_benchmarks if b.url]

    def to_downstream_summary(self) -> str:
        """下游模型可用的总结摘要"""
        if self.downstream_report:
            return self.downstream_report

        sections = []

        sections.append("# " + self.topic + " 领域探索报告\n")

        if self.stage1_overview:
            sections.append("## 1. 领域概况\n" + self.stage1_overview)
        if self.stage1_concepts:
            sections.append("### 核心概念\n" + "\n".join(f"- {c}" for c in self.stage1_concepts))

        if self.stage2_classics:
            sections.append("\n## 2. 经典工作\n")
            for c in self.stage2_classics:
                sections.append(f"- **{c.title}** ({c.year}) — {c.key_idea}")

        if self.stage3_benchmarks:
            sections.append("\n## 3. Benchmark & Leaderboard\n")
            for b in self.stage3_benchmarks:
                url_line = f"[{b.url}]" if b.url else "(无URL)"
                sections.append(f"- **{b.name}** {url_line} — {b.description}")
                if b.metric:
                    sections.append(f"  评估指标: {b.metric} / 数据集: {b.dataset}")

        if self.stage3_trends:
            sections.append("\n## 4. 前沿趋势\n" + "\n".join(f"- {t}" for t in self.stage3_trends))

        return "\n\n".join(sections)
