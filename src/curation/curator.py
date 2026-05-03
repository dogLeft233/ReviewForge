"""策展管理——LLM 驱动的策展编排"""

from collections.abc import Sequence

from src.exceptions import ConfigurationError
from src.models import CurationReport, PaperCard, ResourceCard

import logging

logger = logging.getLogger(__name__)


class Curator:
    """策展人——负责任务规划、策展分析、评估循环"""

    def __init__(self, llm_client=None) -> None:
        self.llm = llm_client

    def curate(self, report: CurationReport) -> CurationReport:
        """对检索结果进行策展分析（非 LLM 版本的轻量级处理）

        在无 LLM 环境下，至少完成基础分类和统计。
        """
        # 分类
        categories: dict[str, list[PaperCard]] = {}
        for paper in report.papers:
            cat = paper.method_category or "uncategorized"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(paper)

        report.categories = categories
        report.classic_count = sum(1 for p in report.papers if p.citation_count > 50)
        report.frontier_count = sum(1 for p in report.papers if p.year >= 2023)

        # 洞察提炼（有 LLM 时走 LLM，否则走简单规则）
        report.insights = self._extract_insights(report.papers)

        # 质量检测
        missing = self._check_missing(report)
        report.missing_directions = missing

        return report

    def evaluate(self, report: CurationReport) -> str:
        """评估报告质量（有 LLM 时用 LLM，否则走规则）"""
        classic_ratio = report.classic_count / max(report.total_papers, 1)
        frontier_ratio = report.frontier_count / max(report.total_papers, 1)
        has_github = report.github_count > 0
        has_benchmark = report.benchmark_count > 0

        score = 0.0
        if classic_ratio > 0.3:
            score += 0.3
        if frontier_ratio > 0.2:
            score += 0.2
        if has_github:
            score += 0.2
        if has_benchmark:
            score += 0.15
        if report.total_papers >= 20:
            score += 0.15

        if score >= 0.8:
            return "优秀"
        elif score >= 0.5:
            return "良好"
        else:
            return "需补充"

    # ── 内部方法 ──

    @staticmethod
    def _extract_insights(papers: Sequence[PaperCard]) -> list[str]:
        """从论文集合中提取基础洞察"""
        if not papers:
            return ["暂无数据"]
        insights: list[str] = []

        years = [p.year for p in papers if p.year > 0]
        if years:
            insights.append(f"论文年份范围: {min(years)} - {max(years)}")

        venues = [p.venue for p in papers if p.venue]
        top_venues = sorted(set(venues), key=venues.count, reverse=True)[:3]
        if top_venues:
            insights.append(f"主要发表会议: {', '.join(top_venues)}")

        cats = [p.method_category for p in papers if p.method_category]
        top_cats = sorted(set(cats), key=cats.count, reverse=True)[:3]
        if top_cats:
            insights.append(f"主要方法分类: {', '.join(top_cats)}")

        high_cite = [p for p in papers if p.citation_count > 100]
        if high_cite:
            insights.append(
                f"高引用工作({len(high_cite)}篇): "
                + "; ".join(p.title[:50] for p in high_cite[:3])
            )

        return insights

    @staticmethod
    def _check_missing(report: CurationReport) -> list[str]:
        """检查遗漏方向"""
        missing: list[str] = []
        if report.classic_count < 3:
            missing.append("经典工作（高引用论文）不足")
        if report.frontier_count < 3:
            missing.append("前沿工作（2023+）不足")
        if report.github_count < 2:
            missing.append("GitHub 资源不足")
        if report.benchmark_count < 1:
            missing.append("缺少 benchmark 信息")
        return missing
