"""领域探索数据模型"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class TermCard:
    """术语卡片"""

    term: str
    definition: str = ""
    source: str = ""  # "bocha" | "hackernews" | "wikipedia" | "llm"
    url: str = ""


@dataclass(slots=True)
class DomainProfile:
    """领域知识探索报告——初步领域认知，供下游检索参考"""

    original_query: str = ""

    # 初步探索总结（新增）
    exploration_summary: str = ""

    # LLM 分析结果
    domain_overview: str = ""
    core_concepts: list[str] = field(default_factory=list)
    key_terms: list[TermCard] = field(default_factory=list)
    related_topics: list[str] = field(default_factory=list)
    active_work: str = ""

    # 社区洞察
    hn_insights: str = ""
    hn_discussions: list[str] = field(default_factory=list)

    # 来源追踪
    sources_used: list[str] = field(default_factory=list)
    bocha_results: list[dict] = field(default_factory=list)
    wiki_results: list[dict] = field(default_factory=list)
    hn_results: list[dict] = field(default_factory=list)
    retrieved_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_human_readable(self) -> str:
        """人类可读的领域报告"""
        lines = [f"# 领域探索报告：{self.original_query}", ""]
        if self.exploration_summary:
            lines.append(f"## 初步探索总结\n{self.exploration_summary}")
        if self.domain_overview:
            lines.append(f"## 领域概览\n{self.domain_overview}")
        if self.core_concepts:
            lines.append("## 核心概念")
            for c in self.core_concepts:
                lines.append(f"- {c}")
        if self.key_terms:
            lines.append("## 关键术语")
            for t in self.key_terms:
                src = f"({t.source})" if t.source else ""
                lines.append(f"- **{t.term}** {src}: {t.definition}")
        if self.related_topics:
            lines.append(f"## 相关主题\n{', '.join(self.related_topics)}")
        if self.active_work:
            lines.append(f"## 代表性工作\n{self.active_work}")
        if self.hn_insights:
            lines.append(f"## HN 社区洞察\n{self.hn_insights}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（供下游使用）"""
        return {
            "original_query": self.original_query,
            "exploration_summary": self.exploration_summary,
            "domain_overview": self.domain_overview,
            "core_concepts": self.core_concepts,
            "key_terms": [
                {"term": t.term, "definition": t.definition, "source": t.source}
                for t in self.key_terms
            ],
            "related_topics": self.related_topics,
            "active_work": self.active_work,
            "hn_insights": self.hn_insights,
            "hn_discussions": self.hn_discussions,
            "sources_used": self.sources_used,
            "retrieved_at": self.retrieved_at,
        }
