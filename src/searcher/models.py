"""Searcher 数据模型 — DomainProfile / QueryConfig / Boost"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClassicalPaper:
    """经典论文"""

    title: str
    arxivid: str = ""
    url: str = ""
    dblp_key: str = ""
    authors: list[str] = field(default_factory=list)
    boost_factor: float = 1.5
    reason: str = ""


@dataclass
class DomainProfile:
    """Explorer 输出的领域知识，在 MetaSearcher 检索前提供先验知识"""

    topic: str
    core_concepts: list[str] = field(default_factory=list)
    related_fields: list[str] = field(default_factory=list)
    era_keywords: list[str] = field(default_factory=list)
    classical_papers: list[ClassicalPaper] = field(default_factory=list)
    search_hints: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls, topic: str) -> "DomainProfile":
        """当 Explorer 不可用时，返回空 Profile"""
        return cls(topic=topic)

    def to_text(self) -> str:
        """转换为可读文本，供提示词使用"""
        parts = [f"Topic: {self.topic}"]
        if self.core_concepts:
            parts.append(f"Core concepts: {', '.join(self.core_concepts)}")
        if self.related_fields:
            parts.append(f"Related fields: {', '.join(self.related_fields)}")
        if self.classical_papers:
            papers_str = "; ".join(
                f"{p.title} ({p.arxivid})" if p.arxivid else p.title
                for p in self.classical_papers
            )
            parts.append(f"Classical papers: {papers_str}")
        if self.search_hints:
            parts.append(f"Search hints: {', '.join(self.search_hints)}")
        return "\n".join(parts)


@dataclass
class QueryVariant:
    """单个查询变体"""

    query: str
    variant_type: str  # "primary" | "title" | "historical" | "recent"
    source: str = ""
    expected_count: int = 5


@dataclass
class BoostConfig:
    """Boost 配置"""

    classical_papers: list[ClassicalPaper] = field(default_factory=list)
    high_citation_threshold: int = 100
    high_citation_boost: float = 1.2


@dataclass
class QueryConfig:
    """LLM 生成的完整查询配置"""

    topic: str
    queries: dict[str, list[QueryVariant]] = field(default_factory=dict)
    boost: BoostConfig = field(default_factory=BoostConfig)
    rerank_query: str = ""

    def get_query(self, source: str, variant: str = "primary") -> str:
        """获取指定源和变体的查询字符串"""
        variants = self.queries.get(source, [])
        for v in variants:
            if v.variant_type == variant:
                return v.query
        return ""

    def all_queries_for_source(self, source: str) -> list[QueryVariant]:
        """获取指定源的所有查询变体"""
        return self.queries.get(source, [])

    def to_fallback(self) -> "QueryConfig":
        """降级为简单查询配置"""
        return QueryConfig(
            topic=self.topic,
            queries={
                "arxiv": [QueryVariant(query=self.topic, variant_type="primary", source="arxiv")],
                "serper": [QueryVariant(query=self.topic, variant_type="primary", source="serper")],
                "github": [QueryVariant(query=self.topic, variant_type="primary", source="github")],
                "huggingface": [QueryVariant(query=self.topic, variant_type="primary", source="huggingface")],
                "bocha": [QueryVariant(query=self.topic, variant_type="primary", source="bocha")],
                "hackernews": [QueryVariant(query=self.topic, variant_type="primary", source="hackernews")],
            },
            boost=BoostConfig(),
            rerank_query=self.topic,
        )


def fallback_query_config(topic: str) -> QueryConfig:
    """为给定 topic 生成降级查询配置"""
    return QueryConfig(topic=topic).to_fallback()