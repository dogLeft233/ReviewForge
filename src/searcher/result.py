"""SearchResult & SearchContext 数据模型"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class SearchResult:
    """搜索结果——MetaSearcher 各阶段的统一输出格式"""

    title: str
    url: str
    snippet: str = ""
    rank_score: float = 0.0
    source: str = ""  # "arxiv" | "serper" | "bocha" | "github" | "huggingface" | "hackernews"
    category: str = ""  # "papers" | "resources" | "news"
    published_date: str = ""
    domain: str = ""

    def __post_init__(self) -> None:
        if not self.domain and self.url:
            from urllib.parse import urlparse

            self.domain = urlparse(self.url).netloc

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "rank_score": self.rank_score,
            "source": self.source,
            "category": self.category,
            "published_date": self.published_date,
            "domain": self.domain,
        }


@dataclass(slots=True)
class SearchContext:
    """MetaSearcher 完整输出——三分类结果 + 元数据"""

    topic: str
    papers: list[SearchResult] = field(default_factory=list)
    resources: list[SearchResult] = field(default_factory=list)
    news: list[SearchResult] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_results(self) -> int:
        return len(self.papers) + len(self.resources) + len(self.news)

    def is_empty(self) -> bool:
        return self.total_results == 0

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "papers": [r.to_dict() for r in self.papers],
            "resources": [r.to_dict() for r in self.resources],
            "news": [r.to_dict() for r in self.news],
            "created_at": self.created_at,
        }