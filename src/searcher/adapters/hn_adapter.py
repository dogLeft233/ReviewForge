"""HNAdapter — HackerNewsRetriever → SearchResult"""

import logging
from typing import TYPE_CHECKING

from src.retrievers.hackernews import HackerNewsRetriever

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.searcher.result import SearchResult


class HNAdapter:
    """HackerNews 检索器适配器"""

    name = "hackernews"
    category = "news"

    def __init__(self) -> None:
        self._retriever = HackerNewsRetriever()

    async def search(self, query: str, max_results: int = 10) -> list["SearchResult"]:
        logger.info("[%s] 检索中: '%s' (max=%d)", self.name, query, max_results)
        from src.searcher.result import SearchResult

        cards = self._retriever.search_resources(query)
        results = [
            SearchResult(
                title=c.name,
                url=c.url,
                snippet=c.description[:300] if c.description else "",
                source="hackernews",
                category="news",
                domain="news.ycombinator.com",
            )
            for c in cards[:max_results]
        ]
        logger.debug("[%s] 返回 %d 条结果", self.name, len(results))
        return results
