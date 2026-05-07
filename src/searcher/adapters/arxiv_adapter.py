"""ArxivAdapter — ArxivRetriever → SearchResult"""

import logging
from typing import TYPE_CHECKING

from src.retrievers.arxiv import ArxivRetriever

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.searcher.result import SearchResult


class ArxivAdapter:
    """Arxiv 检索器适配器"""

    name = "arxiv"
    category = "papers"

    def __init__(self) -> None:
        self._retriever = ArxivRetriever()

    async def search(self, query: str, max_results: int = 10) -> list["SearchResult"]:
        logger.info("[%s] 检索中: '%s' (max=%d)", self.name, query, max_results)
        from src.searcher.result import SearchResult

        cards = self._retriever.search(query, max_results=max_results)
        results = [
            SearchResult(
                title=c.title,
                url=c.url,
                snippet=c.abstract[:300] if c.abstract else "",
                source="arxiv",
                category="papers",
                published_date=str(c.year) if c.year else "",
                domain="arxiv.org",
            )
            for c in cards
        ]
        logger.debug("[%s] 返回 %d 条结果", self.name, len(results))
        return results