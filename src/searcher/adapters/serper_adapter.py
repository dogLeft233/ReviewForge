"""SerperAdapter — SerperRetriever → SearchResult"""

import logging
from typing import TYPE_CHECKING

from src.retrievers.serper import SerperRetriever

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.searcher.result import SearchResult


class SerperAdapter:
    """Serper 检索适配器"""

    name = "serper"
    category = "papers"

    def __init__(self) -> None:
        self._retriever = SerperRetriever()

    async def search(self, query: str, max_results: int = 10) -> list["SearchResult"]:
        logger.info("[%s] 检索中: '%s' (max=%d)", self.name, query, max_results)
        from src.searcher.result import SearchResult

        cards = self._retriever.search(query)
        results = [
            SearchResult(
                title=c.title,
                url=c.url,
                snippet=c.abstract[:300] if c.abstract else "",
                source="serper",
                category="papers",
                published_date=str(c.year) if c.year else "",
                domain=getattr(c, "venue", "") or "web",
            )
            for c in cards[:max_results]
        ]
        logger.debug("[%s] 返回 %d 条结果", self.name, len(results))
        return results
