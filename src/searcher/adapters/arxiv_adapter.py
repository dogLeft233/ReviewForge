"""ArxivAdapter — ArxivRetriever → SearchResult"""

from typing import TYPE_CHECKING

from src.retrievers.arxiv import ArxivRetriever

if TYPE_CHECKING:
    from src.searcher.result import SearchResult


class ArxivAdapter:
    """Arxiv 检索器适配器"""

    name = "arxiv"
    category = "papers"

    def __init__(self) -> None:
        self._retriever = ArxivRetriever()

    async def search(self, query: str, max_results: int = 10) -> list["SearchResult"]:
        from src.searcher.result import SearchResult

        cards = self._retriever.search(query, max_results=max_results)
        return [
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