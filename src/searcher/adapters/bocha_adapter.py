"""BochaAdapter — BochaRetriever → SearchResult"""

from typing import TYPE_CHECKING

from src.retrievers.bocha import BochaRetriever

if TYPE_CHECKING:
    from src.searcher.result import SearchResult


class BochaAdapter:
    """Bocha 搜索适配器"""

    name = "bocha"
    category = "news"

    def __init__(self) -> None:
        self._retriever = BochaRetriever()

    async def search(self, query: str, max_results: int = 10) -> list["SearchResult"]:
        from src.searcher.result import SearchResult

        cards = self._retriever.search(query, max_results=max_results)
        return [
            SearchResult(
                title=c.title,
                url=c.url,
                snippet=c.abstract[:300] if c.abstract else "",
                source="bocha",
                category="news",
                published_date=getattr(c, "retrieved_at", "")[:10] if hasattr(c, "retrieved_at") else "",
                domain=getattr(c, "venue", "") or "web",
            )
            for c in cards
        ]