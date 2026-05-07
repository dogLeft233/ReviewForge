"""HNAdapter — HackerNewsRetriever → SearchResult"""

from typing import TYPE_CHECKING

from src.retrievers.hackernews import HackerNewsRetriever

if TYPE_CHECKING:
    from src.searcher.result import SearchResult


class HNAdapter:
    """HackerNews 检索器适配器"""

    name = "hackernews"
    category = "news"

    def __init__(self) -> None:
        self._retriever = HackerNewsRetriever()

    async def search(self, query: str, max_results: int = 10) -> list["SearchResult"]:
        from src.searcher.result import SearchResult

        cards = self._retriever.search_resources(query)
        return [
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