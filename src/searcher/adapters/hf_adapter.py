"""HuggingFaceAdapter — HuggingFaceRetriever → SearchResult"""

from typing import TYPE_CHECKING

from src.retrievers.huggingface import HuggingFaceRetriever

if TYPE_CHECKING:
    from src.searcher.result import SearchResult


class HuggingFaceAdapter:
    """HuggingFace 检索器适配器"""

    name = "huggingface"
    category = "resources"

    def __init__(self) -> None:
        self._retriever = HuggingFaceRetriever()

    async def search(self, query: str, max_results: int = 10) -> list["SearchResult"]:
        from src.searcher.result import SearchResult

        cards = self._retriever.search_resources(query)
        return [
            SearchResult(
                title=c.name,
                url=c.url,
                snippet=c.description[:300] if c.description else "",
                source="huggingface",
                category="resources",
                domain="huggingface.co",
            )
            for c in cards[:max_results]
        ]