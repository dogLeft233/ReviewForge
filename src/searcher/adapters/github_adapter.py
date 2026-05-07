"""GithubAdapter — GithubRetriever → SearchResult"""

import logging
from typing import TYPE_CHECKING

from src.retrievers.github import GithubRetriever

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.searcher.result import SearchResult


class GithubAdapter:
    """GitHub 检索器适配器"""

    name = "github"
    category = "resources"

    def __init__(self) -> None:
        self._retriever = GithubRetriever()

    async def search(self, query: str, max_results: int = 10) -> list["SearchResult"]:
        logger.info("[%s] 检索中: '%s' (max=%d)", self.name, query, max_results)
        from src.searcher.result import SearchResult

        cards = self._retriever.search_resources(query)
        results = [
            SearchResult(
                title=c.name,
                url=c.url,
                snippet=c.description[:300] if c.description else "",
                source="github",
                category="resources",
                domain="github.com",
            )
            for c in cards[:max_results]
        ]
        logger.debug("[%s] 返回 %d 条结果", self.name, len(results))
        return results
