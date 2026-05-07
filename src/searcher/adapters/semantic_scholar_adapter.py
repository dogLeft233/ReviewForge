"""SemanticScholarAdapter — SemanticScholarRetriever → SearchResult"""

import logging
from typing import TYPE_CHECKING

from src.retrievers.semantic_scholar import SemanticScholarRetriever

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.searcher.result import SearchResult


class SemanticScholarAdapter:
    """Semantic Scholar 检索器适配器"""

    name = "semantic_scholar"
    category = "papers"

    def __init__(self) -> None:
        self._retriever = SemanticScholarRetriever()

    async def search(self, query: str, max_results: int = 10) -> list["SearchResult"]:
        logger.info("[%s] 检索中: '%s' (max=%d)", self.name, query, max_results)
        from src.searcher.result import SearchResult

        try:
            cards = self._retriever.search(query, max_results=max_results)
        except Exception as e:
            logger.warning("[%s] 搜索失败: %s", self.name, e)
            return []

        results = [
            SearchResult(
                title=c.title,
                url=c.url,
                snippet=c.abstract[:300] if c.abstract else "",
                source="semantic_scholar",
                category="papers",
                citation_count=c.citation_count,
                published_date=str(c.year) if c.year else "",
                domain="semanticscholar.org",
            )
            for c in cards
        ]
        logger.debug("[%s] 返回 %d 条结果 (带 citation_count)", self.name, len(results))
        return results
