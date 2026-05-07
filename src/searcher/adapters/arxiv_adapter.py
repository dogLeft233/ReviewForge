"""ArxivAdapter — ArxivRetriever → SearchResult"""

import logging
import re
from typing import TYPE_CHECKING

from src.retrievers.arxiv import ArxivRetriever

logger = logging.getLogger(__name__)


def _clean_arxiv_query(query: str) -> str:
    """Clean LLM-generated arXiv query to something ArxivRetriever can use.

    The LLM generates queries like ``all:transformer AND all:attention cat:cs.LG&sortBy=relevance``
    but ArxivRetriever.search() always wraps the entire query in ``all:`` and appends its own
    ``&sortBy=...``. Field prefixes that would be doubled or are unsupported are stripped:
    ``all:``, ``ti:``, ``abs:``, ``au:``, ``cat:``, ``co:``, ``submitter:``, ``comm:``.
    ``&sortBy=...`` suffix is also removed (retriever adds its own).
    """
    # Remove &sortBy=... (retriever appends its own)
    query = re.sub(r"&sortBy=[^&\s]*", "", query, flags=re.IGNORECASE)
    # Strip all arXiv field:value prefixes (retriever wraps in all: and doesn't support others)
    query = re.sub(r"\b(?:all|ti|abs|au|co|cat|submitter|comm):", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\s+", " ", query).strip()
    return query


class ArxivAdapter:
    """Arxiv 检索器适配器"""

    name = "arxiv"
    category = "papers"

    def __init__(self) -> None:
        self._retriever = ArxivRetriever()

    async def search(self, query: str, max_results: int = 10) -> list["SearchResult"]:
        logger.info("[%s] 检索中: '%s' (max=%d)", self.name, query, max_results)
        from src.searcher.result import SearchResult

        clean_query = _clean_arxiv_query(query)
        cards = self._retriever.search(clean_query, max_results=max_results)
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
