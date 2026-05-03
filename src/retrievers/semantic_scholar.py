"""Semantic Scholar 论文检索器"""

from urllib.parse import quote

from src.config import settings
from src.exceptions import AuthenticationError
from src.models import PaperCard
from src.retrievers.base import BaseRetriever

import logging

logger = logging.getLogger(__name__)

S2_API = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarRetriever(BaseRetriever):
    """Semantic Scholar API 检索器（免费 tier 注册获取 Key）"""

    name = "semantic_scholar"

    def __init__(self) -> None:
        super().__init__()
        self._api_key = settings.semantic_scholar_api_key
        if not self._api_key:
            logger.warning(
                "S2_API_KEY not set — rate limited to 1 req/s (unauthenticated)"
            )

    def search(
        self, query: str, max_results: int | None = None
    ) -> list[PaperCard]:
        if max_results is None:
            max_results = 20

        headers = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        fields = "title,authors,year,venue,abstract,citationCount,externalIds,url"
        url = (
            f"{S2_API}/paper/search"
            f"?query={quote(query)}"
            f"&limit={max_results}"
            f"&fields={fields}"
        )

        logger.info("semantic_scholar: searching '%s' (max=%d)", query, max_results)
        resp = self._get(url, headers=headers)
        data = resp.json()
        return self._parse(data)

    def get_citations(
        self, paper_id: str, max_results: int = 20
    ) -> list[PaperCard]:
        """获取引用该论文的文章列表"""
        headers = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        url = (
            f"{S2_API}/paper/{quote(paper_id)}/citations"
            f"?limit={max_results}"
            f"&fields=title,authors,year,venue,abstract,citationCount"
        )
        resp = self._get(url, headers=headers)
        data = resp.json()
        return self._parse_citations(data)

    # ── 解析 ──

    @staticmethod
    def _parse(data: dict) -> list[PaperCard]:
        cards: list[PaperCard] = []
        for raw in data.get("data", []):
            card = _parse_paper(raw)
            cards.append(card)
        logger.debug("semantic_scholar: parsed %d papers", len(cards))
        return cards

    @staticmethod
    def _parse_citations(data: dict) -> list[PaperCard]:
        cards: list[PaperCard] = []
        for raw in data.get("data", []):
            paper = raw.get("citingPaper", {})
            card = _parse_paper(paper)
            cards.append(card)
        return cards


def _parse_paper(raw: dict) -> PaperCard:
    authors: list[str] = []
    for a in raw.get("authors") or []:
        name = a.get("name", "")
        if name:
            authors.append(name)

    external_ids = raw.get("externalIds") or {}

    return PaperCard(
        title=raw.get("title", ""),
        authors=authors,
        year=raw.get("year") or 0,
        venue=raw.get("venue", ""),
        abstract=raw.get("abstract", ""),
        citation_count=raw.get("citationCount") or 0,
        url=raw.get("url", ""),
        source="semantic_scholar",
    )
