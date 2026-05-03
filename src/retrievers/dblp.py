"""DBLP 论文检索器"""

import xml.etree.ElementTree as ET
from urllib.parse import quote

from src.models import PaperCard
from src.retrievers.base import BaseRetriever

import logging

logger = logging.getLogger(__name__)

DBLP_API = "https://dblp.org/search/publ/api"


class DblpRetriever(BaseRetriever):
    """DBLP 论文检索器（完全免费，无需 API Key）"""

    name = "dblp"

    def search(
        self, query: str, max_results: int | None = None
    ) -> list[PaperCard]:
        if max_results is None:
            max_results = 20

        params = f"q={quote(query)}&h={max_results}&format=json"
        url = f"{DBLP_API}?{params}"

        logger.info("dblp: searching '%s' (max=%d)", query, max_results)

        try:
            resp = self._get(url)
            data = resp.json()
        except Exception:
            # DBLP JSON 格式有时候会变，回退 XML
            return self._fallback_xml(query, max_results)

        return self._parse(data)

    def search_by_author(self, author: str, max_results: int = 20) -> list[PaperCard]:
        """按作者检索"""
        return self.search(f"author:{quote(author)}", max_results)

    # ── 解析 ──

    @staticmethod
    def _parse(data: dict) -> list[PaperCard]:
        cards: list[PaperCard] = []
        result = data.get("result", {})
        hits = result.get("hits", {})
        for hit in hits.get("hit", []):
            info = hit.get("info", {})
            # 提取年份
            year_str = info.get("year") or 0
            if isinstance(year_str, str):
                try:
                    year = int(year_str)
                except ValueError:
                    year = 0
            else:
                year = int(year_str) if year_str else 0

            authors: list[str] = []
            author_field = info.get("authors", {}).get("author", [])
            # DBLP 返回单作者时是 dict，多作者时是 list
            if isinstance(author_field, dict):
                author_field = [author_field]
            for a in author_field:
                name = a if isinstance(a, str) else a.get("text", "")
                if name:
                    authors.append(name)

            venue = info.get("venue", "")
            title = info.get("title", "")

            card = PaperCard(
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                url=info.get("url", ""),
                source="dblp",
            )
            cards.append(card)

        logger.debug("dblp: parsed %d papers", len(cards))
        return cards

    def _fallback_xml(self, query: str, max_results: int) -> list[PaperCard]:
        """JSON 不兼容时回退到 XML 格式"""
        params = f"q={quote(query)}&h={max_results}&format=xml"
        url = f"{DBLP_API}?{params}"

        try:
            resp = self._get(url)
            return self._parse_xml(resp.text)
        except Exception as e:
            logger.warning("dblp fallback also failed: %s", e)
            return []

    @staticmethod
    def _parse_xml(xml_text: str) -> list[PaperCard]:
        cards: list[PaperCard] = []
        root = ET.fromstring(xml_text)
        for hit in root.findall(".//hit"):
            info = hit.find("info")
            if info is None:
                continue

            title_el = info.find("title")
            year_el = info.find("year")
            venue_el = info.find("venue")
            url_el = info.find("url")

            authors: list[str] = []
            for a in info.findall(".//author"):
                if a.text:
                    authors.append(a.text)

            card = PaperCard(
                title=title_el.text or "" if title_el is not None else "",
                authors=authors,
                year=int(year_el.text) if year_el is not None and year_el.text else 0,
                venue=venue_el.text or "" if venue_el is not None else "",
                url=url_el.text or "" if url_el is not None else "",
                source="dblp",
            )
            cards.append(card)

        return cards
