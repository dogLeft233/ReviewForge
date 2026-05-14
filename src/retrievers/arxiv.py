"""arXiv 论文检索器 — 免费，无需 API Key"""

import re
import time
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote

import httpx

from src.config import settings
from src.models import PaperCard
from src.retrievers.base import BaseRetriever

import logging

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"


class ArxivRetriever(BaseRetriever):
    """arXiv API 检索器（无需 API Key，遵守 honor system 限速）

    用法:
        with ArxivRetriever() as arxiv:
            papers = arxiv.search("LoRA fine-tuning", max_results=20)
    """

    name = "arxiv"

    def search(
        self, query: str, max_results: int | None = None
    ) -> list[PaperCard]:
        if max_results is None:
            max_results = 30

        # 如果 query 包含 arXiv 字段前缀（ti:, all:, abs:, cat:, author:），直接使用
        # 否则加上 all: 前缀
        if re.match(r"^(ti:|all:|abs:|cat:|author:)", query.strip()):
            search_query = query.strip()
        else:
            search_query = f"all:{quote(query)}"

        params = (
            f"search_query={search_query}"
            f"&start=0&max_results={max_results}&sortBy=relevance"
        )
        url = f"{ARXIV_API}?{params}"

        logger.info("arxiv: searching '%s' (max=%d)", query, max_results)
        resp = self._get(url)
        return self._parse(resp.text)

    def search_by_category(
        self, query: str, category: str = "cs", max_results: int = 20
    ) -> list[PaperCard]:
        """按学科分类检索

        Args:
            query: 搜索关键词
            category: arXiv 分类如 "cs"、"cs.LG"、"math"
            max_results: 最多返回数
        """
        params = (
            f"search_query=cat:{category}+AND+all:{quote(query)}"
            f"&start=0&max_results={max_results}&sortBy=relevance"
        )
        url = f"{ARXIV_API}?{params}"
        resp = self._get(url)
        return self._parse(resp.text)

    def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        """带重试+速率限制的 GET，arXiv 额外加 User-Agent 头"""
        if settings.arxiv_email:
            headers = kwargs.setdefault("headers", {})
            headers["User-Agent"] = f"ReviewForge/1.0 (mailto:{settings.arxiv_email})"
        start = time.perf_counter()
        try:
            resp = super()._get(url, **kwargs)
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception("arxiv request failed: elapsed=%.3fs url=%s", elapsed, url)
            raise

        elapsed = time.perf_counter() - start
        logger.info(
            "arxiv request finished: status=%s elapsed=%.3fs url=%s",
            resp.status_code,
            elapsed,
            url,
        )
        return resp

    # ── 解析 ──

    @staticmethod
    def _parse(xml_text: str) -> list[PaperCard]:
        ns = {
            "a": "http://www.w3.org/2005/Atom",
            "o": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(xml_text)
        cards: list[PaperCard] = []

        for entry in root.findall("a:entry", ns):
            title_el = entry.find("a:title", ns)
            title = _clean(title_el.text) if title_el is not None else ""

            summary_el = entry.find("a:summary", ns)
            abstract = _clean(summary_el.text) if summary_el is not None else ""

            authors: list[str] = []
            for author_el in entry.findall("a:author", ns):
                name_el = author_el.find("a:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text)

            # id 形如 http://arxiv.org/abs/2301.00001v1
            id_el = entry.find("a:id", ns)
            arxiv_id = _clean(id_el.text) if id_el is not None else ""

            published_el = entry.find("a:published", ns)
            year = _parse_year(published_el.text) if published_el is not None else 0

            # 优先使用 arxiv:primary_category，回退到第一个 category
            primary = entry.find("o:primary_category", ns)
            subject = primary.get("term", "") if primary is not None else ""
            if not subject:
                cats = [
                    c.get("term", "")
                    for c in entry.findall("a:category", ns)
                ]
                subject = cats[0] if cats else ""

            card = PaperCard(
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                url=arxiv_id,
                source="arxiv",
                method_category=subject,
            )
            cards.append(card)

        logger.debug("arxiv: parsed %d papers", len(cards))
        return cards


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _parse_year(published: str | None) -> int:
    if not published:
        return 0
    try:
        return int(published[:4])
    except (ValueError, IndexError):
        return 0
