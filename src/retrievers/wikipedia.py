"""Wikipedia 检索器（免费，无需 API Key）

使用 MediaWiki opensearch API（en.wikipedia.org / zh.wikipedia.org）。
注意：部分网络环境下 Wikipedia API 有 403 限流，这是已知限制。
"""

import logging
import re
from datetime import datetime
from urllib.parse import quote

import httpx

from src.models import PaperCard, ResourceCard
from src.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)

WIKI_EN_API = "https://en.wikipedia.org/w/api.php"
WIKI_ZH_API = "https://zh.wikipedia.org/w/api.php"


class WikipediaRetriever(BaseRetriever):
    """Wikipedia 检索器（免费，无需 API Key）

    注意：在部分网络环境（如 WSL2/Windows 子系统）下，
    Wikipedia API 可能有 403 限流问题，这是 MediaWiki 的
    常见限制，并非代码问题。可考虑在 config 中设置
    `enable_wikipedia: False` 以跳过此源。
    """

    name = "wikipedia"

    def __init__(self, *, lang: str = "en") -> None:
        super().__init__()
        self.lang = lang if lang in ("en", "zh") else "en"

    @property
    def _api_base(self) -> str:
        return WIKI_ZH_API if self.lang == "zh" else WIKI_EN_API

    def search(
        self, query: str, max_results: int | None = None
    ) -> list[PaperCard]:
        limit = min(max_results or 10, 50)
        try:
            raw = self._wiki_search(query)
        except Exception as e:
            logger.debug(
                "WikipediaRetriever: search failed for '%s': %s", query, e
            )
            return []

        cards = [self._to_paper(item) for item in raw[:limit]]
        return [c for c in cards if c is not None]

    def search_resources(
        self, query: str, max_results: int | None = None
    ) -> list[ResourceCard]:
        limit = min(max_results or 10, 50)
        try:
            raw = self._wiki_search(query)
        except Exception:
            return []

        cards = [self._to_resource(item) for item in raw[:limit]]
        return [c for c in cards if c is not None]

    def get_summary(self, title: str) -> str:
        """获取页面摘要"""
        try:
            resp = self._client.get(
                self._api_base,
                params={
                    "action": "query",
                    "titles": title,
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "exsentences": 3,
                    "format": "json",
                },
                headers={
                    "User-Agent": "ReviewForge/1.0 (https://github.com/reviewforge; mailto:research@example.com)",
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                return page.get("extract", "")
            return ""
        except Exception as e:
            logger.debug(
                "WikipediaRetriever: summary failed for '%s': %s", title, e
            )
            return ""

    def _wiki_search(self, query: str) -> list[dict]:
        """opensearch 搜索（最快，且不需要额外 API Key）"""
        try:
            resp = self._client.get(
                self._api_base,
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": 10,
                    "format": "json",
                },
                headers={
                    "User-Agent": "ReviewForge/1.0 (https://github.com/reviewforge; mailto:research@example.com)",
                },
                follow_redirects=True,
            )
            if resp.status_code != 200:
                logger.debug(
                    "Wikipedia opensearch %s: status=%d",
                    query, resp.status_code,
                )
                return []

            data = resp.json()
            titles = data[1] if len(data) > 1 else []
            descs = data[2] if len(data) > 2 else []
            urls = data[3] if len(data) > 3 else []

            results = [
                {
                    "title": titles[i] if i < len(titles) else "",
                    "snippet": descs[i] if i < len(descs) else "",
                    "url": urls[i] if i < len(urls) else "",
                }
                for i in range(len(titles))
            ]
            if results:
                return results
        except Exception as e:
            logger.debug(
                "Wikipedia opensearch failed for '%s': %s", query, e
            )

        return []

    @staticmethod
    def _to_paper(item: dict) -> PaperCard | None:
        title = item.get("title", "")
        if not title:
            return None
        return PaperCard(
            title=title,
            year=0,
            venue="Wikipedia",
            abstract=item.get("snippet", ""),
            url=item.get("url", ""),
            source="wikipedia",
            retrieved_at=datetime.now().isoformat(),
            recommended_use="背景知识参考",
        )

    @staticmethod
    def _to_resource(item: dict) -> ResourceCard | None:
        title = item.get("title", "")
        if not title:
            return None
        return ResourceCard(
            name=title,
            type="other",
            url=item.get("url", ""),
            description=item.get("snippet", ""),
            source="wikipedia",
            quality_indicators=["维基百科", "免费"],
            recommended_use="背景知识参考",
            retrieved_at=datetime.now().isoformat(),
        )
