"""Wikipedia 检索器——探索器专用（返回 dict，无需 API Key）"""
import logging
import re
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_ZH_API = "https://zh.wikipedia.org/w/api.php"


class WikipediaRetriever:
    """Wikipedia 检索器（探索器专用，返回 list[dict]）

    与 retrievers/wikipedia.py 中的 WikipediaRetriever（返回 PaperCard）不同，
    此版本专供 DomainExplorer 使用，返回原始结构化字典。
    """

    name = "wikipedia"

    def __init__(
        self,
        *,
        lang: str = "en",
        max_results: int = 10,
    ) -> None:
        self.lang = lang if lang in ("en", "zh") else "en"
        self.max_results = max_results
        self._client = httpx.Client(timeout=30.0, follow_redirects=True)

    @property
    def _api_base(self) -> str:
        return WIKI_ZH_API if self.lang == "zh" else WIKI_API

    def search(
        self, query: str, max_results: int | None = None
    ) -> list[dict]:
        """Wikipedia 搜索 + 摘要提取"""
        limit = max_results or self.max_results
        try:
            resp = self._client.get(
                self._api_base,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                    "format": "json",
                },
                headers={
                    "User-Agent": "ReviewForge/1.0 (LLM Research Tool; mailto:research@example.com)",
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_search_results(data, limit)
        except Exception as e:
            logger.warning(
                "WikipediaRetriever: search failed for '%s': %s", query, e
            )
            return []

    def get_page_summary(self, title: str) -> dict | None:
        """获取指定标题的 Wikipedia 页面摘要"""
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
                    "User-Agent": "ReviewForge/1.0 (LLM Research Tool; mailto:research@example.com)",
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id == "-1":
                    return None
                return {
                    "title": page.get("title", ""),
                    "extract": page.get("extract", ""),
                    "page_id": page_id,
                }
            return None
        except Exception as e:
            logger.warning(
                "WikipediaRetriever: summary failed for '%s': %s", title, e
            )
            return None

    @staticmethod
    def _parse_search_results(data: dict, limit: int) -> list[dict]:
        results: list[dict] = []
        items = data.get("query", {}).get("search", [])
        for item in items[:limit]:
            title = item.get("title", "")
            results.append(
                {
                    "title": title,
                    "snippet": item.get("snippet", ""),
                    "page_id": item.get("pageid", 0),
                    "description": _strip_html(item.get("snippet", "")),
                    "url": (
                        f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                        if title else ""
                    ),
                }
            )
        return results

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WikipediaRetriever":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    for k, v in {
        "&quot;": '"', "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&#039;": "'", "&nbsp;": " ",
    }.items():
        text = text.replace(k, v)
    return text
