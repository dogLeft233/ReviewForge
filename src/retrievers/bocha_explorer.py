"""Bocha 检索器——探索器专用轻量版（返回 dict）"""
import logging
from typing import Any

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

BOCHA_API = "https://api.bocha.cn/v1/web-search"


class BochaRetriever:
    """博查网页搜索检索器（探索器专用，返回 list[dict]）

    与 retrievers/bocha.py 中的 BochaRetriever（返回 PaperCard）不同，
    此版本专供 DomainExplorer 使用，返回原始结构化字典。
    """

    name = "bocha"

    def __init__(self) -> None:
        self._api_key: str = ""
        self._client = httpx.Client(timeout=30.0, follow_redirects=True)
        self._load_key()

    def _load_key(self) -> None:
        self._api_key = settings.bocha_api_key
        if not self._api_key:
            logger.warning(
                "BochaRetriever (explorer): bocha_api_key not set"
            )

    def search(
        self, query: str, max_results: int = 10, summary: bool = True
    ) -> list[dict]:
        """执行博查搜索，返回标准化结果列表"""
        if not self._api_key:
            logger.warning("BochaRetriever: skipped (no API key)")
            return []

        payload: dict[str, Any] = {
            "query": query,
            "count": min(max_results, 50),
            "freshness": "noLimit",
            "summary": summary,
        }

        try:
            resp = self._client.post(
                BOCHA_API,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            return self._parse(resp.json())
        except Exception as e:
            logger.warning(
                "BochaRetriever: search failed for '%s': %s", query, e
            )
            return []

    @staticmethod
    def _parse(data: dict) -> list[dict]:
        try:
            web_pages = (
                data.get("data", {}).get("webPages", {}).get("value", [])
            )
            return [
                {
                    "title": p.get("name", ""),
                    "url": p.get("url", ""),
                    "description": p.get("snippet", ""),
                    "summary": p.get("summary", ""),
                    "site_name": p.get("siteName", ""),
                    "published_date": p.get("datePublished", ""),
                }
                for p in web_pages
            ]
        except Exception:
            return []

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BochaRetriever":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
