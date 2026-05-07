"""Bocha 网页搜索检索器

基于博查 AI 搜索 API（bochaai.com）的网页检索器。
需要 BOCHA_API_KEY，可从 config 或环境变量获取。
"""

from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from src.config import settings
from src.models import PaperCard, ResourceCard
from src.retrievers.base import BaseRetriever

import logging

logger = logging.getLogger(__name__)

BOCHA_API = "https://api.bocha.cn/v1/web-search"


class BochaRetriever(BaseRetriever):
    """博查网页搜索检索器（需要 BOCHA_API_KEY）"""

    name = "bocha"

    def __init__(self) -> None:
        super().__init__()
        self._api_key = settings.bocha_api_key
        if not self._api_key:
            logger.warning(
                "BochaRetriever: BOCHA_API_KEY not set — "
                "Set BOCHA_API_KEY env var or configure bocha_api_key in config. "
                "Get key at https://open.bochaai.com"
            )

    def search(self, query: str, max_results: int | None = None) -> list[PaperCard]:
        """Bocha 搜索 → 返回 PaperCard 列表"""
        if not self._api_key:
            logger.warning("BochaRetriever: skipped (no API key)")
            return []

        limit = min(max_results or 10, 50)
        try:
            raw = self._bocha_search(query, limit)
        except Exception as e:
            logger.warning("BochaRetriever: search failed for '%s': %s", query, e)
            return []

        cards: list[PaperCard] = []
        for item in raw:
            card = self._to_paper(item, query)
            if card:
                cards.append(card)

        logger.info(
            "BochaRetriever: '%s' → %d raw → %d paper cards",
            query, len(raw), len(cards),
        )
        return cards

    def search_resources(
        self, query: str, max_results: int | None = None
    ) -> list[ResourceCard]:
        """Bocha 搜索 → 返回 ResourceCard 列表"""
        if not self._api_key:
            return []

        limit = min(max_results or 10, 50)
        try:
            raw = self._bocha_search(query, limit)
        except Exception:
            return []

        cards: list[ResourceCard] = []
        for item in raw:
            card = self._to_resource(item, query)
            if card:
                cards.append(card)

        return cards

    def _bocha_search(self, query: str, count: int) -> list[dict]:
        """调用博查 API"""
        payload: dict[str, Any] = {
            "query": query,
            "count": count,
            "freshness": "noLimit",
            "summary": True,
        }

        resp = self._client.post(
            BOCHA_API,
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return self._parse(data)

    @staticmethod
    def _parse(data: dict) -> list[dict]:
        """解析博查 API 响应"""
        try:
            web_pages = data.get("data", {}).get("webPages", {}).get("value", [])
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

    @staticmethod
    def _to_paper(item: dict, query: str) -> PaperCard | None:
        """将搜索结果映射为 PaperCard"""
        title = item.get("title", "")
        link = item.get("url", "")
        snippet = item.get("description", "") or item.get("summary", "")

        if not title or not link:
            return None

        # 提取年份
        year = 0
        for y in range(2020, 2027):
            if str(y) in (item.get("published_date", "") or "") or str(y) in link:
                year = y
                break

        domain = urlparse(link).netloc

        return PaperCard(
            title=title,
            year=year,
            venue=domain,
            abstract=snippet,
            url=link,
            source="bocha",
            retrieved_at=datetime.now().isoformat(),
            recommended_use=_recommend_use(title, snippet, link),
        )

    @staticmethod
    def _to_resource(item: dict, query: str) -> ResourceCard | None:
        """将搜索结果映射为 ResourceCard"""
        title = item.get("title", "")
        link = item.get("url", "")
        snippet = item.get("description", "") or item.get("summary", "")

        if not title or not link:
            return None

        domain = urlparse(link).netloc
        rtype = _detect_resource_type(domain, title, snippet)

        indicators: list[str] = []
        if domain:
            indicators.append(f"来源: {domain}")
        if snippet:
            if "open source" in snippet.lower():
                indicators.append("开源")
            if "free" in snippet.lower():
                indicators.append("免费")

        return ResourceCard(
            name=title,
            type=rtype,
            url=link,
            description=snippet,
            source="bocha",
            quality_indicators=indicators,
            recommended_use=_recommend_use(title, snippet, link),
            retrieved_at=datetime.now().isoformat(),
        )


# ── 辅助函数 ──

_RESOURCE_DOMAINS = {
    "github.com": "github_repo",
    "gitlab.com": "github_repo",
    "pypi.org": "tool",
    "npmjs.com": "tool",
    "kaggle.com": "dataset",
    "huggingface.co": "model",
    "paperswithcode.com": "benchmark",
    "colab.research.google.com": "tool",
}

_TOOL_KW = ["framework", "library", "tool", "platform", "sdk", "toolkit", "engine"]
_DATASET_KW = ["dataset", "benchmark", "corpus", "collection"]
_BENCHMARK_KW = ["benchmark", "leaderboard", "evaluation"]


def _detect_resource_type(domain: str, title: str, snippet: str) -> str:
    text = f"{title} {snippet}".lower()
    for d, rtype in _RESOURCE_DOMAINS.items():
        if d in domain:
            return rtype
    if any(kw in text for kw in _BENCHMARK_KW):
        return "benchmark"
    if any(kw in text for kw in _DATASET_KW):
        return "dataset"
    if any(kw in text for kw in _TOOL_KW):
        return "tool"
    return "other"


def _recommend_use(title: str, snippet: str, link: str) -> str:
    text = f"{title} {snippet}".lower()
    if any(kw in text for kw in ("survey", "review", "overview")):
        return "可用于综述背景章节"
    if any(kw in text for kw in ("tutorial", "guide", "getting started")):
        return "可用于入门背景介绍"
    if "arxiv" in text or "arxiv" in link.lower():
        return "论文引用来源"
    if "blog" in text or "blog" in link.lower():
        return "可作为应用案例参考"
    return "可作为补充参考材料"
