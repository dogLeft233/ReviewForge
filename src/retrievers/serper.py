"""SerperRetriever — Google 搜索 API（通过 serper.dev），提取 arXiv 论文 ID

PaSa 架构的 Stage 1 检索策略：Google 搜索 → 提取 arXiv ID → 用 ArxivRetriever 获取详情

API 文档: https://serper.dev
限速: 1 req/s（有 key）
"""

from __future__ import annotations

import re
import time
import logging
from typing import Any

import httpx

from src.config import settings
from src.models import PaperCard
from src.retrievers.base import BaseRetriever
from src.retrievers.rate_limits import get_source_limiter

logger = logging.getLogger(__name__)

SERPER_API = "https://google.serper.dev/search"


class SerperRetriever(BaseRetriever):
    """Serper.dev Google 搜索检索器

    通过 Google 搜索 `site:arxiv.org` 提取 arXiv 论文 ID，
    再用 ArxivRetriever 获取标题/摘要。

    限速策略:
    - API 限 1 req/s，RateLimiter 保底 1 req/s
    - 多线程并发共享同一个限速器实例（进程级单例）

    用法:
        with SerperRetriever() as serper:
            papers = serper.search("LoRA fine-tuning site:arxiv.org", max_results=20)
    """

    name = "serper"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self._api_key = api_key or settings.serper_api_key
        self._limiter = get_source_limiter("serper")

    def search(self, query: str, max_results: int | None = None) -> list[PaperCard]:
        """搜索论文，返回 PaperCard 列表（标题/摘要/URL=arxiv_id）

        注意：Google 搜索返回的是 paper ID 列表，需要 ArxivRetriever
        补充标题和摘要。返回的 PaperCard.url 字段为 arXiv ID（用于后续 fetch）。
        """
        if max_results is None:
            max_results = 20

        if not self._api_key:
            logger.warning("SerperRetriever: 无 API Key，跳过搜索")
            return []

        # Google 搜索 site:arxiv.org
        payload = {
            "q": query,
            "num": min(max_results, 10),  # Serper 单次最多 10 条
        }
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }

        all_results: list[dict] = []
        # Serper 单次最多 10 条，分页获取
        for start in range(0, max_results, 10):
            self._limiter.wait()
            payload_page = dict(payload)
            payload_page["start"] = start

            try:
                resp = self._post(SERPER_API, json=payload_page, headers=headers)
                data = resp.json()
                items = data.get("organic", [])
                all_results.extend(items)

                # 没有更多结果
                if len(items) < 10:
                    break
                # 避免过于频繁
                time.sleep(1.1)
            except Exception as e:
                logger.error("Serper search failed [start=%d]: %s", start, e)
                break

        # 解析结果，提取 arXiv ID
        papers: list[PaperCard] = []
        seen_ids: set[str] = set()

        for item in all_results:
            url = item.get("link", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")

            # 从 URL 提取 arXiv ID
            arxiv_id = _extract_arxiv_id(url)
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            papers.append(PaperCard(
                title=title,
                authors=[],
                year=0,
                abstract=snippet,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                source="serper",
                method_category="",
            ))

        logger.info("Serper: query='%s' → %d arXiv IDs", query, len(papers))
        return papers

    def search_arxiv_ids(self, query: str, max_results: int = 10) -> list[str]:
        """仅返回 arXiv ID 列表（不获取论文详情）"""
        cards = self.search(query, max_results=max_results)
        ids = []
        for c in cards:
            m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]+)", c.url)
            if m:
                ids.append(m.group(1))
        return ids

    def _post(self, url: str, **kwargs: Any) -> httpx.Response:
        """POST + 重试"""
        client = httpx.Client(timeout=30.0)
        for attempt in range(self._max_retries):
            try:
                resp = client.post(url, **kwargs)
                if resp.status_code == 429:
                    self._limiter.penalize(5)
                    time.sleep(5)
                    continue
                return resp
            except Exception as e:
                logger.warning("Serper POST attempt %d failed: %s", attempt + 1, e)
                time.sleep(self._retry_delay * (attempt + 1))
        raise RuntimeError(f"Serper POST failed after {self._max_retries} retries")


def _extract_arxiv_id(url: str) -> str:
    """从 URL 提取 arXiv ID"""
    if not url:
        return ""
    # 匹配 arxiv.org/abs/2301.00001 或 arxiv.org/abs/2301.00001v2
    m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]+)", url)
    return m.group(1) if m else ""