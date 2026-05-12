"""SerpAPIRetriever — Google 搜索通过 SerpAPI (serpapi.com)

PaSa 架构的 Stage 1 检索策略：SerpAPI Google 搜索 → 提取 arXiv ID → 用 ArxivRetriever 获取详情

API 文档: https://serpapi.com
限速: 100 次/月（免费）；付费可提升
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


class SerpAPIRetriever(BaseRetriever):
    """SerpAPI Google 搜索检索器

    通过 SerpAPI 的 Google 搜索提取 arXiv 论文 ID，
    再用 ArxivRetriever 获取标题/摘要。

    限速策略:
    - 免费版 100 searches/month，保守使用
    - RateLimiter 保底 1 req/s

    用法:
        with SerpAPIRetriever() as serp:
            papers = serp.search("LoRA fine-tuning site:arxiv.org", max_results=20)
    """

    name = "serpapi"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self._api_key = api_key or settings.serpapi_api_key or settings.serper_api_key
        self._limiter = get_source_limiter("serpapi")

    def search(self, query: str, max_results: int | None = None) -> list[PaperCard]:
        """搜索论文，返回 PaperCard 列表（标题/摘要/URL=arxiv_id）
        
        使用 google_scholar 引擎，支持 author: / all: / ti: 等学术查询语法。
        不附加 site:arxiv.org（google_scholar 用 source: 过滤，或直接依赖查询语义）。
        google_scholar 单次最多 20 条。
        """
        if max_results is None:
            max_results = 20

        if not self._api_key:
            logger.warning("SerpAPIRetriever: 无 API Key，跳过搜索")
            return []

        headers = {"Accept": "application/json"}
        all_results: list[dict] = []

        # google_scholar 单次最多 20 条，分页获取（start 偏移）
        page_size = 20
        for start in range(0, max_results, page_size):
            self._limiter.wait()
            params = {
                "q": query,  # 不附加 site:arxiv.org，让 google_scholar 自己处理
                "engine": "google_scholar",
                "num": min(page_size, max_results - start),
                "start": start,
                "api_key": self._api_key,
            }

            try:
                resp = httpx.get(
                    "https://serpapi.com/search",
                    params=params,
                    headers=headers,
                    timeout=30.0,
                )
                if resp.status_code == 429:
                    self._limiter.penalize(10)
                    time.sleep(10)
                    continue
                if resp.status_code != 200:
                    logger.error("SerpAPI error %d: %s", resp.status_code, resp.text[:200])
                    break

                data = resp.json()
                items = data.get("organic_results", [])
                all_results.extend(items)

                if len(items) < 10:
                    break
                time.sleep(1.5)  # 两次请求间隔，礼貌限速
            except Exception as e:
                logger.error("SerpAPI search failed [start=%d]: %s", start, e)
                break

        # 解析结果，提取 arXiv ID
        papers: list[PaperCard] = []
        seen_ids: set[str] = set()

        for item in all_results:
            url = item.get("link", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")

            # 优先从 URL 提取 arXiv ID；若无，尝试从 snippet/title 提取
            arxiv_id = _extract_arxiv_id(url)
            if not arxiv_id:
                arxiv_id = _extract_arxiv_id_from_text(snippet + " " + title)
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            papers.append(PaperCard(
                title=title,
                authors=[],
                year=0,
                abstract=snippet,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                source="serpapi",
                method_category="",
            ))

        logger.info("SerpAPI: query='%s' → %d arXiv IDs", query, len(papers))
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


def _extract_arxiv_id(url: str) -> str:
    """从 URL 提取 arXiv ID（如 https://arxiv.org/abs/2106.09685 → 2106.09685）"""
    if not url:
        return ""
    m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]+)", url)
    return m.group(1) if m else ""


def _extract_arxiv_id_from_text(text: str) -> str:
    """从文本（snippet/title）中提取 arXiv ID
    
    google_scholar 的结果链接往往指向 IEEE/ACM，但 snippet 中会提到 arXiv 版本。
    匹配模式：
      - "arXiv:2401.12345"
      - "arXiv preprint arXiv:2401.12345"
      - "arxiv.org/abs/2401.12345"
      - "arXiv.Org, 2024" （从标题中提取，精度较低）
    """
    if not text:
        return ""
    # 优先匹配明确的 arXiv: ID 格式
    m = re.search(r"(?:arXiv|arxiv)[.:]\s*([0-9]{4}\.[0-9]+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # 次选：匹配 arxiv.org/abs/ID
    m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # 最后：尝试从 "arXiv.Org, YYYY" 格式中猜 ID（精度低，仅作备选）
    m = re.search(r"arXiv\.Org[,月]?\s*(\d{4})", text, re.IGNORECASE)
    if m:
        # 这种格式只有年份，无法确定具体 ID，跳过（精度太低）
        pass
    return ""