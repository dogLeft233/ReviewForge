"""SerpAPIRetriever — Google 搜索通过 Serper API (google.serper.dev)

PaSa 架构的 Stage 1 检索策略：Serper Google 搜索 → 提取 arXiv ID → 用 Ar5ivRetriever 获取详情

API 文档: https://serper.dev
认证: X-API-Key header
端点:
  - https://google.serper.dev/search (普通搜索)
  - https://google.serper.dev/scholar (学术搜索)
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

SERPER_SEARCH_API = "https://google.serper.dev/search"
SERPER_SCHOLAR_API = "https://google.serper.dev/scholar"


class SerpAPIRetriever(BaseRetriever):
    """Serper API Google 搜索检索器

    支持两种搜索引擎：
    - google_scholar: 学术搜索（POST 到 /scholar，engine="scholar"）
    - google: 普通 Google 搜索 + site:arxiv.org（POST 到 /search）

    限速策略:
    - RateLimiter 保底 1 req/s
    - 监控 credits 消耗

    用法:
        with SerpAPIRetriever() as serp:
            papers = serp.search("LoRA fine-tuning", max_results=20)
            papers = serp.search_google("LoRA fine-tuning", max_results=20)
    """

    name = "serpapi"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self._api_key = api_key or settings.serpapi_api_key or settings.serper_api_key
        self._limiter = get_source_limiter("serpapi")

    # ── 内部通用搜索 ──────────────────────────────────────────

    def _search_engine(
        self,
        query: str,
        engine: str,
        max_results: int,
    ) -> list[dict]:
        """通用搜索接口，内部处理分页和限速。

        Serper API:
        - /scholar 端点用于学术搜索，engine="scholar"
        - /search 端点用于普通搜索，engine="search"（或省略）
        """
        all_results: list[dict] = []
        page_size = 10  # Serper 单次最多 10 条

        # Serper API 使用 POST，X-API-Key header
        http_headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }

        # 根据 engine 选择端点
        if engine == "google_scholar":
            api_url = SERPER_SCHOLAR_API
            payload_engine = "scholar"
        else:
            api_url = SERPER_SEARCH_API
            payload_engine = "search"

        # ── 代理/网络环境诊断 ──────────────────────────────────────
        import os, httpx
        env_check = {
            "HTTP_PROXY": os.environ.get("HTTP_PROXY", ""),
            "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", ""),
            "http_proxy": os.environ.get("http_proxy", ""),
            "https_proxy": os.environ.get("https_proxy", ""),
            "NO_PROXY": os.environ.get("NO_PROXY", ""),
            "ALL_PROXY": os.environ.get("ALL_PROXY", ""),
        }
        has_proxy = any(v for v in env_check.values())
        logger.info(
            "[NetworkEnv] proxies configured: %s | env=%s",
            has_proxy, env_check,
        )

        # 显式传递代理给 httpx
        proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or None
        if proxy_url:
            logger.info("[NetworkEnv] using explicit proxy: %s", proxy_url)
        # httpx trust_env status（默认 True，会读取环境变量代理）
        test_client = httpx.Client(timeout=5.0)
        logger.info(
            "[NetworkEnv] httpx Client trust_env=%s (process default)",
            test_client._trust_env,
        )
        test_client.close()
        # ── 代理/网络环境诊断结束 ──────────────────────────────────

        for start in range(0, max_results, page_size):
            self._limiter.wait()

            payload: dict[str, Any] = {
                "q": query,
                "num": min(page_size, max_results - start),
            }
            if start > 0:
                payload["page"] = start // page_size + 1

            try:
                resp = httpx.post(
                    api_url,
                    json=payload,
                    headers=http_headers,
                    timeout=30.0,
                    proxy=proxy_url,
                    trust_env=True,
                )

                if resp.status_code == 429:
                    self._limiter.penalize(10)
                    time.sleep(10)
                    continue
                if resp.status_code in (401, 403):
                    logger.warning("Serper %s auth error %d, will fallback to google_scholar", engine, resp.status_code)
                    return "AUTH_ERROR"
                if resp.status_code != 200:
                    logger.error("Serper %s error %d: %s", engine, resp.status_code, resp.text[:200])
                    break

                data = resp.json()

                # 监控 credits 消耗
                credits = data.get("credits")
                if credits is not None:
                    logger.debug("Serper credits remaining: %s", credits)

                items = data.get("organic", [])
                all_results.extend(items)

                if len(items) < page_size:
                    break
                time.sleep(1.1)
            except Exception as e:
                logger.error("Serper %s search failed [start=%d]: %s", engine, start, e)
                break

        return all_results

    def _parse_arxiv_papers(self, results: list[dict]) -> list[PaperCard]:
        """从 Serper 搜索结果中解析 arXiv 论文。"""
        papers: list[PaperCard] = []
        seen_ids: set[str] = set()

        for item in results:
            url = item.get("link", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")

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

        return papers

    # ── 原有 google_scholar 接口 ─────────────────────────────

    def search(self, query: str, max_results: int | None = None) -> list[PaperCard]:
        """搜索论文，返回 PaperCard 列表（标题/摘要/URL=arxiv_id）

        使用 google_scholar 引擎（Serper /scholar 端点，engine="scholar"）。
        """
        results = self._search_engine(query, "google_scholar", max_results or 20)
        papers = self._parse_arxiv_papers(results)
        logger.info("SerpAPI(google_scholar): query='%s' → %d arXiv IDs", query, len(papers))
        return papers

    # ── PaSa 风格 Google 搜索（核心新增）────────────────────

    def search_google(self, query: str, max_results: int = 20) -> list[PaperCard]:
        """PaSa 风格搜索：使用普通 Google 引擎 + site:arxiv.org

        用于 Stage 1 搜索词搜索，和 Stage 2 引文标题搜索。
        Google 引擎配合 site:arxiv.org 可以高效获取 arXiv ID。
        如果 Google 引擎返回 401/403，立刻重试用 google_scholar 代替。

        Args:
            query: 搜索词（不要手动加 site:arxiv.org，方法内自动附加）
            max_results: 最大结果数
        """
        q = f"{query} site:arxiv.org"
        results = self._search_engine(q, "google", max_results)
        # Fallback: google engine 认证失败时用 google_scholar
        if results == "AUTH_ERROR":
            logger.warning("Google engine auth failed, falling back to google_scholar")
            results = self._search_engine(q, "google_scholar", max_results)
        papers = self._parse_arxiv_papers(results)
        logger.info("SerpAPI(google+arxiv): query='%s' → %d arXiv IDs", query, len(papers))
        return papers

    def search_ref_by_title(self, title: str, max_results: int = 3) -> list[PaperCard]:
        """用论文标题搜索获取 arXiv ID（Stage 2 引文扩展专用）

        对 bibliography 中的参考文献标题执行 Google 搜索，
        获取第一个结果的 arXiv ID。

        Args:
            title: 参考文献标题
            max_results: 最多搜索多少个结果（默认取第一个）
        """
        results = self._search_engine(title, "google", max_results)
        papers = self._parse_arxiv_papers(results)
        logger.debug("search_ref_by_title('%s') → %d arXiv IDs", title[:60], len(papers))
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

    Google 搜索的结果链接往往指向 IEEE/ACM，但 snippet 中会提到 arXiv 版本。
    匹配模式：
      - "arXiv:2401.12345"
      - "arXiv preprint arXiv:2401.12345"
      - "arxiv.org/abs/2401.12345"
    """
    if not text:
        return ""
    m = re.search(r"(?:arXiv|arxiv)[.:]\s*([0-9]{4}\.[0-9]+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return ""
