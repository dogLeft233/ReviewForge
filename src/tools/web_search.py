"""Web 搜索——Bocha AI 搜索 API 封装

用法:
    from src.tools import web_search, WebSearchResult

    results: list[WebSearchResult] = web_search("LLM reasoning survey")
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import httpx

# ── 项目内部导入 ────────────────────────────────────────────

sys.path.insert(0, str(__file__).rsplit("src/", 1)[0])

from src.config import settings
from src.tools.exceptions import ConfigurationError, RateLimitError, SearchError
from src.tools.models import WebSearchResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────

_BOCHA_BASE = "https://api.bocha.cn/v1/web-search"
"""Bocha Search API 地址（POST + JSON body）"""

_BOCHA_MAX_PER_PAGE = 50
_BOCHA_DEFAULT_COUNT = 10
_BOCHA_MIN_INTERVAL = 1.0  # 无 key 时保守 1 req/s


# ──────────────────────────────────────────────────────────────
# 速率控制
# ──────────────────────────────────────────────────────────────

_last_search_time: float = 0.0
_search_lock: time = None  # placeholder — real lock below


import threading

_rate_lock = threading.Lock()
_last_search_mono: float = 0.0


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────


def _bocha_search_raw(
    query: str,
    *,
    count: int = _BOCHA_DEFAULT_COUNT,
    freshness: str | None = None,
    summary: bool = True,
) -> dict[str, Any]:
    """发送原始搜索请求，返回字典（含原始错误信息）"""

    api_key = getattr(settings, "bocha_api_key", "")
    if not api_key:
        raise ConfigurationError(
            "BOCHA_API_KEY not set. "
            "Set env var BOCHA_API_KEY or bocha_api_key in config.json"
        )

    # 速率控制
    global _last_search_mono
    with _rate_lock:
        elapsed = time.monotonic() - _last_search_mono
        if elapsed < _BOCHA_MIN_INTERVAL:
            sleep_for = _BOCHA_MIN_INTERVAL - elapsed
            logger.debug("bocha_search: rate-limiting sleep %.2fs", sleep_for)
            time.sleep(sleep_for)
        _last_search_mono = time.monotonic()

    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    params: dict[str, Any] = {
        "query": query,
        "count": min(count, _BOCHA_MAX_PER_PAGE),
        "summary": summary,
    }
    if freshness:
        params["freshness"] = freshness

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.post(_BOCHA_BASE, headers=headers, json=params)

    match resp.status_code:
        case 200:
            data = resp.json()
            code = data.get("code", 0)
            if code != 200:
                msg = data.get("msg", f"Bocha API error (code={code})")
                raise SearchError(query=query, cause=msg)
            return data
        case 429:
            raise RateLimitError(f"Bocha API rate limited for query: {query}")
        case 401 | 403:
            raise ConfigurationError(
                f"Bocha API auth failed (status {resp.status_code}). "
                "Check your BOCHA_API_KEY."
            )
        case _:
            raise SearchError(
                query=query,
                cause=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )


def _parse_bocha_results(raw: dict[str, Any]) -> list[WebSearchResult]:
    """解析 Bocha API（v1/web-search）响应为 WebSearchResult 列表

    Bocha v1 响应结构:
        {"code": 200, "data": {"webPages": {"value": [{"name": ..., "url": ..., "snippet": ...}]}}}
    """
    pages = raw.get("data", {}).get("webPages", {})
    items = pages.get("value", [])
    if not isinstance(items, list):
        items = []
    results: list[WebSearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            WebSearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                description=item.get("snippet", ""),
                site_name=item.get("siteName", ""),
                published_date=item.get("datePublished", ""),
                raw=item,
            )
        )
    return results


# ──────────────────────────────────────────────────────────────
# 公开 API
# ──────────────────────────────────────────────────────────────


def web_search(
    query: str,
    *,
    count: int = _BOCHA_DEFAULT_COUNT,
    freshness: str | None = None,
    summary: bool = True,
) -> list[WebSearchResult]:
    """搜索网页，返回结构化结果

    参数:
        query: 搜索关键词
        count: 返回结果数量（1-{_BOCHA_MAX_PER_PAGE}）
        freshness: 时间范围过滤（"oneDay"|"oneWeek"|"oneMonth"|"oneYear"）
        summary: 是否包含网页摘要（会增加响应时间）

    返回:
        WebSearchResult 列表

    异常:
        ConfigurationError: BOCHA_API_KEY 未配置
        RateLimitError: 触发速率限制
        SearchError: 搜索引擎调用失败

    用法示例:
        results = web_search("LLM reasoning survey", count=10, freshness="oneYear")
        for r in results:
            print(f"{r.title} — {r.url}")
    """
    logger.info("web_search: '%s' (count=%d, freshness=%s)", query, count, freshness)

    raw = _bocha_search_raw(
        query,
        count=count,
        freshness=freshness,
        summary=summary,
    )

    results = _parse_bocha_results(raw)
    logger.debug("web_search: got %d results for '%s'", len(results), query)
    return results
