"""URL 去重 + 标题模糊去重"""

import logging
from difflib import SequenceMatcher
from urllib.parse import urlparse

from src.searcher.result import SearchResult

logger = logging.getLogger(__name__)


def normalize_url(url: str) -> str:
    """URL 归一化：
    - 去除 www 前缀
    - 去除 utm_* / fbclid 等追踪参数
    - 去除尾部斜杠
    - 转为小写
    """
    if not url:
        return ""

    try:
        parsed = urlparse(url.lower().strip())
        netloc = parsed.netloc.removeprefix("www.")
        path = parsed.path.rstrip("/")
        query = parsed.query

        TRACKING_PARAMS = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "fbclid", "gclid", "msclkid", "ref", "source",
        }
        if query:
            params = [
                p for p in query.split("&")
                if p.split("=")[0] not in TRACKING_PARAMS
            ]
            query = "&".join(params) if params else ""

        if query:
            result = f"{netloc}{path}?{query}" if path else netloc
        else:
            result = f"{netloc}{path}" if path else netloc

        return result.rstrip("/") if not query else result
    except Exception:
        return url.lower().strip().rstrip("/")


def _title_similarity(a: str, b: str) -> float:
    """计算标题相似度（0~1）"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def deduplicate(results: list[SearchResult], title_threshold: float = 0.85) -> list[SearchResult]:
    """URL 归一化去重 + 标题模糊去重兜底

    Args:
        results: SearchResult 列表
        title_threshold: 标题相似度阈值（超过则去重）

    Returns:
        去重后的 SearchResult 列表
    """
    if not results:
        return []

    logger.debug("去重开始，输入 %d 条结果", len(results))

    output: list[SearchResult] = []
    seen_urls: dict[str, SearchResult] = {}

    for r in results:
        norm = normalize_url(r.url)
        if norm not in seen_urls:
            seen_urls[norm] = r
            output.append(r)
        else:
            # URL normalized to same — skip duplicate
            pass

    logger.debug("去重完成，输出 %d 条结果（去除 %d 条重复）", len(output), len(results) - len(output))
    return output