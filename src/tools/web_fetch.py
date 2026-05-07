"""Web 页面抓取——支持 HTML / PDF，返回结构化文本

用法:
    from src.tools import web_fetch, WebFetchResult

    result = web_fetch("https://example.com/paper")
    print(result.title, result.content[:500])
"""

from __future__ import annotations

import logging
import re
import sys
import time
from typing import Any

import httpx

from src.tools.exceptions import ConfigurationError, FetchError, RateLimitError
from src.tools.models import WebFetchResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────

_MAX_CONTENT_LENGTH = 5000  # 单次返回最大字符数（LLM context 友好截断）
_MAX_HTML_LENGTH = 10000  # 原始 HTML 超过此值先截断再提取正文
_MIN_TEXT_LENGTH = 50  # 提取后正文少于此值认为提取失败，raw 降级
_FETCH_MIN_INTERVAL = 2.0  # 同一域名请求间隔（秒）
_SOURCE_RATE_LIMITS: dict[str, float] = {
    "arxiv.org": 3.0,
    "github.com": 2.0,
    "huggingface.co": 1.5,
    "arxiv.org": 3.0,
}
"""按域名配置的特殊限速（秒）"""

_pdf_prefixes = (
    "%PDF",
    "\x25PDF",
    "JVBER",
    "0x25504446",
)


# ──────────────────────────────────────────────────────────────
# HTML 正文提取（零外部依赖）
# ──────────────────────────────────────────────────────────────

_ARTICLE_TAGS = {
    "article", "main", "div", "section",
    "header", "footer", "nav", "aside",
}
_SKIP_TAGS = {
    "script", "style", "noscript", "iframe",
    "svg", "canvas", "form", "input", "button",
}


def _strip_html(html: str) -> str:
    """移除 <style>、<script>、HTML 注释，提取纯文本"""
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.IGNORECASE)
    # 换行保留
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</p>", "\n", html, flags=re.IGNORECASE)
    # 移除所有剩余标签
    html = re.sub(r"<[^>]+>", "", html)
    # HTML 实体解码
    html = html.replace("&nbsp;", " ").replace("&amp;", "&")
    html = html.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    # 空白压缩
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def _extract_title(html: str, fallback_url: str) -> str:
    """从 <title> 或 <meta og:title> 提取标题"""
    # og:title
    m = re.search(r'<meta\s[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if not m:
        m = re.search(r'<meta\s[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:title["\']', html, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # <title>
    m = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return fallback_url


# ──────────────────────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────────────────────

def _domain_from_url(url: str) -> str:
    m = re.search(r"https?://([^/]+)", url)
    return m.group(1) if m else ""


_last_fetch_times: dict[str, float] = {}
_fetch_global_lock = __import__("threading").Lock()


def _respectful_get(url: str, client: httpx.Client) -> httpx.Response:
    """带域名级速率控制的 GET"""
    domain = _domain_from_url(url)

    with _fetch_global_lock:
        last = _last_fetch_times.get(domain, 0.0)
        interval = _SOURCE_RATE_LIMITS.get(domain, _FETCH_MIN_INTERVAL)
        elapsed = time.monotonic() - last
        if elapsed < interval:
            sleep_for = interval - elapsed
            logger.debug("web_fetch: domain %s rate-limit sleep %.2fs", domain, sleep_for)
            time.sleep(sleep_for)
        _last_fetch_times[domain] = time.monotonic()

    return client.get(url)


def _fetch_html(url: str, timeout: float = 30.0) -> tuple[str, str, int]:
    """FETCH HTML page, return (raw_html, content_type, status_code)"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ReviewForge/1.0; "
            "+https://github.com/reviewforge)"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*",
        "Accept-Encoding": "gzip, deflate",
    }
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = _respectful_get(url, client)

    raw_html = resp.text
    content_type = resp.headers.get("Content-Type", "")
    status_code = resp.status_code

    return raw_html, content_type, status_code


def _fetch_pdf_bytes(url: str, timeout: float = 30.0) -> tuple[bytes, int]:
    """Fetch raw PDF bytes"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ReviewForge/1.0; "
            "+https://github.com/reviewforge)"
        ),
        "Accept": "application/pdf,*/*",
    }
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = _respectful_get(url, client)
    return resp.content, resp.status_code


# ──────────────────────────────────────────────────────────────
# 公开 API
# ──────────────────────────────────────────────────────────────

def web_fetch(
    url: str,
    *,
    max_chars: int = _MAX_CONTENT_LENGTH,
    timeout: float = 30.0,
) -> WebFetchResult:
    """抓取 URL，返回结构和正文

    参数:
        url: 目标页面 URL
        max_chars: 正文最大字符数（截断，默认为 {_MAX_CONTENT_LENGTH}）
        timeout: 请求超时（秒）

    返回:
        WebFetchResult

        - content: 提取的纯文本（HTML 页面）
        - raw_content: 原始 HTML（调试用）
        - is_html / is_pdf: 内容类型标记

    异常:
        FetchError: 页面不存在 / 超时 / 服务器错误
        RateLimitError: 触发目标站点的速率限制
        ConfigurationError: 非 HTTPS 或无效 URL

    用法示例:
        result = web_fetch("https://arxiv.org/abs/2301.00001")
        print(result.title)
        print(result.content[:500])
    """
    logger.info("web_fetch: '%s'", url)

    # ── 前置校验 ─────────────────────────────────────────────
    if not url.startswith("http"):
        raise ConfigurationError(f"URL must start with https://, got: {url}")

    # ── PDF 检测（通过 HEAD）───────────────────────────────────
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            head_resp = client.head(url)
        content_type = head_resp.headers.get("Content-Type", "")
        is_pdf_url = "pdf" in content_type.lower() or url.lower().endswith(".pdf")
    except Exception:
        is_pdf_url = url.lower().endswith(".pdf")

    if is_pdf_url:
        raw_bytes, status = _fetch_pdf_bytes(url, timeout)
        # PDF 无法零依赖解析，仅标记，返回原始字节的十六进制前缀作内容
        content = raw_bytes[:max_chars].hex()
        return WebFetchResult(
            url=url,
            title=url.split("/")[-1],
            content=f"[PDF binary data, first {len(raw_bytes)} bytes]",
            raw_content=content,
            content_type="application/pdf",
            status_code=status,
        )

    # ── HTML 抓取 ────────────────────────────────────────────
    raw_html, content_type, status_code = _fetch_html(url, timeout)

    if status_code == 404:
        raise FetchError(url=url, status_code=404, cause="Not Found")
    if status_code >= 500:
        raise FetchError(url=url, status_code=status_code, cause="Server Error")

    # 截断原始 HTML 再提取正文（避免大页面内存问题）
    if len(raw_html) > _MAX_HTML_LENGTH:
        raw_html = raw_html[:_MAX_HTML_LENGTH]
        logger.warning("web_fetch: raw_html truncated for %s", url)

    title = _extract_title(raw_html, fallback_url=url)
    text = _strip_html(raw_html)

    # 降级策略：提取文本过少时用 raw HTML
    if len(text) < _MIN_TEXT_LENGTH and len(raw_html) > _MIN_TEXT_LENGTH:
        logger.warning("web_fetch: text extraction weak for %s, using raw", url)
        text = raw_html[:max_chars]
    elif len(text) > max_chars:
        text = text[:max_chars]

    return WebFetchResult(
        url=url,
        title=title,
        content=text,
        raw_content=raw_html[:_MAX_HTML_LENGTH],
        content_type=content_type,
        status_code=status_code,
    )
