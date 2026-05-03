"""WebPageFetcher — 用 httpx 抓取网页并提取可读内容

相当于 Python 版的 web_fetch 工具，无额外依赖（html.parser 来自标准库）。
"""

import logging
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# 阻塞类内容标签
_BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "blockquote", "pre", "article", "section",
    "td", "th", "br", "tr",
}
# 要忽略的标签
_SKIP_TAGS = {"script", "style", "noscript", "iframe", "svg", "nav", "header", "footer"}
# 内容太多时截断
_MAX_TEXT_CHARS = 8000


@dataclass(slots=True)
class FetchedPage:
    """单个网页抓取结果"""

    url: str
    title: str = ""
    meta_description: str = ""
    text: str = ""
    status_code: int = 0
    content_type: str = ""
    fetch_time_ms: float = 0.0
    error: str = ""

    @property
    def success(self) -> bool:
        return self.status_code == 200 and not self.error

    def summary(self, max_chars: int = 500) -> str:
        """格式化为摘要文本"""
        if self.error:
            return f"[抓取失败] {self.url}: {self.error}"
        lines = [f"标题: {self.title}"]
        if self.meta_description:
            lines.append(f"描述: {self.meta_description}")
        text = self.text[:max_chars]
        lines.append(f"正文 ({len(self.text)} 字符):")
        lines.append(text)
        if len(self.text) > max_chars:
            lines.append("...")
        return "\n".join(lines)

    @classmethod
    def error_result(cls, url: str, error: str) -> "FetchedPage":
        return cls(url=url, error=error, status_code=0)


class _TextExtractor(HTMLParser):
    """从 HTML 中提取纯文本"""

    def __init__(self) -> None:
        super().__init__()
        self._text_parts: list[str] = []
        self._skip_depth = 0
        self._in_pre = False
        self._add_space = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in _SKIP_TAGS:
            self._skip_depth += 1
        if t == "pre":
            self._in_pre = True
        if t in _BLOCK_TAGS:
            self._add_space = True

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if t == "pre":
            self._in_pre = False
        if t in _BLOCK_TAGS:
            self._add_space = True

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = data.strip()
        if not text:
            return
        if self._add_space and self._text_parts:
            self._text_parts.append(" ")
        self._add_space = False
        self._text_parts.append(text)

    def get_text(self) -> str:
        raw = "".join(self._text_parts)
        # 压缩空白
        return re.sub(r"\s{3,}", "\n\n", raw).strip()


def _extract_title(html: str) -> str:
    """从 HTML 中提取 <title> 内容"""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _extract_meta_description(html: str) -> str:
    """从 HTML 中提取 meta description"""
    m = re.search(
        r'<meta\s+[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']',
        html,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    # 交换顺序的写法
    m = re.search(
        r'<meta\s+[^>]*content=["\']([^"\']*)["\'][^>]*name=["\']description["\']',
        html,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return ""


class WebPageFetcher:
    """网页抓取器——fetch URL → 提取可读正文"""

    def __init__(
        self,
        timeout: float = 15.0,
        max_text_chars: int = _MAX_TEXT_CHARS,
        user_agent: str | None = None,
    ) -> None:
        self._timeout = timeout
        self._max_text_chars = max_text_chars
        self._user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

    def fetch(self, url: str) -> FetchedPage:
        """抓取单页"""
        start = time.monotonic()
        try:
            resp = httpx.get(
                url,
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": self._user_agent},
            )
            elapsed = (time.monotonic() - start) * 1000
        except httpx.TimeoutException:
            return FetchedPage.error_result(url, "请求超时")
        except httpx.ConnectError:
            return FetchedPage.error_result(url, "连接失败")
        except httpx.HTTPStatusError as e:
            return FetchedPage.error_result(url, f"HTTP {e.response.status_code}")
        except Exception as e:
            return FetchedPage.error_result(url, f"异常: {e}")

        content_type = resp.headers.get("content-type", "")
        html = resp.text

        title = _extract_title(html)
        meta_desc = _extract_meta_description(html)

        # 文本提取
        extractor = _TextExtractor()
        try:
            extractor.feed(html)
        except Exception:
            pass
        text = extractor.get_text()

        if len(text) > self._max_text_chars:
            text = text[: self._max_text_chars] + "\n\n[...已截断]"

        return FetchedPage(
            url=url,
            title=title,
            meta_description=meta_desc,
            text=text,
            status_code=resp.status_code,
            content_type=content_type,
            fetch_time_ms=round(elapsed, 1),
        )

    def fetch_batch(
        self,
        urls: list[str],
        max_concurrent: int = 3,
    ) -> list[FetchedPage]:
        """并发抓取多页（简单并发，httpx 的异步接口更优但暂不用）"""
        results: list[FetchedPage] = []
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            for url in batch:
                results.append(self.fetch(url))
        return results

    def fetch_results(
        self,
        urls: list[str],
        max_pages: int = 5,
    ) -> list[FetchedPage]:
        """并发抓取结果列表，跳过失败，最多抓 max_pages 页"""
        pages: list[FetchedPage] = []
        for url in urls[:max_pages]:
            page = self.fetch(url)
            if page.success:
                pages.append(page)
        return pages
