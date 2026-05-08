"""搜索工具实现（供 searcher.py 调用）

提供 web_search 和 web_fetch 函数，
底层通过 exec 调用 bocha-search CLI 和 curl 实现。
"""

from __future__ import annotations

import json
import subprocess
from typing import Any


def web_search(query: str, *, count: int = 10, freshness: str | None = None) -> list[dict[str, Any]]:
    """网页搜索

    底层调用 bocha-search CLI。

    Returns:
        list of {title, url, description, siteName, publishedDate}
    """
    cmd = ["bocha", "search", query, "--count", str(count), "--summary"]
    if freshness:
        cmd.extend(["--freshness", freshness])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return [{"error": result.stderr or "search failed"}]

        # bocha search 输出为 JSON 数组（每条含 title/url/description）
        try:
            data = json.loads(result.stdout)
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            elif isinstance(data, list):
                return data
            else:
                return [{"error": f"unexpected output format: {result.stdout[:200]}"}]
        except json.JSONDecodeError:
            return [{"error": f"json decode failed: {result.stdout[:200]}"}]
    except subprocess.TimeoutExpired:
        return [{"error": "search timeout"}]
    except FileNotFoundError:
        return [{"error": "bocha CLI not found, install: npm install -g bocha-search"}]


def web_fetch(url: str, *, max_chars: int = 5000, timeout: float = 30.0) -> dict[str, Any]:
    """页面抓取

    底层用 curl 抓取，提取正文（尝试 title + meta description）。

    Returns:
        {title, content, status_code}
    """
    import re

    try:
        # 用 curl 抓取 HTML
        proc = subprocess.run(
            [
                "curl", "-s", "-L", "--max-time", str(int(timeout)),
                "-A",
                "Mozilla/5.0 (compatible; ReviewForge/1.0; +https://example.com/bot)",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )

        html = proc.stdout
        if not html:
            return {"title": url, "content": "", "status_code": 0, "error": "empty response"}

        # 提取 <title>
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else url

        # 提取 meta description
        desc_match = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE,
        )
        if not desc_match:
            desc_match = re.search(
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']description["\']',
                html, re.IGNORECASE,
            )
        description = desc_match.group(1).strip() if desc_match else ""

        # 简单正文提取：移除 script/style/title 标签，压缩空白
        content = re.sub(r"(?si)<script[^>]*>.*?</script>", "", html)
        content = re.sub(r"(?si)<style[^>]*>.*?</style>", "", content)
        content = re.sub(r"(?si)<nav[^>]*>.*?</nav>", "", content)
        content = re.sub(r"(?si)<footer[^>]*>.*?</footer>", "", content)
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).strip()

        status_code = 200 if proc.returncode == 0 else proc.returncode

        return {
            "title": title,
            "content": content[:max_chars],
            "status_code": status_code,
        }

    except subprocess.TimeoutExpired:
        return {"title": url, "content": "", "status_code": -1, "error": "timeout"}
    except Exception as e:
        return {"title": url, "content": "", "status_code": -1, "error": str(e)}


def execute(tool_name: str, kwargs: dict[str, Any]) -> str:
    """统一执行入口（供 searcher.py 调用）"""
    if tool_name == "web_search":
        query = kwargs.get("query", "")
        count = kwargs.get("count", 5)
        results = web_search(query, count=count)
        return _format_search_results(results)
    elif tool_name == "web_fetch":
        url = kwargs.get("url", "")
        max_chars = kwargs.get("max_chars", 3000)
        result = web_fetch(url, max_chars=max_chars)
        return _format_fetch_result(result)
    else:
        return f"[unknown tool: {tool_name}]"


def _format_search_results(results: list) -> str:
    if not results:
        return "（无搜索结果）"
    lines = []
    for r in results[:5]:
        title = r.get("title", "")
        url = r.get("url", "")
        desc = r.get("description", r.get("snippet", ""))
        lines.append(f"- {title}\n  URL: {url}\n  {str(desc)[:200]}")
    return "\n".join(lines) if lines else "（无结果）"


def _format_fetch_result(result: dict) -> str:
    title = result.get("title", "")
    content = result.get("content", "")
    error = result.get("error", "")
    if error:
        return f"抓取失败: {error}"
    return f"# {title}\n\n{content[:3000]}"