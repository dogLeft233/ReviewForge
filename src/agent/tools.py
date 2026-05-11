"""工具注册和管理——将项目已有工具注册为 LangChain Tool"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool

from src.tools.web_search import web_search, WebSearchResult
from src.tools.web_fetch import web_fetch, WebFetchResult
from src.logging_config import get_logger

logger = get_logger(__name__)


def _format_web_search_result(results: list[WebSearchResult]) -> str:
    """将 web_search 结果格式化为字符串"""
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"- **{r.title}** ({r.site_name or 'unknown'})")
        lines.append(f"  URL: {r.url}")
        if r.description:
            lines.append(f"  Summary: {r.description}")
        lines.append("")
    return "\n".join(lines)


def _format_web_fetch_result(result: WebFetchResult) -> str:
    """将 web_fetch 结果格式化为字符串"""
    if result.status_code != 200:
        return f"Failed to fetch {result.url}: status {result.status_code}"
    lines = [
        f"Title: {result.title or result.url}",
        f"URL: {result.url}",
        f"Content ({len(result.content)} chars):",
        result.content[:3000],
    ]
    if len(result.content) > 3000:
        lines.append(f"... (truncated, total {len(result.content)} chars)")
    return "\n".join(lines)


def get_tools() -> list[BaseTool]:
    """返回已注册的 LangChain Tool 列表

    工具说明:
        web_search: 网页搜索（Bocha API）
        web_fetch: 页面内容抓取
    """
    tools: list[BaseTool] = []

    # web_search tool
    web_search_description = """Web search tool for finding online information.
Use this when you need to find recent information, facts, or data from the web.
Input is a search query string."""

    async def _web_search_wrapper(query: str, count: int = 10, freshness: str | None = None) -> str:
        """同步 web_search 的包装器"""
        logger.debug("web_search called: query=%s, count=%d, freshness=%s", query, count, freshness)
        results = web_search(query, count=count, freshness=freshness, summary=True)
        logger.debug("web_search returned %d results", len(results))
        return _format_web_search_result(results)

    web_search_tool = BaseTool(
        name="web_search",
        description=web_search_description,
        args_schema={
            "query": {"type": "string", "description": "Search query", "default": ""},
            "count": {"type": "integer", "description": "Number of results (1-50)", "default": 10},
            "freshness": {
                "type": "string",
                "description": "Time filter: oneDay, oneWeek, oneMonth, oneYear",
                "default": None,
            },
        },
    )
    # 直接绑定函数会更简洁
    tools.append(web_search_tool)

    # web_fetch tool
    web_fetch_description = """Fetch web page content from a URL.
Use this after web_search to get detailed information from specific URLs.
Input is a URL string."""

    web_fetch_tool = BaseTool(
        name="web_fetch",
        description=web_fetch_description,
        args_schema={
            "url": {"type": "string", "description": "Target URL", "default": ""},
            "max_chars": {"type": "integer", "description": "Max characters to return", "default": 5000},
            "timeout": {"type": "number", "description": "Request timeout in seconds", "default": 30.0},
        },
    )
    tools.append(web_fetch_tool)

    logger.info("Registered %d tools: %s", len(tools), [t.name for t in tools])
    return tools
