"""Web 工具包——供 LLM 调用的大模型友好搜索/抓取接口"""

from .web_search import WebSearchResult, web_search
from .web_fetch import WebFetchResult, web_fetch

__all__ = [
    "WebSearchResult",
    "web_search",
    "WebFetchResult",
    "web_fetch",
]
