"""Web 工具包内部异常"""

from dataclasses import dataclass


class WebToolError(Exception):
    """Web 工具基异常"""


@dataclass(frozen=True)
class SearchError(WebToolError):
    """搜索引擎调用失败"""

    query: str
    cause: str

    def __str__(self) -> str:
        return f"SearchError(query='{self.query}', cause={self.cause})"


@dataclass(frozen=True)
class FetchError(WebToolError):
    """页面抓取失败"""

    url: str
    status_code: int | None = None
    cause: str = ""

    def __str__(self) -> str:
        base = f"FetchError(url='{self.url}'"
        if self.status_code is not None:
            base += f", status={self.status_code}"
        if self.cause:
            base += f", cause={self.cause}"
        return base + ")"


class RateLimitError(WebToolError):
    """速率限制触发"""


class ConfigurationError(WebToolError):
    """配置缺失（API Key 等）"""
