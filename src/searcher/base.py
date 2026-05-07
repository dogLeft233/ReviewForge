"""BaseAdapter 抽象基类——所有检索适配器的接口"""

from abc import ABC, abstractmethod

from src.searcher.result import SearchResult


class BaseAdapter(ABC):
    """检索适配器基类"""

    name: str = ""
    category: str = ""  # "papers" | "resources" | "news"

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """执行搜索，返回 SearchResult 列表"""
        ...

    def can_handle(self, query: str) -> bool:
        """判断是否能处理该 query（可选覆盖）"""
        return True