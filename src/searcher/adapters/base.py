"""适配器基类——封装各检索器调用"""

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.searcher.result import SearchResult


class AdapterBase:
    """适配器基类，对应 src/searcher/base.py 的 BaseAdapter"""

    name: str = ""
    category: str = ""

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list["SearchResult"]:
        ...