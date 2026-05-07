"""AdapterRouter — 智能路由"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.searcher.adapters.base import AdapterBase


@dataclass
class CategoryConfig:
    """单个分类的适配器配置"""

    adapters: list[type["AdapterBase"]]


class AdapterRouter:
    """适配器路由——根据 query 和 category 返回对应的适配器组合"""

    # 固定映射：category → 适配器类型列表
    CATEGORY_ADAPTERS = {
        "papers": ["ArxivAdapter", "SerperAdapter"],
        "resources": ["GithubAdapter", "HuggingFaceAdapter"],
        "news": ["BochaAdapter", "HNAdapter"],
    }

    def __init__(self) -> None:
        self._instances: dict[str, "AdapterBase"] = {}

    def route(self, query: str, category: str) -> list["AdapterBase"]:
        """返回指定 category 对应的适配器实例列表"""
        adapter_names = self.CATEGORY_ADAPTERS.get(category, [])
        result = []
        for name in adapter_names:
            if name not in self._instances:
                self._instances[name] = self._instantiate(name)
            result.append(self._instances[name])
        return result

    @staticmethod
    def _instantiate(name: str) -> "AdapterBase":
        from src.searcher.adapters.arxiv_adapter import ArxivAdapter
        from src.searcher.adapters.serper_adapter import SerperAdapter
        from src.searcher.adapters.github_adapter import GithubAdapter
        from src.searcher.adapters.hf_adapter import HuggingFaceAdapter
        from src.searcher.adapters.bocha_adapter import BochaAdapter
        from src.searcher.adapters.hn_adapter import HNAdapter

        mapping = {
            "ArxivAdapter": ArxivAdapter,
            "SerperAdapter": SerperAdapter,
            "GithubAdapter": GithubAdapter,
            "HuggingFaceAdapter": HuggingFaceAdapter,
            "BochaAdapter": BochaAdapter,
            "HNAdapter": HNAdapter,
        }
        cls = mapping.get(name)
        if cls is None:
            raise ValueError(f"Unknown adapter: {name}")
        return cls()