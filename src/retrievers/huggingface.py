"""HuggingFace 模型/数据集检索器"""

from urllib.parse import quote

from src.models import ResourceCard
from src.retrievers.base import BaseRetriever

import logging

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api"


class HuggingFaceRetriever(BaseRetriever):
    """HuggingFace Hub API 检索器（免费，无需 API Key）"""

    name = "huggingface"

    def search(
        self, query: str, max_results: int | None = None
    ) -> list:
        """HuggingFace 不直接返回论文，返回空列表"""
        return []

    def search_models(
        self, query: str, max_results: int = 10
    ) -> list[ResourceCard]:
        """按关键词搜索模型"""
        url = f"{HF_API}/models?search={quote(query)}&sort=downloads&direction=-1&limit={max_results}"

        logger.info("huggingface: searching models '%s' (max=%d)", query, max_results)
        resp = self._get(url)
        data = resp.json()
        return self._parse_models(data, query)

    def search_datasets(
        self, query: str, max_results: int = 10
    ) -> list[ResourceCard]:
        """按关键词搜索数据集"""
        url = f"{HF_API}/datasets?search={quote(query)}&sort=downloads&direction=-1&limit={max_results}"

        logger.info(
            "huggingface: searching datasets '%s' (max=%d)", query, max_results
        )
        resp = self._get(url)
        data = resp.json()
        return self._parse_datasets(data, query)

    def search_resources(
        self, query: str, max_results: int | None = None
    ) -> list[ResourceCard]:
        """同时搜索模型和数据集的便捷方法"""
        if max_results is None:
            max_results = 5

        models = self.search_models(query, max_results)
        datasets = self.search_datasets(query, max_results)
        return models + datasets

    # ── 解析 ──

    @staticmethod
    def _parse_models(data: list[dict], query: str) -> list[ResourceCard]:
        cards: list[ResourceCard] = []
        for item in data:
            indicators: list[str] = []
            pipelines = item.get("pipeline_tags", [])
            if pipelines:
                indicators.append(f"Pipeline: {', '.join(pipelines)}")
            if item.get("likes", 0) > 100:
                indicators.append("高认可度")

            card = ResourceCard(
                name=item.get("modelId") or item.get("id", ""),
                type="model",
                url=f"https://huggingface.co/{item.get('modelId', '')}",
                description=item.get("config", {}).get("model_type", ""),
                stars=item.get("likes", 0),
                quality_indicators=indicators,
                recommended_use=f"可用于 {query} 综述的模型资源章节",
                source="huggingface",
            )
            cards.append(card)

        logger.debug("huggingface: parsed %d models", len(cards))
        return cards

    @staticmethod
    def _parse_datasets(data: list[dict], query: str) -> list[ResourceCard]:
        cards: list[ResourceCard] = []
        for item in data:
            indicators: list[str] = []
            subsets = item.get("configs", [])
            if subsets:
                indicators.append(f"包含 {len(subsets)} 个子配置")

            desc = item.get("description", "") or ""
            # 去除 HTML 标签
            desc = __import__("re").sub(r"<[^>]+>", " ", desc)
            desc = __import__("re").sub(r"\s+", " ", desc).strip()[:200]

            card = ResourceCard(
                name=item.get("id", ""),
                type="dataset",
                url=f"https://huggingface.co/datasets/{item.get('id', '')}",
                description=desc,
                owner=item.get("id", "").split("/")[0] if "/" in item.get("id", "") else "",
                quality_indicators=indicators,
                recommended_use=f"可用于 {query} 综述的数据集章节",
                source="huggingface",
            )
            cards.append(card)

        logger.debug("huggingface: parsed %d datasets", len(cards))
        return cards
