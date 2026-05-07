"""HuggingFace Hub 检索器 — 免费使用，无需 API Key"""

from urllib.parse import quote

from src.models import PaperCard, ResourceCard
from src.retrievers.base import BaseRetriever

import logging

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api"


class HuggingFaceRetriever(BaseRetriever):
    """HuggingFace Hub API 检索器（免费，无需 API Key）

    - 不返回论文（search() 返回空列表）
    - 返回模型/数据集资源（search_resources()、search_models()、search_datasets()）

    用法:
        with HuggingFaceRetriever() as hf:
            models = hf.search_models("text generation")
            datasets = hf.search_datasets("wikipedia")
            all_resources = hf.search_resources("LoRA")
    """

    name = "huggingface"

    def search(
        self, query: str, max_results: int | None = None
    ) -> list[PaperCard]:
        """HuggingFace 不直接返回论文，返回空列表"""
        return []

    def search_resources(
        self, query: str, max_results: int | None = None
    ) -> list[ResourceCard]:
        """同时搜索模型和数据集（合并结果）

        Args:
            query: 搜索关键词
            max_results: 每类返回数（默认 5，总计最多 10）
        """
        if max_results is None:
            max_results = 5

        models = self.search_models(query, max_results)
        datasets = self.search_datasets(query, max_results)
        return models + datasets

    def search_models(
        self, query: str, max_results: int = 10
    ) -> list[ResourceCard]:
        """按关键词搜索模型，按下载量降序

        Args:
            query: 搜索关键词
            max_results: 最多返回数
        """
        url = (
            f"{HF_API}/models"
            f"?search={quote(query)}&sort=downloads&direction=-1&limit={max_results}"
        )

        logger.info(
            "huggingface: searching models '%s' (max=%d)", query, max_results
        )
        resp = self._get(url)
        data = resp.json()
        return self._parse_models(data, query)

    def search_datasets(
        self, query: str, max_results: int = 10
    ) -> list[ResourceCard]:
        """按关键词搜索数据集，按下载量降序

        Args:
            query: 搜索关键词
            max_results: 最多返回数
        """
        url = (
            f"{HF_API}/datasets"
            f"?search={quote(query)}&sort=downloads&direction=-1&limit={max_results}"
        )

        logger.info(
            "huggingface: searching datasets '%s' (max=%d)", query, max_results
        )
        resp = self._get(url)
        data = resp.json()
        return self._parse_datasets(data, query)

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
        import re as _re

        cards: list[ResourceCard] = []
        for item in data:
            indicators: list[str] = []
            subsets = item.get("configs", [])
            if subsets:
                indicators.append(f"包含 {len(subsets)} 个子配置")

            desc = item.get("description", "") or ""
            desc = _re.sub(r"<[^>]+>", " ", desc)
            desc = _re.sub(r"\s+", " ", desc).strip()[:200]

            card = ResourceCard(
                name=item.get("id", ""),
                type="dataset",
                url=f"https://huggingface.co/datasets/{item.get('id', '')}",
                description=desc,
                owner=(
                    item.get("id", "").split("/")[0]
                    if "/" in item.get("id", "")
                    else ""
                ),
                quality_indicators=indicators,
                recommended_use=f"可用于 {query} 综述的数据集章节",
                source="huggingface",
            )
            cards.append(card)

        logger.debug("huggingface: parsed %d datasets", len(cards))
        return cards
