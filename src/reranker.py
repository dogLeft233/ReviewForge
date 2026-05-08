"""重排序模型客户端——从 config 读取配置，对文档进行相关性排序

用法:
    from src.reranker import RerankerClient

    client = RerankerClient()
    results = client.rerank(
        query="LoRA fine-tuning",
        documents=["doc1", "doc2", "doc3"],
        top_n=2,
    )
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

_RERANK_PATH = "/rerank"


class RerankerClient:
    """重排序模型调用客户端

    从 config.json 读取配置（reranker_base_url / reranker_api_key / reranker_model）。
    支持任意兼容 SiliconFlow Rerank API 格式的 provider。

    Attributes:
        model: 当前使用的 reranker 模型名
    """

    def __init__(self) -> None:
        self._api_key = settings.reranker_api_key or settings.llm_api_key or None
        if self._api_key is None:
            raise ValueError(
                "Reranker API key not set. "
                "Set env var LLM_API_KEY or reranker_api_key in config.json"
            )

        self._base_url = (
            settings.reranker_base_url
            or settings.embedding_base_url
            or settings.llm_base_url
            or "https://api.siliconflow.cn/v1"
        ).rstrip("/")
        self.model = settings.reranker_model

        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        logger.debug(
            "RerankerClient init: model=%s base=%s",
            self.model, self._base_url,
        )

    # ── 公开接口 ─────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        return_documents: bool = True,
    ) -> list[RerankResult]:
        """对文档列表进行重排序，返回按相关性排序的结果

        Args:
            query: 搜索查询
            documents: 待排序的文档列表
            top_n: 返回前 n 条结果（None = 返回全部）
            return_documents: 响应中是否包含文档内容

        Returns:
            RerankResult 列表，按 relevance_score 降序排列
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": return_documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        url = f"{self._base_url}{_RERANK_PATH}"
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers=self._headers)

        match resp.status_code:
            case 200:
                data = resp.json()
                return self._parse_results(data)
            case 401 | 403:
                raise APIError(
                    status_code=resp.status_code,
                    detail=f"Auth failed: {resp.text[:200]}",
                )
            case 429:
                raise RateLimitError(f"Rate limited: {resp.text[:200]}")
            case 500 | 502 | 503 | 504:
                raise APIError(
                    status_code=resp.status_code,
                    detail=f"Server error: {resp.text[:200]}",
                )
            case _:
                raise APIError(
                    status_code=resp.status_code,
                    detail=resp.text[:300],
                )

    # ── 内部实现 ─────────────────────────────────────────────

    @staticmethod
    def _parse_results(data: dict[str, Any]) -> list[RerankResult]:
        raw = data.get("results", [])
        return [RerankResult(**item) for item in raw]

    def __enter__(self) -> RerankerClient:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ── 数据模型 ────────────────────────────────────────────────


class RerankResult:
    """单条重排序结果"""

    def __init__(
        self,
        index: int,
        relevance_score: float,
        document: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.relevance_score = relevance_score
        self.document = document  # {"text": "..."} when return_documents=True

    def __repr__(self) -> str:
        return f"RerankResult(index={self.index}, score={self.relevance_score:.4f})"


# ── 异常 ───────────────────────────────────────────────────


class RerankerError(Exception):
    """Reranker 调用基异常"""


class APIError(RerankerError):
    """API 调用失败"""

    def __init__(self, status_code: int | None = None, detail: str = "") -> None:
        self.status_code = status_code
        self.detail = detail

    def __str__(self) -> str:
        base = f"APIError(status={self.status_code})"
        if self.detail:
            base += f": {self.detail}"
        return base


class RateLimitError(RerankerError):
    """触发速率限制"""
