"""嵌入模型客户端——从 config 读取配置，对文本进行向量编码

用法:
    from src.embedding import EmbeddingClient

    client = EmbeddingClient()
    vec = client.encode("Hello world")           # 单条 → list[float]
    vecs = client.encode_batch(["a", "b", "c"])  # 批量 → list[list[float]]
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_BATCH = 32
_EMBEDDINGS_PATH = "/embeddings"


class EmbeddingClient:
    """嵌入模型调用客户端

    从 config.json 读取模型和连接配置（llm_base_url / llm_api_key / embedding_model）。
    单条和批量调用共享一个 httpx 客户端，批量编码自动按 batch_size 分块。

    Attributes:
        model: 当前使用的嵌入模型名
        dimensions: 输出向量维度
    """

    def __init__(self) -> None:
        self._api_key = settings.llm_api_key
        if not self._api_key:
            raise ValueError(
                "LLM_API_KEY not set. "
                "Set env var LLM_API_KEY or llm_api_key in config.json"
            )

        self._base_url = (
            settings.embedding_base_url
            or settings.llm_base_url
            or "https://api.siliconflow.cn/v1"
        ).rstrip("/")
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions

        self._batch_size = getattr(settings, "embedding_batch_size", _DEFAULT_BATCH)
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        logger.debug(
            "EmbeddingClient init: model=%s dim=%d batch=%d base=%s",
            self.model, self.dimensions, self._batch_size, self._base_url,
        )

    # ── 公开接口 ─────────────────────────────────────────────

    def encode(self, text: str) -> list[float]:
        """对单条文本进行编码，返回向量"""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码，自动按 batch_size 分块后合并结果"""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for chunk_start in range(0, len(texts), self._batch_size):
            chunk = texts[chunk_start : chunk_start + self._batch_size]
            chunk_result = self._call_api(chunk)
            all_embeddings.extend(chunk_result)

        return all_embeddings

    # ── 内部实现 ─────────────────────────────────────────────

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """发送单次嵌入请求到推理 API"""
        url = f"{self._base_url}{_EMBEDDINGS_PATH}"

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers=self._headers)

        match resp.status_code:
            case 200:
                data = resp.json()
                return self._extract_vectors(data)
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

    @staticmethod
    def _extract_vectors(data: dict[str, Any]) -> list[list[float]]:
        """从 API 响应中提取 embedding 向量列表"""
        raw = data.get("data", [])
        # 按 index 排序保证批次顺序一致
        sorted_raw = sorted(raw, key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in sorted_raw]

    def __enter__(self) -> EmbeddingClient:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ── 异常 ──────────────────────────────────────────────────


class EmbeddingError(Exception):
    """嵌入调用基异常"""


class APIError(EmbeddingError):
    """API 调用失败"""

    def __init__(self, status_code: int | None = None, detail: str = "") -> None:
        self.status_code = status_code
        self.detail = detail

    def __str__(self) -> str:
        base = f"APIError(status={self.status_code})"
        if self.detail:
            base += f": {self.detail}"
        return base


class RateLimitError(EmbeddingError):
    """触发速率限制"""
