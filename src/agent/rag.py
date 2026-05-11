"""RAG 接口预留——后续接入向量数据库"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RAGRetriever:
    """RAG 检索器接口（预留）。

    后续接入向量数据库（如 Milvus、Chroma）时实现此接口。
    """

    def __init__(self, **kwargs: Any) -> None:
        self._initialized = False
        logger.warning("RAGRetriever not yet initialized (vector DB not connected)")

    def add_documents(self, texts: list[str], metadata: list[dict[str, Any]] | None = None) -> None:
        """添加文档到检索库（预留）"""
        raise NotImplementedError("RAG not connected yet")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """根据 query 检索相关文档（预留）"""
        raise NotImplementedError("RAG not connected yet")

    def retrieve_with_score(
        self, query: str, top_k: int = 5, score_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """带相关性分数的检索（预留）"""
        raise NotImplementedError("RAG not connected yet")

    def clear(self) -> None:
        """清除所有文档（预留）"""
        raise NotImplementedError("RAG not connected yet")


def get_rag_retriever(**kwargs: Any) -> RAGRetriever:
    """获取 RAG 检索器实例（预留）"""
    return RAGRetriever(**kwargs)


def build_rag_context(query: str, top_k: int = 5) -> str:
    """根据 query 构建 RAG 上下文字符串（预留）"""
    logger.debug("RAG not connected, returning empty context for query: %s", query)
    return ""
