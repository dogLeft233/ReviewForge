"""嵌入服务——SiliconFlow BGE-M3 适配器

支持：
- 多维度输出（通过 dimensions 参数请求降采样）
- 自动维度归一化（预计算时记录实际维度，运行时对齐）
- 批量预计算 retriever domain embeddings
- 重试 + 错误处理

使用方式：
    svc = EmbeddingService(api_key="sk-...")
    svc.build_retriever_index()          # 预计算（仅需一次）
    top_sources = svc.route_query("your research query")
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)


# 各检索器擅长领域的描述——用于预计算 embedding
RETRIVER_DESCRIPTIONS: dict[str, str] = {
    "arxiv": (
        "Academic preprints on computer science, machine learning, AI, deep learning, "
        "natural language processing, computer vision, reinforcement learning, "
        "graph neural networks, foundation models, LLM pretraining and fine-tuning"
    ),
    "semantic_scholar": (
        "Academic papers with citation graphs, citation counts, bibliographic data, "
        "computer science venues including ACL, NeurIPS, ICML, AAAI, KDD, SIGIR, WWW, "
        "research paper recommendations and impact analysis"
    ),
    "dblp": (
        "Computer science bibliography entries, conference papers, journal articles, "
        "author/title/venue metadata from ACM, IEEE, Springer, Elsevier, "
        "academic venue information and publication trends"
    ),
    "github": (
        "Open source software repositories, code implementations, developer tools, "
        "software libraries, system prototypes with stars and activity metrics, "
        "MLOps pipelines, data processing frameworks, benchmark implementations"
    ),
    "papers_with_code": (
        "Machine learning papers with official code implementations, benchmarks, "
        "evaluation results, reproducibility resources, state-of-the-art comparisons"
    ),
    "huggingface": (
        "Pre-trained machine learning models, datasets, model cards, fine-tuned models, "
        "AI applications and demos, model hubs, AI community resources"
    ),
    "hackernews": (
        "Tech industry news, startup announcements, developer discussions, "
        "trending technology topics, startup insights, Hacker News community links, "
        " Silicon Valley trends"
    ),
    "serper": (
        "General web search for papers, technical blog posts, tutorials, "
        "conference announcements, product releases, Google search results, "
        "recent news and trends in AI and software engineering"
    ),
}


@dataclass
class RetrieverEmbedding:
    """预计算的检索器 domain embedding"""
    retriever_name: str
    embedding: np.ndarray  # 归一化为 unit vector（L2=1）
    actual_dim: int


class EmbeddingService:
    """SiliconFlow BGE-M3 嵌入服务

    支持任意维度适配：当 API 返回的向量维度与预计算不一致时，
    通过 _align_embedding 自动对齐（截断长向量 / 填充短向量）。
    """

    DEFAULT_MODEL = "BAAI/bge-m3"
    MAX_DIMENSIONS = 1024

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.siliconflow.cn/v1",
        model: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or ""
        self.base_url = base_url
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries

        self._client = httpx.Client(timeout=timeout)
        self._retriever_embeddings: dict[str, RetrieverEmbedding] = {}
        self._ready = False
        self._actual_dim: int | None = None

    # ── 公开 API ──

    def embed(self, text: str, dimensions: int | None = None) -> np.ndarray:
        """对单条文本生成嵌入向量

        Args:
            text: 输入文本
            dimensions: 可选，请求 API 返回降采样维度
                        （API 可能只支持 256/512/1024，最终以实际返回为准）

        Returns:
            归一化的 numpy 向量（unit vector，L2=1）
        """
        body: dict[str, Any] = {
            "input": text,
            "model": self.model,
            "encoding_format": "float",
        }
        if dimensions is not None:
            body["dimensions"] = dimensions

        resp = self._post(body)
        data = resp.get("data", [])
        if not data:
            raise ValueError("Empty embedding response from API")

        embedding = data[0]["embedding"]
        vec = np.array(embedding, dtype=np.float32)

        # L2 归一化（余弦相似度计算要求）
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def embed_batch(self, texts: list[str], dimensions: int | None = None) -> list[np.ndarray]:
        """批量嵌入（API 支持 list 输入，单次请求更高效）

        Args:
            texts: 文本列表
            dimensions: 可选，请求降采样维度

        Returns:
            归一化向量列表
        """
        if not texts:
            return []

        body: dict[str, Any] = {
            "input": texts,
            "model": self.model,
            "encoding_format": "float",
        }
        if dimensions is not None:
            body["dimensions"] = dimensions

        resp = self._post(body)
        results: list[np.ndarray] = []
        for item in resp.get("data", []):
            vec = np.array(item["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec)
        return results

    @staticmethod
    def compute_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度（两个向量均已归一化，直接点积即余弦）"""
        return float(np.dot(a, b))

    def build_retriever_index(self, dimensions: int | None = None) -> None:
        """预计算所有检索器 domain embedding，建立路由索引

        只需调用一次，结果缓存于 self._retriever_embeddings。

        Args:
            dimensions: 可选，指定嵌入维度（API 可能只支持 256/512/1024）
        """
        names = list(RETRIVER_DESCRIPTIONS.keys())
        descs = [RETRIVER_DESCRIPTIONS[n] for n in names]

        embeddings = self.embed_batch(descs, dimensions=dimensions)

        # 检测实际嵌入维度（API 可能返回不同于请求的维度）
        self._actual_dim = embeddings[0].shape[0]

        self._retriever_embeddings.clear()
        for name, vec in zip(names, embeddings):
            self._retriever_embeddings[name] = RetrieverEmbedding(
                retriever_name=name,
                embedding=vec,
                actual_dim=vec.shape[0],
            )

        self._ready = True
        logger.info(
            "Retriever index built: %d retrievers, actual_dim=%d",
            len(self._retriever_embeddings), self._actual_dim,
        )

    def is_ready(self) -> bool:
        return self._ready

    def actual_dimension(self) -> int | None:
        """返回实际嵌入维度（build 后才有效）"""
        return self._actual_dim

    def get_retriever_embedding(self, name: str) -> RetrieverEmbedding | None:
        return self._retriever_embeddings.get(name)

    def route_query(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.30,
    ) -> list[tuple[str, float]]:
        """基于嵌入相似度路由查询到最相关的 retriever

        Args:
            query: 查询文本
            top_k: 返回前 k 个最相关的 retriever
            threshold: 最低相似度阈值，低于此值的 retriever 被过滤

        Returns:
            [(retriever_name, similarity_score), ...]，按相似度降序
        """
        if not self._ready:
            self.build_retriever_index()

        q_vec = self.embed(query)

        scored: list[tuple[str, float]] = []
        for name, ret_emb in self._retriever_embeddings.items():
            # 对齐到 retriever 向量维度（ret_emb 维度是基准）
            # q_vec 被截断/填充到与 ret_emb 同维，再做点积
            emb_aligned = self._align_embedding(q_vec, ret_emb.embedding)
            score = self.compute_similarity(emb_aligned, ret_emb.embedding)
            if score >= threshold:
                scored.append((name, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ── 私有方法 ──

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """发送请求，带重试和退避"""
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    logger.warning("Embedding API rate limited, attempt %d", attempt)
                    time.sleep(2 * attempt)
                    continue
                else:
                    resp.raise_for_status()
            except Exception as e:
                last_err = e
                logger.warning("Embedding request failed (attempt %d/%d): %s",
                               attempt, self.max_retries, e)

        raise RuntimeError(
            f"Embedding request failed after {self.max_retries} retries: {last_err}"
        )

    @staticmethod
    def _align_embedding(query: np.ndarray, target: np.ndarray) -> np.ndarray:
        """对齐两个维度不同的向量

        用于处理 API 返回维度与预计算维度不一致的情况。

        Args:
            query: 查询向量
            target: 预计算的 retriever 向量

        Returns:
            与 target 同维度的对齐后向量
        """
        qlen, tlen = query.shape[0], target.shape[0]
        if qlen == tlen:
            return query

        if qlen > tlen:
            # query 更长 → 截断 query 到 target 长度
            return query[:tlen]
        else:
            # target 更长 → query 填充 0 到 target 长度
            aligned = np.zeros(tlen, dtype=np.float32)
            aligned[:qlen] = query
            return aligned

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EmbeddingService":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()