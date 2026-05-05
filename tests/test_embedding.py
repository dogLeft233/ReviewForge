"""EmbeddingService 单元与集成测试"""

import json
import math
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.embedding import (
    EmbeddingService,
    RETRIVER_DESCRIPTIONS,
    RetrieverEmbedding,
)


# ── Mock 响应工厂 ──

def make_embedding_response(texts: list[str], dim: int = 1024) -> dict:
    """生成模拟的 SiliconFlow 嵌入响应"""
    embedding = [0.1] * dim
    # 让向量有变化，便于归一化测试
    embedding[0] = 0.8
    items = []
    for i, _ in enumerate(texts):
        items.append({
            "object": "embedding",
            "index": i,
            "embedding": embedding,
        })
    return {
        "object": "list",
        "data": items,
        "model": "BAAI/bge-m3",
        "usage": {"prompt_tokens": 10, "total_tokens": 50},
    }


# ── 辅助函数 ──

class FakeEmbeddingService(EmbeddingService):
    """Fake 嵌入服务——返回预设向量，用于测试（不调用真实 API）"""

    def __init__(
        self,
        retriever_vectors: dict[str, np.ndarray] | None = None,
        actual_dim: int = 1024,
    ) -> None:
        super().__init__(api_key="fake-key")
        self._fake_vectors = retriever_vectors or {}
        self._fake_dim = actual_dim
        self._built = False

    def embed(self, text: str, dimensions: int | None = None) -> np.ndarray:
        if "return_dim" in self._fake_vectors:
            dim = self._fake_vectors["return_dim"]
            vec = np.random.rand(dim)
        else:
            vec = np.random.rand(self._fake_dim)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: list[str], dimensions: int | None = None) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]

    def build_retriever_index(self, dimensions: int | None = None) -> None:
        if self._fake_vectors:
            for name, vec in self._fake_vectors.items():
                if name == "return_dim":
                    continue
                self._retriever_embeddings[name] = RetrieverEmbedding(
                    retriever_name=name,
                    embedding=vec if isinstance(vec, np.ndarray) else np.array(vec),
                    actual_dim=vec.shape[0] if isinstance(vec, np.ndarray) else self._fake_dim,
                )
        else:
            # 默认：生成标准随机向量
            for name in RETRIVER_DESCRIPTIONS:
                vec = np.random.rand(self._fake_dim)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self._retriever_embeddings[name] = RetrieverEmbedding(
                    retriever_name=name,
                    embedding=vec,
                    actual_dim=self._fake_dim,
                )
        self._actual_dim = self._fake_dim
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready


# ── 测试类 ──

class TestEmbeddingServiceUnit:
    """EmbeddingService 单元测试（不依赖真实 API）"""

    def test_align_truncates_long_query(self):
        long = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        short = np.array([0.1, 0.2])
        result = EmbeddingService._align_embedding(long, short)
        assert result.shape[0] == 2
        np.testing.assert_array_equal(result, short)

    def test_align_pads_short_query(self):
        short = np.array([0.1, 0.2])
        long = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = EmbeddingService._align_embedding(short, long)
        assert result.shape[0] == 5
        assert result[:2] == pytest.approx(short)
        assert result[2:] == pytest.approx(np.zeros(3))

    def test_align_equal_dims_returns_unchanged(self):
        vec = np.array([0.1, 0.2, 0.3])
        result = EmbeddingService._align_embedding(vec, vec)
        np.testing.assert_array_equal(result, vec)

    def test_unit_vector_normalization(self):
        """验证 embed_batch 返回 L2=1 的单位向量"""
        svc = FakeEmbeddingService()
        vec = svc.embed("test text")
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_similarity_same_vector_is_one(self):
        vec = np.array([0.6, 0.8])
        vec = vec / np.linalg.norm(vec)
        sim = EmbeddingService.compute_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-5

    def test_similarity_orthogonal_vectors_is_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = EmbeddingService.compute_similarity(a, b)
        assert abs(sim) < 1e-5

    def test_similarity_anti_parallel_is_minus_one(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        sim = EmbeddingService.compute_similarity(a, b)
        assert abs(sim + 1.0) < 1e-5

    def test_retriever_descriptions_covers_all_sources(self):
        expected_sources = {
            "arxiv", "semantic_scholar", "dblp", "github",
            "papers_with_code", "huggingface", "hackernews", "serper",
        }
        assert set(RETRIVER_DESCRIPTIONS.keys()) == expected_sources

    def test_fake_service_build_index(self):
        svc = FakeEmbeddingService()
        svc.build_retriever_index()
        assert svc.is_ready()
        assert svc.actual_dimension() == 1024
        assert len(svc._retriever_embeddings) == 8

    def test_fake_service_different_dimension(self):
        svc = FakeEmbeddingService(actual_dim=512)
        svc.build_retriever_index()
        assert svc.actual_dimension() == 512
        for name, emb in svc._retriever_embeddings.items():
            assert emb.actual_dim == 512

    def test_fake_service_route_query(self):
        svc = FakeEmbeddingService()
        svc.build_retriever_index()
        results = svc.route_query("machine learning", top_k=3, threshold=0.1)
        assert len(results) <= 3
        assert all(0.1 <= score <= 1.0 for _, score in results)
        # 应该返回有序
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestEmbeddingServiceAPI:
    """EmbeddingService 真实 API 测试（需要 api_key，标记 skip）"""

    @pytest.mark.skipif(
        not __import__("os").environ.get("SILICONFLOW_API_KEY"),
        reason="需要 SILICONFLOW_API_KEY 环境变量",
    )
    def test_real_embed_returns_1024_dim_vector(self):
        api_key = __import__("os").environ.get("SILICONFLOW_API_KEY")
        svc = EmbeddingService(api_key=api_key)
        vec = svc.embed("deep learning optimization")
        assert vec.shape[0] == 1024
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

    @pytest.mark.skipif(
        not __import__("os").environ.get("SILICONFLOW_API_KEY"),
        reason="需要 SILICONFLOW_API_KEY 环境变量",
    )
    def test_real_build_retriever_index(self):
        api_key = __import__("os").environ.get("SILICONFLOW_API_KEY")
        svc = EmbeddingService(api_key=api_key)
        svc.build_retriever_index()
        assert svc.is_ready()
        assert svc.actual_dimension() == 1024
        assert len(svc._retriever_embeddings) == 8

    @pytest.mark.skipif(
        not __import__("os").environ.get("SILICONFLOW_API_KEY"),
        reason="需要 SILICONFLOW_API_KEY 环境变量",
    )
    def test_real_route_query_returns_sources(self):
        api_key = __import__("os").environ.get("SILICONFLOW_API_KEY")
        svc = EmbeddingService(api_key=api_key)
        svc.build_retriever_index()
        results = svc.route_query("transformer attention mechanism", top_k=4)
        assert len(results) > 0
        names = [n for n, _ in results]
        # arxiv 应该在前几名（学术论文相关）
        assert "arxiv" in names

    @pytest.mark.skipif(
        not __import__("os").environ.get("SILICONFLOW_API_KEY"),
        reason="需要 SILICONFLOW_API_KEY 环境变量",
    )
    def test_dimensions_parameter_respected(self):
        api_key = __import__("os").environ.get("SILICONFLOW_API_KEY")
        svc = EmbeddingService(api_key=api_key)
        # 请求 256 维（API 可能不支持，实际以返回为准）
        vec = svc.embed("test", dimensions=256)
        # BGE-M3 不支持任意维度，实际返回 1024
        # 这里只测：调用不报错，且返回单位向量
        assert len(vec.shape) == 1
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

    @pytest.mark.skipif(
        not __import__("os").environ.get("SILICONFLOW_API_KEY"),
        reason="需要 SILICONFLOW_API_KEY 环境变量",
    )
    def test_batch_embed(self):
        api_key = __import__("os").environ.get("SILICONFLOW_API_KEY")
        svc = EmbeddingService(api_key=api_key)
        vecs = svc.embed_batch(["query one", "query two", "query three"])
        assert len(vecs) == 3
        for v in vecs:
            assert v.shape[0] == 1024
            assert abs(np.linalg.norm(v) - 1.0) < 1e-4


class TestDimensionAdaptation:
    """维度适配测试——模拟 API 返回维度与预计算不一致"""

    def test_mixed_dimensions_in_index(self):
        """同一 index 中不同 retriever 有不同维度"""
        vectors = {
            "arxiv": np.random.rand(1024),
            "github": np.random.rand(512),
            "hackernews": np.random.rand(256),
        }
        # 归一化
        for name in vectors:
            vec = vectors[name]
            vectors[name] = vec / np.linalg.norm(vec)

        svc = FakeEmbeddingService(retriever_vectors=vectors)
        svc.build_retriever_index()

        arxiv_emb = svc.get_retriever_embedding("arxiv")
        assert arxiv_emb.actual_dim == 1024

        # 路由时维度对齐
        results = svc.route_query("test query", top_k=3)
        assert len(results) > 0

    def test_query_and_index_different_dimensions(self):
        """查询向量维度与索引维度不一致时的对齐"""
        svc = FakeEmbeddingService(actual_dim=512)
        svc.build_retriever_index()

        # 模拟 1024 维查询向量（来自某次不同的 embed 调用）
        query_1024 = np.random.rand(1024)
        query_1024 = query_1024 / np.linalg.norm(query_1024)

        arxiv_emb = svc.get_retriever_embedding("arxiv")
        # query 1024 → 对齐到 512
        aligned = EmbeddingService._align_embedding(query_1024, arxiv_emb.embedding)
        assert aligned.shape[0] == 512

    def test_cross_model_dimension_flexibility(self):
        """跨模型切换时的维度弹性（无需重写代码）"""
        # 模拟切换到 256 维模型
        svc_256 = FakeEmbeddingService(actual_dim=256)
        svc_256.build_retriever_index()
        assert svc_256.actual_dimension() == 256

        # 模拟切换到 1024 维模型
        svc_1024 = FakeEmbeddingService(actual_dim=1024)
        svc_1024.build_retriever_index()
        assert svc_1024.actual_dimension() == 1024

        # 两者路由行为一致（不对代码做修改）
        results_256 = svc_256.route_query("machine learning", top_k=3)
        results_1024 = svc_1024.route_query("machine learning", top_k=3)
        assert len(results_256) > 0
        assert len(results_1024) > 0