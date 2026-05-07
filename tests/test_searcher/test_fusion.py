"""RRF 融合测试"""

import pytest

from src.searcher.result import SearchResult
from src.searcher.rankers.fusion import rrf_fusion


class TestRRFFusion:
    """RRF (Reciprocal Rank Fusion) 测试"""

    def test_single_source(self):
        """单源结果直接返回"""
        results = [
            SearchResult(title="A", url="https://a.com", source="arxiv", category="papers", domain="a.com"),
            SearchResult(title="B", url="https://b.com", source="arxiv", category="papers", domain="b.com"),
        ]
        fused = rrf_fusion({"arxiv": results})
        assert len(fused) == 2
        assert [r.title for r in fused] == ["A", "B"]

    def test_two_sources_same_url(self):
        """同一 URL 在两个来源出现时，分数应叠加，排名最高"""
        results_arxiv = [
            SearchResult(title="Paper X", url="https://arxiv.org/abs/1", source="arxiv", category="papers", domain="arxiv.org"),
            SearchResult(title="Paper Y", url="https://arxiv.org/abs/2", source="arxiv", category="papers", domain="arxiv.org"),
        ]
        results_serper = [
            SearchResult(title="Paper X (serper)", url="https://arxiv.org/abs/1", source="serper", category="papers", domain="arxiv.org"),
            SearchResult(title="Paper Z", url="https://arxiv.org/abs/3", source="serper", category="papers", domain="arxiv.org"),
        ]

        fused = rrf_fusion({"arxiv": results_arxiv, "serper": results_serper})

        # 同一 URL 应该只有一条，且分数最高
        assert len(fused) == 3
        # Paper X 的 URL 在两个来源都出现，分数应该最高（1/(60+1) + 1/(60+1)）
        paper_x = next(r for r in fused if r.title == "Paper X")
        assert paper_x.rank_score > 0.03  # 两个来源叠加

    def test_multiple_sources_order(self):
        """多源融合后排序应按 RRF 分数降序"""
        results1 = [
            SearchResult(title="A", url="https://a.com", source="src1", category="papers", domain="a.com"),
            SearchResult(title="B", url="https://b.com", source="src1", category="papers", domain="b.com"),
        ]
        results2 = [
            SearchResult(title="B", url="https://b.com", source="src2", category="papers", domain="b.com"),
            SearchResult(title="C", url="https://c.com", source="src2", category="papers", domain="c.com"),
        ]

        fused = rrf_fusion({"src1": results1, "src2": results2})

        # B 同时出现在两个来源，分数最高
        titles = [r.title for r in fused]
        assert titles.index("B") < titles.index("A")
        assert titles.index("B") < titles.index("C")

    def test_empty_source(self):
        """空来源不报错"""
        fused = rrf_fusion({"arxiv": []})
        assert fused == []

    def test_k_parameter(self):
        """k 参数越大，对排名靠后的结果越平滑"""
        results1 = [
            SearchResult(title=f"R{i}", url=f"https://r{i}.com", source="s1", category="papers", domain=f"r{i}.com")
            for i in range(5)
        ]
        results2 = [
            SearchResult(title=f"R{i}", url=f"https://r{i}.com", source="s2", category="papers", domain=f"r{i}.com")
            for i in range(5)
        ]

        fused_k1 = rrf_fusion({"s1": results1, "s2": results2}, k=1)
        fused_k100 = rrf_fusion({"s1": results1, "s2": results2}, k=100)

        # k=1 时排名靠前的结果权重更大
        r0_k1 = next(r for r in fused_k1 if r.title == "R0")
        r1_k1 = next(r for r in fused_k1 if r.title == "R1")
        assert r0_k1.rank_score > r1_k1.rank_score

        # k=100 时权重更平滑
        r0_k100 = next(r for r in fused_k100 if r.title == "R0")
        r1_k100 = next(r for r in fused_k100 if r.title == "R1")
        assert r0_k100.rank_score > r1_k100.rank_score

    def test_rank_score_preserved(self):
        """融合后 rank_score 应正确设置"""
        results = [
            SearchResult(title="A", url="https://a.com", source="src1", category="papers", domain="a.com"),
        ]
        fused = rrf_fusion({"src1": results})
        assert fused[0].rank_score == pytest.approx(1 / 61, rel=1e-3)