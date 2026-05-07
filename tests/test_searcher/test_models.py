"""模型测试 — DomainProfile / QueryConfig / Boost"""

import pytest

from src.searcher.models import (
    BoostConfig,
    ClassicalPaper,
    DomainProfile,
    QueryConfig,
    QueryVariant,
    fallback_query_config,
)
from src.searcher.result import SearchResult
from src.searcher.rankers.boost import apply_boost


class TestClassicalPaper:
    def test_basic(self):
        p = ClassicalPaper(title="Attention Is All You Need", arxivid="1706.03762", boost_factor=1.5)
        assert p.arxivid == "1706.03762"
        assert p.boost_factor == 1.5


class TestDomainProfile:
    def test_empty(self):
        p = DomainProfile.empty("transformer")
        assert p.topic == "transformer"
        assert p.core_concepts == []
        assert p.classical_papers == []

    def test_to_text(self):
        p = DomainProfile(
            topic="transformer",
            core_concepts=["self-attention", "encoder-decoder"],
            classical_papers=[
                ClassicalPaper(title="Attention Is All You Need", arxivid="1706.03762")
            ],
        )
        text = p.to_text()
        assert "transformer" in text
        assert "self-attention" in text
        assert "1706.03762" in text


class TestQueryConfig:
    def test_get_query_found(self):
        cfg = QueryConfig(
            topic="transformer",
            queries={
                "arxiv": [
                    QueryVariant(query="all:transformer", variant_type="primary", source="arxiv"),
                    QueryVariant(query="ti:transformer", variant_type="title", source="arxiv"),
                ]
            },
        )
        assert cfg.get_query("arxiv", "primary") == "all:transformer"
        assert cfg.get_query("arxiv", "title") == "ti:transformer"

    def test_get_query_not_found(self):
        cfg = QueryConfig(topic="transformer", queries={})
        assert cfg.get_query("arxiv") == ""

    def test_all_queries_for_source(self):
        cfg = QueryConfig(
            topic="transformer",
            queries={
                "arxiv": [
                    QueryVariant(query="all:transformer", variant_type="primary", source="arxiv"),
                    QueryVariant(query="ti:transformer", variant_type="title", source="arxiv"),
                ]
            },
        )
        variants = cfg.all_queries_for_source("arxiv")
        assert len(variants) == 2

    def test_to_fallback(self):
        cfg = QueryConfig(topic="transformer").to_fallback()
        assert cfg.topic == "transformer"
        assert "arxiv" in cfg.queries
        assert cfg.get_query("arxiv") == "transformer"
        assert cfg.rerank_query == "transformer"


def test_fallback_query_config():
    cfg = fallback_query_config("BERT")
    assert cfg.topic == "BERT"
    assert cfg.get_query("arxiv") == "BERT"
    assert cfg.get_query("github") == "BERT"
    assert len(cfg.queries) >= 6  # all sources


class TestApplyBoost:
    def test_classical_paper_boost(self):
        papers = [
            SearchResult(title="Recent Paper X", url="https://arxiv.org/abs/1234.5678",
                         rank_score=0.9, source="arxiv", category="papers", domain="arxiv.org"),
            SearchResult(title="Attention Is All You Need", url="https://arxiv.org/abs/1706.03762",
                         rank_score=0.7, source="arxiv", category="papers", domain="arxiv.org"),
        ]
        boost_cfg = BoostConfig(
            classical_papers=[
                ClassicalPaper(
                    title="Attention Is All You Need",
                    arxivid="1706.03762",
                    url="https://arxiv.org/abs/1706.03762",
                    boost_factor=1.5,
                )
            ]
        )
        boosted = apply_boost(papers, boost_cfg)
        attention = next(p for p in boosted if "1706.03762" in p.url)
        assert attention.rank_score == 1.05  # 0.7 * 1.5

    def test_no_boost_match(self):
        papers = [
            SearchResult(title="Random Paper", url="https://arxiv.org/abs/9999.9999",
                         rank_score=0.8, source="arxiv", category="papers", domain="arxiv.org"),
        ]
        boost_cfg = BoostConfig(
            classical_papers=[
                ClassicalPaper(
                    title="Attention Is All You Need",
                    arxivid="1706.03762",
                    url="https://arxiv.org/abs/1706.03762",
                    boost_factor=1.5,
                )
            ]
        )
        boosted = apply_boost(papers, boost_cfg)
        assert boosted[0].rank_score == 0.8  # unchanged

    def test_empty_results(self):
        boosted = apply_boost([], BoostConfig())
        assert boosted == []

    def test_override_boost(self):
        papers = [
            SearchResult(title="Paper X", url="https://arxiv.org/abs/1111.1111",
                         rank_score=0.5, source="arxiv", category="papers", domain="arxiv.org"),
        ]
        override = {"https://arxiv.org/abs/1111.1111": 2.0}
        boosted = apply_boost(papers, BoostConfig(), override=override)
        assert boosted[0].rank_score == 1.0  # 0.5 * 2.0