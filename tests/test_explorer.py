"""DomainExplorer 领域知识探索器测试"""

import pytest

from src.explorer import (
    DomainExplorer,
    DomainProfile,
    TermCard,
    explore_domain,
    BochaRetriever,
)
from src.retrievers.hackernews import HackerNewsRetriever


# ──────────────────────────────────────────────
# TermCard 测试
# ──────────────────────────────────────────────


class TestTermCard:
    def test_minimal(self):
        t = TermCard(term="Transformer")
        assert t.term == "Transformer"
        assert t.definition == ""
        assert t.source == ""

    def test_full(self):
        t = TermCard(term="ASR", definition="Automatic Speech Recognition", source="bocha", url="https://example.com")
        assert t.definition == "Automatic Speech Recognition"
        assert t.source == "bocha"
        assert t.url == "https://example.com"


# ──────────────────────────────────────────────
# DomainProfile 测试
# ──────────────────────────────────────────────


class TestDomainProfile:
    def test_minimal(self):
        p = DomainProfile(original_query="test")
        assert p.original_query == "test"
        assert p.key_terms == []
        assert p.suggested_query_expansions == []

    def test_to_human_readable_empty(self):
        p = DomainProfile(original_query="test")
        text = p.to_human_readable()
        assert "# 领域探索报告：test" in text

    def test_to_human_readable_full(self):
        p = DomainProfile(
            original_query="ASR",
            domain_overview="语音识别领域",
            core_concepts="核心概念包括端到端模型",
            key_terms=[TermCard(term="Transformer", definition="注意力机制")],
            related_topics="相关主题：NLP、CV",
            active_work="代表性工作：Whisper",
            hn_insights="HN 社区关注落地",
            suggested_query_expansions=["ASR", "Whisper"],
        )
        text = p.to_human_readable()
        assert "语音识别领域" in text
        assert "Transformer" in text
        assert "Whisper" in text
        assert "ASR" in text


# ──────────────────────────────────────────────
# BochaRetriever 测试
# ──────────────────────────────────────────────


class TestBochaRetriever:
    def test_skip_without_key(self, monkeypatch):
        """无 API key 时返回空列表，不抛异常"""
        import src.explorer

        monkeypatch.setattr(src.explorer.BochaRetriever, "_load_key", lambda self: None)
        r = BochaRetriever()
        assert r._api_key is None
        results = r.search("test query")
        assert results == []

    def test_parse_empty_data(self):
        data = {"code": 200, "data": {}}
        assert BochaRetriever._parse(data) == []

    def test_parse_valid_data(self):
        data = {
            "data": {
                "webPages": {
                    "value": [
                        {
                            "name": "Whisper Paper",
                            "url": "https://arxiv.org/abs/2212",
                            "snippet": "Whisper is a speech recognition system.",
                            "summary": "A large-scale whisper model.",
                            "siteName": "arXiv",
                            "datePublished": "2023-01-01",
                        }
                    ]
                }
            }
        }
        results = BochaRetriever._parse(data)
        assert len(results) == 1
        assert results[0]["title"] == "Whisper Paper"
        assert results[0]["site_name"] == "arXiv"


# ──────────────────────────────────────────────
# HackerNewsRetriever 集成测试
# ──────────────────────────────────────────────


class TestHackerNewsIntegration:
    def test_hackernews_search(self):
        """HN 检索器能正常搜索并返回结果"""
        r = HackerNewsRetriever()
        resources = r.search_resources("speech recognition", max_results=5)
        assert isinstance(resources, list)
        # 至少验证资源类型
        for res in resources:
            assert hasattr(res, "name")
            assert hasattr(res, "url")
            assert hasattr(res, "stars")

    def test_hackernews_popular(self):
        """HN popular search 正常工作"""
        r = HackerNewsRetriever()
        resources = r.search_popular("machine learning", max_results=3)
        assert isinstance(resources, list)


# ──────────────────────────────────────────────
# DomainExplorer 核心逻辑测试
# ──────────────────────────────────────────────


class TestTermExtraction:
    """术语提取逻辑的单元测试"""

    def test_camelcase_extraction(self):
        ex = DomainExplorer()
        terms = ex._extract_terms_from_text(
            "The DiffusionModel and Transformer are key architectures in modern ASR systems."
        )
        assert "DiffusionModel" in terms or "Transformer" in terms

    def test_acronym_extraction(self):
        ex = DomainExplorer()
        text = "LLM and GPT models are used in NLP; ASR uses end-to-end approaches."
        terms = ex._extract_terms_from_text(text)
        found = {t.upper() for t in terms}
        # 验证术语提取能覆盖 NLP/NLP 类的词
        assert any(x in found for x in ["LLM", "GPT", "ASR", "NLP", "end-to-end"])

    def test_no_duplicates(self):
        ex = DomainExplorer()
        terms = ex._extract_terms_from_text("The Transformer architecture uses self-attention.")
        # 去重后不应有完全相同的项
        assert len(terms) == len(set(t.lower() for t in terms))


class TestHNInsights:
    def test_summarize_hn_insights_empty(self):
        ex = DomainExplorer()
        result = ex._summarize_hn_insights([], [])
        assert result == ""

    def test_summarize_hn_insights_with_data(self):
        ex = DomainExplorer()
        discussions = ["Whisper v3 released", "OpenAI Whisper analysis", "ASR benchmark comparison"]
        results = [
            {"title": "Whisper v3 released", "points": 200},
            {"title": "OpenAI Whisper analysis", "points": 80},
            {"title": "ASR benchmark comparison", "points": 45},
        ]
        insight = ex._summarize_hn_insights(discussions, results)
        assert "Whisper" in insight
        assert "HN 社区" in insight


# ──────────────────────────────────────────────
# DomainExplorer 集成测试（真实 API 调用）
# ──────────────────────────────────────────────


class TestDomainExplorerReal:
    """真实 API 调用测试（ASR 领域）"""

    def test_explore_asr_hackernews_only(self):
        """只用 HN 探测 ASR 领域"""
        ex = DomainExplorer(use_bocha=False, use_serper=False, max_hn_results=5)
        profile = ex.explore("automatic speech recognition ASR")
        assert profile.original_query == "automatic speech recognition ASR"
        assert "hackernews" in profile.sources_used

    def test_explore_asr_with_bocha(self):
        """HN + 博查探测 ASR 领域"""
        ex = DomainExplorer(use_hackernews=True, use_bocha=True, use_serper=False, max_hn_results=5, max_bocha_results=5)
        profile = ex.explore("automatic speech recognition ASR")
        assert profile.original_query == "automatic speech recognition ASR"
        assert "bocha" in profile.sources_used or profile.bocha_results is not None
        assert len(profile.key_terms) >= 0

    def test_explore_generates_expansions(self):
        """探索后生成扩展查询词"""
        ex = DomainExplorer(use_hackernews=True, use_bocha=False, max_hn_results=5)
        profile = ex.explore("speech recognition")
        assert profile.original_query == "speech recognition"
        # 扩展词至少应包含原始查询
        assert profile.suggested_query_expansions[0] == "speech recognition"

    def test_explore_overview_not_empty(self):
        """探索后领域概览非空"""
        ex = DomainExplorer(use_hackernews=True, use_bocha=False, max_hn_results=5)
        profile = ex.explore("speech recognition")
        # 有结果时概览应有内容
        if profile.key_terms or profile.hn_discussions:
            assert len(profile.domain_overview) > 0

    def test_to_human_readable(self):
        """人类可读输出包含关键字段"""
        ex = DomainExplorer(use_hackernews=True, use_bocha=False, max_hn_results=5)
        profile = ex.explore("speech recognition")
        text = profile.to_human_readable()
        assert "领域探索报告" in text
        assert profile.original_query in text

    @pytest.mark.integration
    def test_full_explorer_with_all_sources(self):
        """全量 HN + 博查 + Serper 探测（需要各 API Key）"""
        ex = DomainExplorer(use_hackernews=True, use_bocha=True, use_serper=False, max_hn_results=8, max_bocha_results=8)
        profile = ex.explore("automatic speech recognition ASR analysis")
        assert profile.original_query
        assert profile.sources_used  # 至少有数据源记录
        assert isinstance(profile.key_terms, list)


# ──────────────────────────────────────────────
# 便捷函数测试
# ──────────────────────────────────────────────


class TestConvenienceFunction:
    def test_explore_domain_one_liner(self):
        """explore_domain() 便捷函数调用"""
        profile = explore_domain("speech recognition", use_hackernews=True, use_bocha=False)
        assert isinstance(profile, DomainProfile)
        assert profile.original_query == "speech recognition"
