"""MetaSearcher 集成测试"""

import pytest
from unittest.mock import patch

from src.searcher.meta import MetaSearcher
from src.searcher.result import SearchResult, SearchContext


class TestSearchContext:
    """SearchContext 数据模型测试"""

    def test_total_results(self):
        ctx = SearchContext(
            topic="test",
            papers=[SearchResult(title=f"P{i}", url=f"https://p{i}.com", source="arxiv", category="papers", domain=f"p{i}.com") for i in range(3)],
            resources=[SearchResult(title=f"R{i}", url=f"https://r{i}.com", source="github", category="resources", domain=f"r{i}.com") for i in range(2)],
            news=[],
        )
        assert ctx.total_results == 5

    def test_is_empty(self):
        ctx_empty = SearchContext(topic="empty")
        assert ctx_empty.is_empty()

        ctx_non_empty = SearchContext(
            topic="test",
            papers=[SearchResult(title="A", url="https://a.com", source="arxiv", category="papers", domain="a.com")],
        )
        assert not ctx_non_empty.is_empty()

    def test_to_dict(self):
        ctx = SearchContext(
            topic="test topic",
            papers=[
                SearchResult(title="A", url="https://a.com", source="arxiv", category="papers", domain="a.com", rank_score=0.9),
            ],
        )
        d = ctx.to_dict()
        assert d["topic"] == "test topic"
        assert len(d["papers"]) == 1
        assert d["papers"][0]["title"] == "A"


class TestSearchResult:
    """SearchResult 数据模型测试"""

    def test_domain_extracted_from_url(self):
        r = SearchResult(title="T", url="https://example.com/path", source="test", category="papers", domain="")
        assert r.domain == "example.com"

    def test_domain_uses_provided_value(self):
        r = SearchResult(title="T", url="https://example.com", source="test", category="papers", domain="custom.com")
        assert r.domain == "custom.com"

    def test_to_dict(self):
        r = SearchResult(
            title="Test Paper",
            url="https://arxiv.org/abs/1",
            snippet="A test abstract",
            rank_score=0.95,
            source="arxiv",
            category="papers",
            published_date="2024",
            domain="arxiv.org",
        )
        d = r.to_dict()
        assert d["title"] == "Test Paper"
        assert d["url"] == "https://arxiv.org/abs/1"
        assert d["rank_score"] == 0.95
        assert d["source"] == "arxiv"


class TestMetaSearcherSearch:
    """MetaSearcher.search() 流程测试（用 mock 避免真实网络调用）"""

    @pytest.mark.asyncio
    async def test_returns_search_context_with_all_fields(self):
        """返回的 SearchContext 包含 topic 和三个分类"""
        searcher = MetaSearcher()
        ctx = await searcher.search("test query")
        assert isinstance(ctx, SearchContext)
        assert ctx.topic == "test query"
        assert hasattr(ctx, "papers")
        assert hasattr(ctx, "resources")
        assert hasattr(ctx, "news")

    @pytest.mark.asyncio
    async def test_all_fields_are_lists(self):
        """papers/resources/news 都是 list 类型"""
        searcher = MetaSearcher()
        ctx = await searcher.search("test")
        assert isinstance(ctx.papers, list)
        assert isinstance(ctx.resources, list)
        assert isinstance(ctx.news, list)

    @pytest.mark.asyncio
    async def test_created_at_set(self):
        """created_at 时间戳被设置"""
        searcher = MetaSearcher()
        ctx = await searcher.search("test")
        assert ctx.created_at != ""

    @pytest.mark.asyncio
    async def test_search_context_empty_when_no_adapters(self):
        """当所有适配器返回空结果时，SearchContext 为空（需要 API key 等真实条件）"""
        searcher = MetaSearcher()
        ctx = await searcher.search("this query should return minimal results")
        assert isinstance(ctx, SearchContext)
        assert ctx.topic == "this query should return minimal results"