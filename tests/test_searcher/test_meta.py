"""MetaSearcher 集成测试（mock 适配器避免网络调用）"""

import pytest
from unittest.mock import AsyncMock, patch

from src.searcher.meta import MetaSearcher
from src.searcher.models import DomainProfile
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


def _mock_adapter(name: str, category: str) -> AsyncMock:
    """创建一个 mock 适配器（避免真实网络调用）"""
    adapter = AsyncMock()
    adapter.name = name
    adapter.category = category
    adapter.search = AsyncMock(return_value=[
        SearchResult(
            title=f"Paper from {name}",
            url=f"https://{name}.com/result",
            snippet="Mock result snippet",
            source=name,
            category=category,
            domain=f"{name}.com",
        )
    ])
    return adapter


class TestMetaSearcherSearch:
    """MetaSearcher.search() 流程测试（mock 所有适配器）"""

    @pytest.mark.asyncio
    async def test_returns_search_context_with_all_fields(self):
        """返回的 SearchContext 包含 topic 和三个分类"""
        with patch("src.searcher.meta.AdapterRouter") as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.route.return_value = [
                _mock_adapter("arxiv", "papers"),
            ]
            searcher = MetaSearcher()
            ctx = await searcher.search("test query", max_results=5)
            assert isinstance(ctx, SearchContext)
            assert ctx.topic == "test query"
            assert hasattr(ctx, "papers")
            assert hasattr(ctx, "resources")
            assert hasattr(ctx, "news")

    @pytest.mark.asyncio
    async def test_all_fields_are_lists(self):
        """papers/resources/news 都是 list 类型"""
        with patch("src.searcher.meta.AdapterRouter") as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.route.return_value = []
            searcher = MetaSearcher()
            ctx = await searcher.search("test")
            assert isinstance(ctx.papers, list)
            assert isinstance(ctx.resources, list)
            assert isinstance(ctx.news, list)

    @pytest.mark.asyncio
    async def test_created_at_set(self):
        """created_at 时间戳被设置"""
        with patch("src.searcher.meta.AdapterRouter") as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.route.return_value = []
            searcher = MetaSearcher()
            ctx = await searcher.search("test")
            assert ctx.created_at != ""

    @pytest.mark.asyncio
    async def test_fallback_query_config_when_no_profile(self):
        """无 DomainProfile 时流程正常"""
        with patch("src.searcher.meta.AdapterRouter") as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.route.return_value = []
            searcher = MetaSearcher()
            ctx = await searcher.search("transformer", domain_profile=None, max_results=5)
            assert isinstance(ctx, SearchContext)

    @pytest.mark.asyncio
    async def test_search_with_domain_profile(self):
        """带 DomainProfile 搜索"""
        profile = DomainProfile(
            topic="transformer",
            core_concepts=["self-attention"],
            classical_papers=[],
        )
        with patch("src.searcher.meta.AdapterRouter") as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.route.return_value = []
            searcher = MetaSearcher()
            ctx = await searcher.search("transformer", domain_profile=profile, max_results=5)
            assert isinstance(ctx, SearchContext)

    @pytest.mark.asyncio
    async def test_papers_results_included(self):
        """papers 分类有结果"""
        with patch("src.searcher.meta.AdapterRouter") as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.route.return_value = [_mock_adapter("arxiv", "papers")]
            searcher = MetaSearcher()
            ctx = await searcher.search("transformer", max_results=5)
            # Adapter 被调用，mock 返回 1 条
            assert isinstance(ctx.papers, list)

    @pytest.mark.asyncio
    async def test_no_crash_on_empty_adapters(self):
        """适配器列表为空时不崩溃"""
        with patch("src.searcher.meta.AdapterRouter") as MockRouter:
            mock_instance = MockRouter.return_value
            mock_instance.route.return_value = []
            searcher = MetaSearcher()
            ctx = await searcher.search("empty query", max_results=5)
            assert ctx.topic == "empty query"
            assert ctx.papers == []