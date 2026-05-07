"""SemanticScholarAdapter 测试"""

import pytest
from unittest.mock import patch

from src.searcher.result import SearchResult


class TestSemanticScholarAdapter:
    """SemanticScholarAdapter 搜索流程测试"""

    @pytest.mark.asyncio
    async def test_returns_search_results_with_citation_count(self):
        """验证 adapter 返回 SearchResult 并携带 citation_count"""
        from src.searcher.adapters.semantic_scholar_adapter import SemanticScholarAdapter
        from src.models import PaperCard

        paper = PaperCard(
            title="Attention Is All You Need",
            authors=["Vaswani"],
            year=2017,
            venue="NeurIPS",
            abstract="The dominant sequence transduction models are based on complex...",
            citation_count=95000,
            url="https://api.semanticscholar.org/CorpusID:12345",
            source="semantic_scholar",
        )

        with patch(
            "src.searcher.adapters.semantic_scholar_adapter.SemanticScholarRetriever"
        ) as MockRetrieverClass:
            mock_instance = MockRetrieverClass.return_value
            mock_instance.search.return_value = [paper]

            adapter = SemanticScholarAdapter()
            # adapter._retriever 现在是 MockRetrieverClass() 的实例
            results = await adapter.search("transformer attention", max_results=10)

        assert len(results) == 1
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.title == "Attention Is All You Need"
        assert r.citation_count == 95000
        assert r.source == "semantic_scholar"
        assert r.category == "papers"
        assert r.domain == "semanticscholar.org"

    @pytest.mark.asyncio
    async def test_empty_on_search_error(self):
        """验证检索失败时返回空列表"""
        from src.searcher.adapters.semantic_scholar_adapter import SemanticScholarAdapter

        with patch(
            "src.searcher.adapters.semantic_scholar_adapter.SemanticScholarRetriever"
        ) as MockRetrieverClass:
            mock_instance = MockRetrieverClass.return_value
            mock_instance.search.side_effect = RuntimeError("API timeout")

            adapter = SemanticScholarAdapter()
            adapter._retriever = mock_instance

            results = await adapter.search("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_no_citation_count_falls_to_zero(self):
        """验证无 citation_count 时默认为 0"""
        from src.searcher.adapters.semantic_scholar_adapter import SemanticScholarAdapter
        from src.models import PaperCard

        paper = PaperCard(
            title="Test Paper",
            url="https://semanticscholar.org/paper/1",
        )

        with patch(
            "src.searcher.adapters.semantic_scholar_adapter.SemanticScholarRetriever"
        ) as MockRetrieverClass:
            mock_instance = MockRetrieverClass.return_value
            mock_instance.search.return_value = [paper]

            adapter = SemanticScholarAdapter()
            results = await adapter.search("test")
        assert len(results) == 1
        assert results[0].citation_count == 0
