"""ArxivAdapter 单元测试"""

import pytest

from src.searcher.adapters.arxiv_adapter import _clean_arxiv_query


class TestCleanArxivQuery:
    """_clean_arxiv_query 将 LLM 生成的查询清理为 ArxivRetriever 可用的格式"""

    def test_all_prefix_stripped(self):
        assert _clean_arxiv_query("all:transformer") == "transformer"

    def test_ti_prefix_stripped(self):
        assert _clean_arxiv_query("ti:transformer") == "transformer"

    def test_au_prefix_stripped(self):
        assert _clean_arxiv_query("au:Vaswani") == "Vaswani"

    def test_abs_prefix_stripped(self):
        assert _clean_arxiv_query("abs:machine learning") == "machine learning"

    def test_cat_prefix_stripped(self):
        assert _clean_arxiv_query("cat:cs.LG") == "cs.LG"

    def test_sortby_suffix_removed(self):
        assert _clean_arxiv_query("transformer&sortBy=relevance") == "transformer"

    def test_compound_query(self):
        result = _clean_arxiv_query(
            "all:transformer AND all:attention cat:cs.LG&sortBy=relevance"
        )
        assert "transformer" in result
        assert "attention" in result

    def test_bare_query_unchanged(self):
        assert _clean_arxiv_query("BERT") == "BERT"

    def test_empty_input(self):
        assert _clean_arxiv_query("") == ""

    def test_mixed_case_preserved(self):
        assert _clean_arxiv_query("all:BERT AND ti:attention") == "BERT AND attention"
