"""去重测试"""

import pytest

from src.searcher.result import SearchResult
from src.searcher.rankers.dedup import normalize_url, deduplicate


class TestNormalizeUrl:
    """URL 归一化测试"""

    def test_basic(self):
        assert normalize_url("https://example.com/") == "example.com"
        assert normalize_url("http://example.com") == "example.com"

    def test_www_removal(self):
        assert normalize_url("https://www.example.com/") == "example.com"
        assert normalize_url("http://www.example.com") == "example.com"

    def test_tracking_params_removed(self):
        assert normalize_url("https://example.com/?utm_source=google") == "example.com"
        assert normalize_url("https://example.com/?fbclid=abc&ref=source") == "example.com"

    def test_mixed_tracking_params(self):
        url = "https://example.com/path?gclid=xyz&utm_source=email&utm_medium=link"
        assert "gclid" not in normalize_url(url)
        assert "utm_source" not in normalize_url(url)

    def test_preserve_other_params(self):
        url = "https://example.com/page?id=123&sort=asc"
        normalized = normalize_url(url)
        assert "id=123" in normalized
        assert "sort=asc" in normalized

    def test_case_normalization(self):
        assert normalize_url("HTTPS://EXAMPLE.COM/PATH") == "example.com/path"

    def test_fragment_removed(self):
        assert normalize_url("https://example.com/page#section") == "example.com/page"

    def test_empty_url(self):
        assert normalize_url("") == ""
        assert normalize_url("   ") == ""

    def test_invalid_url(self):
        assert normalize_url("not a valid url") == "not a valid url"


class TestDedup:
    """去重测试"""

    def test_exact_url_dedup(self):
        """完全相同的 URL 只保留第一条"""
        results = [
            SearchResult(title="A", url="https://a.com", source="src1", category="papers", domain="a.com"),
            SearchResult(title="A", url="https://a.com", source="src2", category="papers", domain="a.com"),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0].title == "A"

    def test_normalized_url_dedup(self):
        """归一化后相同的 URL（utm_source 去除后和 trailing slash 版本相同）只保留一条"""
        results = [
            SearchResult(title="A", url="https://a.com/?utm_source=x", source="src1", category="papers", domain="a.com"),
            SearchResult(title="A (same)", url="https://a.com/", source="src2", category="papers", domain="a.com"),
        ]
        deduped = deduplicate(results)
        # normalize: both -> "a.com"
        assert len(deduped) == 1

    def test_different_urls_kept(self):
        results = [
            SearchResult(title="A", url="https://a.com", source="src1", category="papers", domain="a.com"),
            SearchResult(title="B", url="https://b.com", source="src1", category="papers", domain="b.com"),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 2

    def test_title_similarity_not_used_for_exact_url(self):
        """精确相同 URL 时，标题相似度不影响去重逻辑"""
        results = [
            SearchResult(title="Attention Is All You Need", url="https://arxiv.org/abs/1", source="arxiv", category="papers", domain="arxiv.org"),
            SearchResult(title="Attention Is All You Need", url="https://arxiv.org/abs/1", source="serper", category="papers", domain="arxiv.org"),
        ]
        deduped = deduplicate(results, title_threshold=0.85)
        assert len(deduped) == 1

    def test_different_pages_same_domain(self):
        """不同页面（不同 URL）各自保留"""
        results = [
            SearchResult(title="Page A", url="https://arxiv.org/abs/1", source="arxiv", category="papers", domain="arxiv.org"),
            SearchResult(title="Page B", url="https://arxiv.org/abs/2", source="arxiv", category="papers", domain="arxiv.org"),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 2

    def test_empty_list(self):
        assert deduplicate([]) == []

    def test_single_item(self):
        results = [
            SearchResult(title="Solo", url="https://solo.com", source="src1", category="papers", domain="solo.com"),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 1