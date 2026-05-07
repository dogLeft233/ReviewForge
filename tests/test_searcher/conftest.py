"""tests/test_searcher/ fixtures"""

import pytest

from src.searcher.result import SearchResult


@pytest.fixture
def sample_search_result() -> SearchResult:
    return SearchResult(
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        snippet="We propose a new simple network architecture, the Transformer",
        rank_score=0.9,
        source="arxiv",
        category="papers",
        published_date="2017",
        domain="arxiv.org",
    )


@pytest.fixture
def sample_papers_results() -> list[SearchResult]:
    return [
        SearchResult(title="Paper A", url="https://arxiv.org/abs/1", snippet="", source="arxiv", category="papers", domain="arxiv.org"),
        SearchResult(title="Paper B", url="https://arxiv.org/abs/2", snippet="", source="serper", category="papers", domain="arxiv.org"),
        SearchResult(title="Paper C", url="https://arxiv.org/abs/3", snippet="", source="arxiv", category="papers", domain="arxiv.org"),
    ]


@pytest.fixture
def sample_resources_results() -> list[SearchResult]:
    return [
        SearchResult(title="Repo X", url="https://github.com/x/repo", snippet="", source="github", category="resources", domain="github.com"),
        SearchResult(title="Model Y", url="https://huggingface.co/y/model", snippet="", source="huggingface", category="resources", domain="huggingface.co"),
    ]


@pytest.fixture
def sample_news_results() -> list[SearchResult]:
    return [
        SearchResult(title="News 1", url="https://bocha.com/news/1", snippet="", source="bocha", category="news", domain="bocha.com"),
        SearchResult(title="HN Post", url="https://news.ycombinator.com/item?id=1", snippet="", source="hackernews", category="news", domain="news.ycombinator.com"),
    ]