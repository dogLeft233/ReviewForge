"""Unit tests for retrievers"""


def test_imports():
    """所有模块都能导入"""
    from src.models import PaperCard, ResourceCard, CurationReport
    from src.exceptions import (
        ReviewForgeError,
        ValidationError,
        ConfigurationError,
        RetrieverError,
        RateLimitError,
    )
    from src.config import Settings
    from src.retrievers.base import BaseRetriever

    assert PaperCard
    assert ResourceCard
    assert CurationReport
    assert ReviewForgeError
    assert BaseRetriever
