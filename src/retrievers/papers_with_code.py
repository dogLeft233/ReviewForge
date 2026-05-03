"""Papers With Code 检索器（API 已迁移至 HuggingFace，此检索器已废弃）

Notes from 2026:
- Papers With Code was acquired by HuggingFace
- The old API (api/v1) now redirects to HuggingFace papers with HTML
- For paper search, use ArxivRetriever / SemanticScholarRetriever
- For benchmarks/SOTA, this retriever is preserved for legacy compatibility
"""

from src.models import PaperCard, ResourceCard
from src.retrievers.base import BaseRetriever

import logging

logger = logging.getLogger(__name__)


class PapersWithCodeRetriever(BaseRetriever):
    """Papers With Code 检索器（已废弃 - API 已迁移至 HuggingFace）

    Papers With Code was acquired by HuggingFace in 2024.
    The API v1 is deprecated and no longer returns JSON.
    Use ArxivRetriever or SemanticScholarRetriever for paper search instead.
    """

    name = "papers_with_code"

    def search(
        self, query: str, max_results: int | None = None
    ) -> list[PaperCard]:
        logger.warning(
            "PapersWithCode API v1 is deprecated (acquired by HuggingFace). "
            "Use ArxivRetriever or SemanticScholarRetriever instead."
        )
        return []

    def search_resources(
        self, query: str, max_results: int | None = None
    ) -> list[ResourceCard]:
        logger.warning(
            "PapersWithCode API v1 is deprecated (acquired by HuggingFace). "
            "Use HuggingFaceRetriever for models/datasets instead."
        )
        return []

    def get_sota(self, task_id: str) -> list[ResourceCard]:
        logger.warning(
            "PapersWithCode API v1 is deprecated. SOTA data unavailable."
        )
        return []
