"""检索器统一入口

所有外部数据源通过此包访问。用法:

    from src.retrievers import (
        ArxivRetriever,
        GithubRetriever,
        HuggingFaceRetriever,
    )

    with ArxivRetriever() as arxiv:
        papers = arxiv.search("LoRA fine-tuning")
"""

from src.retrievers.arxiv import ArxivRetriever
from src.retrievers.ar5iv import Ar5ivRetriever
from src.retrievers.github import GithubRetriever
from src.retrievers.huggingface import HuggingFaceRetriever
from src.retrievers.semantic_scholar import SemanticScholarRetriever

__all__ = [
    "ArxivRetriever",
    "Ar5ivRetriever",
    "GithubRetriever",
    "HuggingFaceRetriever",
    "SemanticScholarRetriever",
]
