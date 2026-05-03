"""All-in-One 检索器管理器"""

from src.config import settings
from src.models import CurationReport, PaperCard, ResourceCard
from src.retrievers.arxiv import ArxivRetriever
from src.retrievers.semantic_scholar import SemanticScholarRetriever
from src.retrievers.dblp import DblpRetriever
from src.retrievers.github import GithubRetriever
from src.retrievers.papers_with_code import PapersWithCodeRetriever
from src.retrievers.huggingface import HuggingFaceRetriever
from src.retrievers.hackernews import HackerNewsRetriever

import logging

logger = logging.getLogger(__name__)


class RetrieverManager:
    """管理所有检索器，统一对外接口"""

    def __init__(self) -> None:
        self.arxiv = ArxivRetriever()
        self.semantic_scholar = SemanticScholarRetriever()
        self.dblp = DblpRetriever()
        self.github = GithubRetriever()
        self.papers_with_code = PapersWithCodeRetriever()
        self.huggingface = HuggingFaceRetriever()
        self.hackernews = HackerNewsRetriever()

    def search_all(
        self, query: str, max_results: int | None = None
    ) -> CurationReport:
        """全源检索：论文 + 资源"""
        max_r = max_results or settings.default_max_results

        # 论文检索
        papers: list[PaperCard] = []
        try:
            papers.extend(self.arxiv.search(query, max_r // 3))
            logger.info("arxiv: %d papers", len([p for p in papers if p.source == "arxiv"]))
        except Exception as e:
            logger.error("arxiv failed: %s", e)

        try:
            papers.extend(self.semantic_scholar.search(query, max_r // 3))
        except Exception as e:
            logger.error("semantic_scholar failed: %s", e)

        try:
            papers.extend(self.dblp.search(query, max_r // 3))
        except Exception as e:
            logger.error("dblp failed: %s", e)

        # PapersWithCode API 已废弃，跳过论文检索
        # 资源检索
        resources: list[ResourceCard] = []
        try:
            resources.extend(self.github.search_resources(query, 10))
        except Exception as e:
            logger.error("github failed: %s", e)

        try:
            resources.extend(self.huggingface.search_resources(query, 5))
        except Exception as e:
            logger.error("huggingface failed: %s", e)

        try:
            resources.extend(self.hackernews.search_resources(query, 5))
        except Exception as e:
            logger.error("hackernews failed: %s", e)

        # 构建报告
        classic = [p for p in papers if p.citation_count > 50]
        frontier = [p for p in papers if p.year >= 2023]

        report = CurationReport(
            topic=query,
            papers=papers,
            resources=resources,
            total_papers=len(papers),
            classic_count=len(classic),
            frontier_count=len(frontier),
            github_count=sum(
                1 for r in resources if r.type == "github_repo"
            ),
            benchmark_count=sum(
                1 for r in resources if r.type == "benchmark"
            ),
            dataset_count=sum(
                1 for r in resources if r.type == "dataset"
            ),
            queries=[query],
        )
        return report

    def close(self) -> None:
        for attr in (
            "arxiv", "semantic_scholar", "dblp", "github",
            "papers_with_code", "huggingface", "hackernews",
        ):
            retriever = getattr(self, attr)
            retriever.close()

    def __enter__(self) -> "RetrieverManager":
        return self

    def __exit__(self, *args) -> None:
        self.close()
