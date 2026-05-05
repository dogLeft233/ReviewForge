"""All-in-One 检索器管理器"""

import time

from src.config import settings
from src.models import CurationReport, PaperCard, ResourceCard, SuppleReport
from src.retrievers.arxiv import ArxivRetriever
from src.retrievers.semantic_scholar import SemanticScholarRetriever
from src.retrievers.dblp import DblpRetriever
from src.retrievers.github import GithubRetriever
from src.retrievers.papers_with_code import PapersWithCodeRetriever
from src.retrievers.huggingface import HuggingFaceRetriever
from src.retrievers.hackernews import HackerNewsRetriever
from src.retrievers.serper import SerperRetriever

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
        self.serper = SerperRetriever()

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

        # Google Serper —— 通用搜索，补充论文+资源
        try:
            papers.extend(self.serper.search(query, max_r // 5))
        except Exception as e:
            logger.error("serper search failed: %s", e)
        try:
            resources.extend(self.serper.search_resources(query, max_r // 5))
        except Exception as e:
            logger.error("serper resources failed: %s", e)

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

    def supplementary_search(
        self,
        gap_queries: list[str],
        existing_report: CurationReport | None = None,
        section_query_map: dict[str, list[str]] | None = None,
        max_results: int | None = None,
    ) -> SuppleReport:
        """对细纲缺口执行补搜

        接收来自 Outline 的缺口查询列表，逐个调用 search_all() 执行检索，
        并与已有结果去重。

        Args:
            gap_queries: 缺口的查询字符串列表
            existing_report: 已有的检索结果（用于去重）
            section_query_map: 章节→查询映射 {section_id: [queries]}
            max_results: 每个查询的最大结果数

        Returns:
            SuppleReport 对象（含新发现汇总）
        """
        start = time.time()

        # 去重
        unique_queries = list(dict.fromkeys(gap_queries))

        result_map: dict[str, CurationReport] = {}
        all_new_papers: list[str] = []  # titles for dedup
        total_new_papers = 0
        total_new_resources = 0
        new_classics = 0
        new_frontiers = 0

        # 已有论文标题（用于去重）
        existing_titles: set[str] = set()
        if existing_report:
            for p in existing_report.papers:
                if p.title:
                    existing_titles.add(p.title.lower().strip())

        for q in unique_queries:
            logger.info("supplementary search: %r", q)
            try:
                report = self.search_all(q, max_results)
            except Exception as e:
                logger.warning("supplementary search %r failed: %s", q, e)
                continue

            result_map[q] = report

            # 统计新增项
            new_papers = 0
            for p in report.papers:
                key = p.title.lower().strip() if p.title else ""
                if key and key not in existing_titles:
                    new_papers += 1
                    existing_titles.add(key)
                    if p.citation_count > 50:
                        new_classics += 1
                    if p.year >= 2023:
                        new_frontiers += 1

            total_new_papers += new_papers
            total_new_resources += len(report.resources)

        # 构建章节映射（仅记录有补搜的章节）
        sections_improved: list[str] = []
        gap_mapping: dict[str, list[str]] = {}
        if section_query_map:
            for sec_id, queries in section_query_map.items():
                # 只要有任何该章节的查询被执行了且查到东西
                active_queries = [q for q in queries if q in result_map]
                if active_queries:
                    gap_mapping[sec_id] = active_queries
                    sections_improved.append(sec_id)

        duration = time.time() - start

        return SuppleReport(
            topic=existing_report.topic if existing_report else "补搜",
            queries_executed=unique_queries,
            result_map=result_map,
            total_new_papers=total_new_papers,
            total_new_resources=total_new_resources,
            new_classics=new_classics,
            new_frontiers=new_frontiers,
            sections_improved=sections_improved,
            gap_queries=gap_mapping,
            duration_seconds=duration,
        )

    def close(self) -> None:
        for attr in (
            "arxiv", "semantic_scholar", "dblp", "github",
            "papers_with_code", "huggingface", "hackernews",
            "serper",
        ):
            retriever = getattr(self, attr)
            retriever.close()

    def __enter__(self) -> "RetrieverManager":
        return self

    def __exit__(self, *args) -> None:
        self.close()
