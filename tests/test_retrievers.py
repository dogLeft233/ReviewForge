"""检索器测试 + 模型测试 + 策展测试"""

import pytest

from src.models import PaperCard, ResourceCard, CurationReport


# ═══════════════════════════════════════════════
# 模型测试
# ═══════════════════════════════════════════════

class TestPaperCard:
    def test_minimal_card(self):
        card = PaperCard(title="Test Paper")
        assert card.title == "Test Paper"
        assert card.authors == []
        assert card.year == 0
        assert card.citation_count == 0
        assert card.source == ""

    def test_full_card(self):
        card = PaperCard(
            title="Full Paper",
            authors=["Author A"],
            year=2023,
            venue="NeurIPS",
            abstract="An important paper.",
            citation_count=200,
            url="https://arxiv.org/abs/2301.00001",
            source="arxiv",
            method_category="distributed_systems",
            key_contribution="Proposed a new framework",
            relevance_score=0.95,
        )
        assert card.year == 2023
        assert card.citation_count == 200
        assert card.relevance_score == 0.95


class TestResourceCard:
    def test_github_resource(self):
        card = ResourceCard(
            name="apache/spark",
            type="github_repo",
            url="https://github.com/apache/spark",
            stars=40000,
            owner="apache",
        )
        assert card.stars == 40000
        assert card.type == "github_repo"

    def test_default_fields(self):
        card = ResourceCard(name="test", type="dataset")
        assert card.quality_indicators == []
        assert card.related_papers == []
        assert card.stars == 0


class TestCurationReport:
    def test_empty_report(self):
        report = CurationReport(topic="Test Topic")
        assert report.topic == "Test Topic"
        assert report.papers == []
        assert report.resources == []
        assert report.total_papers == 0

    def test_with_papers(self):
        papers = [
            PaperCard(title="Paper A", year=2020, citation_count=100),
            PaperCard(title="Paper B", year=2024, citation_count=10),
        ]
        report = CurationReport(
            topic="Big Data",
            papers=papers,
            total_papers=len(papers),
            classic_count=1,
            frontier_count=1,
        )
        assert report.total_papers == 2
        assert report.classic_count == 1
        assert report.frontier_count == 1


# ═══════════════════════════════════════════════
# arXiv 解析测试
# ═══════════════════════════════════════════════

class TestArxivParsing:
    def test_parse_xml(self, sample_arxiv_xml):
        from src.retrievers.arxiv import ArxivRetriever

        cards = ArxivRetriever._parse(sample_arxiv_xml)
        assert len(cards) == 2
        assert cards[0].title == "Example Paper Title"
        assert cards[1].title == "Another Test Paper"
        assert cards[0].authors == ["Author One", "Author Two"]
        assert cards[0].year == 2023
        assert cards[1].year == 2023

    def test_cs_ai_category(self, sample_arxiv_xml):
        from src.retrievers.arxiv import ArxivRetriever

        cards = ArxivRetriever._parse(sample_arxiv_xml)
        assert cards[0].method_category == "cs.AI"
        assert cards[1].method_category == "cs.LG"


# ═══════════════════════════════════════════════
# Semantic Scholar 解析测试
# ═══════════════════════════════════════════════

class TestSemanticScholarParsing:
    def test_parse_response(self, sample_s2_response):
        from src.retrievers.semantic_scholar import SemanticScholarRetriever

        cards = SemanticScholarRetriever._parse(sample_s2_response)
        assert len(cards) == 1
        assert cards[0].title == "Big Data Processing Survey"
        assert cards[0].citation_count == 150
        assert cards[0].year == 2023
        assert cards[0].venue == "VLDB"


# ═══════════════════════════════════════════════
# GitHub 解析测试
# ═══════════════════════════════════════════════

class TestGitHubParsing:
    def test_parse_resources(self, sample_github_response):
        from src.retrievers.github import GithubRetriever

        cards = GithubRetriever._parse_resources(sample_github_response, "big data")
        assert len(cards) == 1
        assert cards[0].name == "apache/spark"
        assert cards[0].stars == 40000
        assert cards[0].type == "github_repo"


# ═══════════════════════════════════════════════
# 策展测试
# ═══════════════════════════════════════════════

class TestCurator:
    def test_empty_curation(self):
        from src.curation.curator import Curator

        curator = Curator()
        report = CurationReport(topic="Test")
        result = curator.curate(report)

        assert result.missing_directions is not None
        assert result.insights == ["暂无数据"]

    def test_curation_with_papers(self):
        from src.curation.curator import Curator

        papers = [
            PaperCard(title="A", year=2020, citation_count=200, venue="VLDB", method_category="distributed"),
            PaperCard(title="B", year=2023, citation_count=80, venue="VLDB", method_category="stream"),
            PaperCard(title="C", year=2024, citation_count=10, venue="SIGMOD", method_category="ml"),
        ]
        report = CurationReport(topic="Big Data", papers=papers, total_papers=3)
        curator = Curator()
        result = curator.curate(report)

        assert result.classic_count == 2  # A (200) + B (80) > 50
        assert result.frontier_count == 2  # B (2023) + C (2024) >= 2023
        assert len(result.categories) == 3
        # missing_directions checks thresholds (both < 3)
        assert "GitHub 资源不足" in result.missing_directions

    def test_evaluate(self):
        from src.curation.curator import Curator

        full_report = CurationReport(
            topic="Test",
            total_papers=30,
            classic_count=10,
            frontier_count=8,
            github_count=5,
            benchmark_count=3,
        )
        curator = Curator()
        assert curator.evaluate(full_report) == "优秀"

        poor_report = CurationReport(
            topic="Test",
            total_papers=3,
            classic_count=0,
            frontier_count=1,
            github_count=0,
            benchmark_count=0,
        )
        assert curator.evaluate(poor_report) == "需补充"


# ═══════════════════════════════════════════════
# 管理器测试
# ═══════════════════════════════════════════════

class TestRetrieverManager:
    def test_close_on_exit(self):
        from src.manager import RetrieverManager

        with RetrieverManager() as mgr:
            assert mgr.arxiv is not None
            assert mgr.semantic_scholar is not None
            assert mgr.github is not None
            assert mgr.papers_with_code is not None
            assert mgr.hackernews is not None

    def test_search_all_produces_report(self):
        from src.manager import RetrieverManager

        with RetrieverManager() as mgr:
            report = mgr.search_all("test query", max_results=5)
            assert report.topic == "test query"
            assert isinstance(report.total_papers, int)
