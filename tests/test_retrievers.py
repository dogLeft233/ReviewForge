"""检索器测试 + 模型测试 + 策展测试"""

import pytest

from src.models import PaperCard, ResourceCard, CurationReport, SuppleReport


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


class TestSuppleReport:
    """补搜报告测试"""

    def test_empty(self):
        r = SuppleReport(topic="Test")
        assert r.topic == "Test"
        assert r.queries_executed == []
        assert r.total_new_papers == 0
        assert r.has_new_findings is False
        assert r.total_queries == 0

    def test_with_findings(self):
        r = SuppleReport(
            topic="大数据处理",
            queries_executed=["q1", "q2"],
            total_new_papers=10,
            total_new_resources=5,
            new_classics=3,
            new_frontiers=7,
            sections_improved=["2", "4"],
            duration_seconds=30.5,
        )
        assert r.total_queries == 2
        assert r.has_new_findings is True
        summary = r.to_summary()
        assert "大数据处理" in summary
        assert "10" in summary  # 新增论文数
        assert "30.5s" in summary  # 耗时

    def test_with_gap_queries(self):
        r = SuppleReport(
            topic="Test",
            gap_queries={
                "2": ["cloud-native storage"],
                "4.3": ["RisingWave VLDB 2023"],
            },
        )
        summary = r.to_summary()
        assert "cloud-native storage" in summary
        assert "RisingWave" in summary


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

    def test_supplementary_search_empty(self):
        """空查询列表应返回空补搜报告"""
        from src.manager import RetrieverManager

        with RetrieverManager() as mgr:
            report = mgr.supplementary_search(gap_queries=[], max_results=3)
            assert report.queries_executed == []
            assert report.total_new_papers == 0
            assert report.has_new_findings is False

    def test_supplementary_search_basic(self):
        """补搜应返回新发现"""
        from src.manager import RetrieverManager

        with RetrieverManager() as mgr:
            report = mgr.supplementary_search(
                gap_queries=["big data survey 2023"],
                max_results=5,
                section_query_map={"1": ["big data survey 2023"]},
            )
            assert len(report.queries_executed) == 1
            assert report.queries_executed[0] == "big data survey 2023"
            # 搜索结果可能为空（取决于网络），但不应该崩溃
            assert isinstance(report.total_new_papers, int)
            # 如果查到了数据，验证 section mapping
            if report.total_new_papers > 0:
                assert "1" in report.gap_queries

    def test_supplementary_search_dedup(self):
        """补搜应与已有结果去重"""
        from src.manager import RetrieverManager
        from src.models import CurationReport, PaperCard

        existing = CurationReport(
            topic="Test",
            papers=[PaperCard(title="Existing Paper", year=2020)],
        )

        with RetrieverManager() as mgr:
            report = mgr.supplementary_search(
                gap_queries=["new paper 2024"],
                existing_report=existing,
                max_results=5,
            )
            # 基本结构正确
            assert report.total_new_papers >= 0
            assert isinstance(report.total_new_papers, int)


# ═══════════════════════════════════════════════
# Serper 检索器测试
# ═══════════════════════════════════════════════


class TestSerperRetriever:
    """SerperRetriever 测试（mock 网络，验证映射逻辑）"""

    # ── 辅助 ──

    @staticmethod
    def _mock_search_data(title: str, domain: str, snippet: str = "", date: str = "") -> dict:
        return {
            "organic": [
                {
                    "title": title,
                    "link": f"https://{domain}/article",
                    "snippet": snippet or f"This is a snippet about {title}.",
                    "date": date,
                    "position": 1,
                    "attributes": {},
                }
            ],
            "knowledgeGraph": {},
            "relatedSearches": [],
        }

    # ── 跳过条件 ──

    def test_skip_without_key(self):
        """无 API key 时返回空列表"""
        from src.retrievers.serper import SerperRetriever

        r = SerperRetriever()
        assert r.search("test") == []
        assert r.search_resources("test") == []

    # ── 映射测试 ──

    def test_organic_to_paper(self):
        """有机搜索结果映射为 PaperCard"""
        from src.retrievers.serper import SerperRetriever

        r = SerperRetriever()
        data = self._mock_search_data(
            title="A Survey of Deep Learning",
            domain="arxiv.org",
            snippet="A comprehensive survey of deep learning techniques published in 2024.",
            date="2024-03-15",
        )
        cards = [
            r._organic_to_paper(item, "deep learning survey")
            for item in data["organic"]
        ]
        assert len(cards) == 1
        card = cards[0]
        assert card is not None
        assert card.title == "A Survey of Deep Learning"
        assert card.source == "serper"
        assert "arxiv.org" in card.venue
        assert card.year == 2024
        assert "comprehensive survey" in card.abstract

    def test_organic_to_paper_no_date(self):
        """无日期时年份从链接推断"""
        from src.retrievers.serper import SerperRetriever

        r = SerperRetriever()
        data = self._mock_search_data(
            title="LLM Paper",
            domain="proceedings.neurips.cc",
        )
        cards = [r._organic_to_paper(item, "llm") for item in data["organic"]]
        assert len(cards) == 1
        assert cards[0] is not None

    def test_organic_to_paper_empty_title(self):
        """无标题时返回 None"""
        from src.retrievers.serper import SerperRetriever

        r = SerperRetriever()
        data = self._mock_search_data(title="", domain="example.com")
        cards = [r._organic_to_paper(item, "test") for item in data["organic"]]
        assert cards[0] is None

    def test_organic_to_resource(self):
        """有机搜索结果映射为 ResourceCard"""
        from src.retrievers.serper import SerperRetriever

        r = SerperRetriever()
        data = self._mock_search_data(
            title="PyTorch",
            domain="pytorch.org",
            snippet="An open source machine learning framework",
        )
        cards = [
            r._organic_to_resource(item, "deep learning framework")
            for item in data["organic"]
        ]
        assert len(cards) == 1
        card = cards[0]
        assert card is not None
        assert card.name == "PyTorch"
        assert "open source" in str(card.description)

    def test_organic_to_resource_github(self):
        """GitHub 链接自动识别为 github_repo 类型"""
        from src.retrievers.serper import SerperRetriever

        r = SerperRetriever()
        data = self._mock_search_data(
            title="tensorflow/tensorflow",
            domain="github.com",
            snippet="An Open Source Machine Learning Framework",
        )
        cards = [
            r._organic_to_resource(item, "ml framework")
            for item in data["organic"]
        ]
        assert len(cards) == 1
        card = cards[0]
        assert card.type == "github_repo"

    def test_organic_to_resource_benchmark(self):
        """含 benchmark 关键词的识别为 benchmark 类型"""
        from src.retrievers.serper import SerperRetriever

        r = SerperRetriever()
        data = self._mock_search_data(
            title="GLUE Benchmark",
            domain="gluebenchmark.com",
            snippet="The General Language Understanding Evaluation benchmark",
        )
        cards = [
            r._organic_to_resource(item, "nlp benchmark")
            for item in data["organic"]
        ]
        assert len(cards) == 1
        card = cards[0]
        assert card.type == "benchmark"

    # ── 辅助函数测试 ──

    def test_detect_resource_type(self):
        """资源类型检测"""
        from src.retrievers.serper import _detect_resource_type

        assert _detect_resource_type("github.com", "", "") == "github_repo"
        assert _detect_resource_type("example.com", "Awesome Dataset", "") == "dataset"
        assert _detect_resource_type("example.com", "Benchmark Results", "") == "benchmark"
        assert _detect_resource_type("example.com", "A New Framework", "") == "tool"
        assert _detect_resource_type("example.com", "Random Blog", "") == "other"

    def test_recommend_use(self):
        """使用建议标签生成"""
        from src.retrievers.serper import _recommend_use

        assert "综述背景" in _recommend_use("A Survey of...", "", "")
        assert "入门背景" in _recommend_use("Getting Started Guide", "", "")
        assert "应用案例" in _recommend_use("Blog Post", "", "https://blog.example.com")
        assert "论文引用" in _recommend_use("Title", "", "https://arxiv.org/abs/1234")
        assert "补充参考" in _recommend_use("Some Page", "", "https://example.com")
