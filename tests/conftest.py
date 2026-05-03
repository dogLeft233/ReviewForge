"""pytest fixtures"""

from pathlib import Path

import pytest

from src.config import settings


@pytest.fixture
def sample_arxiv_xml() -> str:
    """示例 arXiv API 响应 XML"""
    return """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001</id>
    <title>   Example Paper Title   </title>
    <summary>   This is an example abstract for testing purposes.   </summary>
    <author><name>Author One</name></author>
    <author><name>Author Two</name></author>
    <published>2023-01-01T00:00:00Z</published>
    <category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2302.00002</id>
    <title>Another Test Paper</title>
    <summary>Another abstract for testing.</summary>
    <author><name>Author Three</name></author>
    <published>2023-02-01T00:00:00Z</published>
    <category term="cs.LG"/>
  </entry>
</feed>"""


@pytest.fixture
def sample_s2_response() -> dict:
    """示例 Semantic Scholar API 响应"""
    return {
        "data": [
            {
                "paperId": "abc123",
                "title": "Big Data Processing Survey",
                "authors": [{"name": "Alice"}, {"name": "Bob"}],
                "year": 2023,
                "venue": "VLDB",
                "abstract": "A comprehensive survey of big data processing.",
                "citationCount": 150,
                "externalIds": {"ArXiv": "2301.00001"},
                "url": "https://arxiv.org/abs/2301.00001",
            }
        ]
    }


@pytest.fixture
def sample_github_response() -> dict:
    """示例 GitHub API 响应"""
    return {
        "items": [
            {
                "full_name": "apache/spark",
                "name": "spark",
                "html_url": "https://github.com/apache/spark",
                "stargazers_count": 40000,
                "description": "Apache Spark - unified analytics engine",
                "topics": ["big-data", "distributed-computing", "machine-learning"],
                "owner": {"login": "apache"},
                "has_pages": False,
                "license": {"spdx_id": "Apache-2.0"},
                "language": "Scala",
            }
        ]
    }


@pytest.fixture
def tmp_output(tmp_path: Path) -> str:
    """临时输出目录"""
    return str(tmp_path / "output")
