"""Semantic Scholar 论文检索器 — 真实引文关系图

API 文档: https://api.semanticscholar.org/api-docs/
速率限制:
  - 无 key: 1000 req/s (共享，建议用于生产环境)
  - 有 key (free tier): 1 RPS
  - 有 key (introductory): 1 RPS
  - 有 key (进阶): 需要申请
"""

import logging
import httpx
from typing import Any
from urllib.parse import quote

from src.models import PaperCard
from src.retrievers.base import BaseRetriever
from src.config import settings

logger = logging.getLogger(__name__)

S2_API_BASE = "https://api.semanticscholar.org/graph/v1/paper"
FIELDS = "title,abstract,year,authors.name,citationCount,externalIds,openAccessPdf"


class SemanticScholarRetriever(BaseRetriever):
    """Semantic Scholar API 检索器

    用法:
        with SemanticScholarRetriever() as s2:
            papers = s2.search("LoRA fine-tuning", max_results=20)

        # 查单篇论文的参考文献（引文关系）
        with SemanticScholarRetriever() as s2:
            refs = s2.get_references("arxiv:2301.00001")

        # 查引用该论文的所有论文
        with SemanticScholarRetriever() as s2:
            cites = s2.get_citations("arxiv:2301.00001")
    """

    name = "semantic_scholar"

    def __init__(self) -> None:
        super().__init__()
        self._api_key = getattr(settings, "semantic_scholar_api_key", "") or None

    def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        """带可选 API Key header 的 GET"""
        if self._api_key:
            kwargs.setdefault("headers", {})
            kwargs["headers"]["x-api-key"] = self._api_key
        return super()._get(url, **kwargs)

    # ── 通用检索 ──────────────────────────────────────────────

    def search(
        self, query: str, max_results: int | None = None
    ) -> list[PaperCard]:
        if max_results is None:
            max_results = 30

        url = (
            f"{S2_API_BASE}/paper/search"
            f"?query={quote(query)}"
            f"&limit={max_results}"
            f"&fields={FIELDS}"
        )
        logger.info("s2: search '%s' (max=%d)", query, max_results)
        resp = self._get(url)
        return self._parse_search(resp.text)

    # ── 引文关系 API ─────────────────────────────────────────

    def get_references(
        self, paper_id: str, *, max_results: int = 20
    ) -> list[PaperCard]:
        """获取指定论文的参考文献列表

        Args:
            paper_id: 论文 ID，支持多种格式:
                - arXiv ID: "ArXiv:2301.00001"
                - Semantic Scholar ID: "S2......"
                - DOI: "doi:10.1000/..."
            max_results: 最多返回参考文献数（默认 20）

        Returns:
            参考文献 PaperCard 列表
        """
        url = (
            f"{S2_API_BASE}/{quote(paper_id)}/references"
            f"?limit={max_results}&fields={FIELDS}"
        )
        logger.info("s2: references of '%s'", paper_id)
        resp = self._get(url)
        return self._parse_paper_list(resp.text)

    def get_citations(
        self, paper_id: str, *, max_results: int = 20
    ) -> list[PaperCard]:
        """获取引用了指定论文的所有论文（被引用列表）

        Args:
            paper_id: 论文 ID，支持格式同 get_references
            max_results: 最多返回引用数（默认 20）

        Returns:
            引用方 PaperCard 列表
        """
        url = (
            f"{S2_API_BASE}/{quote(paper_id)}/citations"
            f"?limit={max_results}&fields={FIELDS}"
        )
        logger.info("s2: citations of '%s'", paper_id)
        resp = self._get(url)
        return self._parse_paper_list(resp.text)

    def get_paper(self, paper_id: str) -> PaperCard | None:
        """获取单篇论文的完整信息"""
        url = f"{S2_API_BASE}/{quote(paper_id)}?fields={FIELDS}"
        logger.info("s2: get_paper '%s'", paper_id)
        resp = self._get(url)
        cards = self._parse_paper_list(resp.text)
        return cards[0] if cards else None

    # ── 解析 ─────────────────────────────────────────────────

    @staticmethod
    def _parse_search(json_text: str) -> list[PaperCard]:
        """解析 search API 返回"""
        import json

        data = json.loads(json_text)
        return [
            SemanticScholarRetriever._paper_from_dict(p)
            for p in data.get("data", [])
        ]

    @staticmethod
    def _parse_paper_list(json_text: str) -> list[PaperCard]:
        """解析 references / citations API 返回"""
        import json

        data = json.loads(json_text)
        # references/citations 返回 { "data": [{ "citedPaper": {...} }, ...] }
        # 或 { "data": [{ "citingPaper": {...} }, ...] }
        papers = []
        for item in data.get("data", []):
            # 尝试多种可能的 paper 字段
            paper_dict = (
                item.get("citedPaper")
                or item.get("citingPaper")
                or item
            )
            if paper_dict and paper_dict.get("title"):
                papers.append(
                    SemanticScholarRetriever._paper_from_dict(paper_dict)
                )
        return papers

    @staticmethod
    def _paper_from_dict(d: dict) -> PaperCard:
        """从 S2 API dict 构造 PaperCard"""
        external = d.get("externalIds", {}) or {}
        arxiv_id = external.get("ArXiv", "") or external.get("arXiv", "") or ""
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else d.get("url", "")

        oa_pdf = d.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url") or ""

        return PaperCard(
            title=d.get("title", ""),
            authors=[a.get("name", "") for a in d.get("authors", [])],
            year=d.get("year", 0) or 0,
            abstract=d.get("abstract", ""),
            citation_count=d.get("citationCount", 0) or 0,
            url=arxiv_url or pdf_url or d.get("url", ""),
            source="semantic_scholar",
        )
