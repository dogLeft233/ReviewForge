"""ar5iv.org 论文内容检索器 — HTML5 格式，无 rate limit，免费无需 API Key

ar5iv.org 将 arXiv 论文的 LaTeX 源码转换为 HTML5，
可直接通过 URL pattern 访问，无需任何认证。

URL Pattern:
    https://ar5iv.org/html/<arxiv_id>    # 返回完整 HTML（推荐）
    https://ar5iv.org/abs/<arxiv_id>     # 重定向到 /html/

特点:
- 无 API Key，免费使用
- 无明确 rate limit（官方建议文明使用，不激进爬取）
- 论文内容以 HTML 呈现，比 PDF 更易解析
- 数据覆盖至 2026 年 4 月底，非实时
"""

import re
from typing import Any
from urllib.parse import quote

import httpx

from src.config import settings
from src.models import PaperCard
from src.retrievers.base import BaseRetriever

import logging

logger = logging.getLogger(__name__)

AR5IV_BASE = "https://ar5iv.org"


class Ar5ivRetriever(BaseRetriever):
    """ar5iv.org 全文内容检索器

    注意：ar5iv 本身不提供检索功能（无 search API），
    需要配合 arxiv retriever 做 ID 查询，再抓取全文。

    用法:
        with Ar5ivRetriever() as ar5iv:
            content = ar5iv.fetch_full_text("2301.00001")
    """

    name = "ar5iv"

    def search(self, query: str, max_results: int | None = None) -> list[PaperCard]:
        """ar5iv 无搜索功能，返回空列表提示用户使用 arxiv 检索 ID"""
        logger.warning(
            "ar5iv retriever 没有搜索功能，请使用 arxiv retriever 检索 paper ID，"
            "再用 fetch_full_text() 抓取全文"
        )
        return []

    def fetch_full_text(self, arxiv_id: str) -> str:
        """抓取指定 arxiv 论文的 HTML 全文

        Args:
            arxiv_id: 形如 "2301.00001" 或 "2301.00001v2"

        Returns:
            HTML 正文内容（原始字符串，未经解析）
        """
        url = f"{AR5IV_BASE}/html/{arxiv_id}"
        logger.info("ar5iv: fetching %s", url)

        resp = self._get(url, follow_redirects=True)
        if resp.status_code != 200:
            logger.error("ar5iv: %s returned %d", url, resp.status_code)
            return ""

        ct = resp.headers.get("content-type", "")
        if "text/html" not in ct and "application/xhtml+xml" not in ct:
            logger.warning("ar5iv: unexpected content-type %s for %s", ct, url)

        return resp.text

    def fetch_by_abs_page(self, arxiv_id: str) -> str:
        """通过 /abs/ 路径抓取（自动重定向到 /html/）"""
        url = f"{AR5IV_BASE}/abs/{arxiv_id}"
        logger.info("ar5iv: fetching abs page %s", url)
        resp = self._get(url, follow_redirects=True)
        return resp.text if resp.status_code == 200 else ""

    def extract_text_from_html(self, html: str) -> str:
        """从 HTML 中提取纯文本（基础实现，完整解析需配合 readability）"""
        text = html
        for tag in ["script", "style", "nav", "header", "footer"]:
            text = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE
            )
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
