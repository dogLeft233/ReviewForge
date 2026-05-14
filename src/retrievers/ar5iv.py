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

参考文献解析:
    - extract_bibliography(html): 提取参考文献列表（cite_key / title / authors / year）
    - extract_section_citations(html): 提取章节到引用 key 的映射
    - fetch_bibliography(arxiv_id): 直接获取指定论文的参考文献列表
"""

import re
import time
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.config import settings
from src.models import PaperCard
from src.retrievers.base import BaseRetriever

import logging

logger = logging.getLogger(__name__)

AR5IV_BASE = "https://ar5iv.org"
_MAX_RETRIES = 5
_RETRY_DELAY = 3.0
_CACHE_DIR = Path("tmp/ar5iv_cache")


def _get_cache_path(arxiv_id: str) -> Path:
    safe_id = arxiv_id.replace("/", "_").replace(".", "_")
    return _CACHE_DIR / f"{safe_id}.html"


class Ar5ivRetriever(BaseRetriever):
    """ar5iv.org 全文内容检索器

    注意：ar5iv 本身不提供检索功能（无 search API），
    需要配合 arxiv retriever 做 ID 查询，再抓取全文。

    参考文献解析方法:
        - extract_bibliography(html): 解析参考文献列表
        - extract_section_citations(html): 解析章节到引用 key 的映射
        - fetch_bibliography(arxiv_id): 抓取 + 解析一步完成

    用法:
        with Ar5ivRetriever() as ar5iv:
            bibs = ar5iv.fetch_bibliography("2301.00001")
            for b in bibs:
                print(b["cite_key"], b["title"])
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
        """抓取指定 arxiv 论文的 HTML 全文（含本地缓存）

        Args:
            arxiv_id: 形如 "2301.00001" 或 "2301.00001v2"

        Returns:
            HTML 正文内容（原始字符串，未经解析）
        """
        cache_path = _get_cache_path(arxiv_id)

        if cache_path.exists():
            logger.info("ar5iv: cache hit for %s", arxiv_id)
            return cache_path.read_text(encoding="utf-8")

        url = f"{AR5IV_BASE}/html/{arxiv_id}"
        logger.info("ar5iv: fetching %s", url)

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._get(url, follow_redirects=True)
                if resp.status_code == 429:
                    logger.warning("ar5iv: 429 rate limit, attempt %d/%d, sleeping %ds",
                                   attempt + 1, _MAX_RETRIES, _RETRY_DELAY)
                    time.sleep(_RETRY_DELAY)
                    continue
                if resp.status_code != 200:
                    logger.error("ar5iv: %s returned %d", url, resp.status_code)
                    return ""
                ct = resp.headers.get("content-type", "")
                if "text/html" not in ct and "application/xhtml+xml" not in ct:
                    logger.warning("ar5iv: unexpected content-type %s for %s", ct, url)

                html_content = resp.text

                try:
                    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(html_content, encoding="utf-8")
                    logger.info("ar5iv: cached %s (%d bytes)", arxiv_id, len(html_content))
                except Exception as cache_err:
                    logger.warning("ar5iv: failed to write cache for %s: %s", arxiv_id, cache_err)

                return html_content
            except Exception as e:
                logger.error("ar5iv: request failed attempt %d/%d: %s", attempt + 1, _MAX_RETRIES, e)
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY)
                continue
        return ""

    def fetch_by_abs_page(self, arxiv_id: str) -> str:
        """通过 /abs/ 路径抓取（自动重定向到 /html/）"""
        url = f"{AR5IV_BASE}/abs/{arxiv_id}"
        logger.info("ar5iv: fetching abs page %s", url)
        resp = self._get(url, follow_redirects=True)
        return resp.text if resp.status_code == 200 else ""

    def extract_text_from_html(self, html: str) -> str:
        """从 HTML 中提取纯文本（基础实现）"""
        text = html
        for tag in ["script", "style", "nav", "header", "footer"]:
            text = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE
            )
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_bibliography(self, html: str) -> list[dict]:
        """从 ar5iv HTML 中提取参考文献列表

        支持两种格式:
          1. 语义格式（推荐）: <span class="ltx_bibtitle">
          2. 原始格式: 只有 <span class="ltx_bibblock"> 含纯文本

        Args:
            html: ar5iv HTML 字符串

        Returns:
            [{"cite_key": "bib.bib1", "title": "...", "authors": "...", "year": "..."}, ...]
        """
        if not html:
            return []
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return []

        bib_list = soup.find("ul", class_="ltx_biblist")
        if not bib_list:
            return []

        results: list[dict] = []

        for li in bib_list.find_all("li", recursive=False):
            cite_key = li.get("id", "").strip()
            if not cite_key:
                continue

            title = ""
            authors = ""
            year = ""

            # 方式1: 优先找语义化的 ltx_bibtitle / ltx_bibauthors / ltx_bibyear
            title_span = li.find("span", class_="ltx_bibtitle")
            if title_span:
                title = title_span.get_text(strip=True).rstrip('.').rstrip(',')

            author_span = li.find("span", class_="ltx_bibauthors")
            if author_span:
                authors = author_span.get_text(strip=True)

            year_span = li.find("span", class_="ltx_bibyear")
            if year_span:
                year = year_span.get_text(strip=True)
            else:
                # Fallback: 从所有 ltx_bibblock 文本中找年份
                for block in li.find_all("span", class_="ltx_bibblock"):
                    texts = [str(c) for c in block.descendants if isinstance(c, str)]
                    block_text = " ".join(texts)
                    year_m = re.search(r'\b(19|20)\d{2}\b', block_text)
                    if year_m:
                        year = year_m.group(0)
                        break

            # 方式2: 如果没找到 ltx_bibtitle，从 ltx_bibblock 纯文本中解析
            if not title:
                bibblocks = li.find_all("span", class_="ltx_bibblock")
                for block in bibblocks:
                    # 取 block 内所有纯文本（忽略 <a> 等标签）
                    texts: list[str] = []
                    for child in block.descendants:
                        if isinstance(child, str):
                            texts.append(str(child))
                    full_text = " ".join(texts)

                    if len(full_text) < 5:
                        continue

                    # 优先从引号中提取标题（支持 ASCII 和 Unicode smart quotes）
                    quote_pattern = r'["\u201c\u201d]([^\"\u201c\u201d]{10,300})["\u201c\u201d]'
                    quote_title = re.search(quote_pattern, full_text)
                    if quote_title:
                        title = quote_title.group(1).strip().rstrip(',').rstrip('.')
                        break

                    # 从 full_text 中找年份（19xx / 20xx）
                    if not year:
                        year_m = re.search(r'\b(19|20)\d{2}\b', full_text)
                        if year_m:
                            year = year_m.group(0)

                    # 如果还没拿到标题，取第一个逗号前的文字作为标题
                    if not title and len(full_text) > 10:
                        # 去掉 "Accessed" 等前缀
                        clean = re.sub(r'^(Accessed|n\.d\.)[^,]*,?\s*', '', full_text, flags=re.IGNORECASE)
                        m = re.match(r'^[^,]{3,200}', clean)
                        if m:
                            title = m.group(0).strip().rstrip(',').rstrip('.')
                            break

            results.append({
                "cite_key": cite_key,
                "title": title[:300] if title else "",
                "authors": authors[:200] if authors else "",
                "year": year,
            })

        return results

    def extract_section_citations(self, html: str) -> dict[str, list[str]]:
        """从 ar5iv HTML 中提取章节到引用 key 的映射

        Args:
            html: ar5iv HTML 字符串

        Returns:
            {"1. Introduction": ["bib.bib1", "bib.bib3"], "2. Related Work": [...], ...}

        逻辑:
            1. 遍历 <section> 标签，提取 id 和标题
            2. 在每个 section 的正文中找 <cite> 标签
            3. <cite> 内的 <a class="ltx_ref" href="#bib.xxx"> 指向参考文献
            4. 也支持直接 <a class="ltx_ref" href="#bib.xxx"> 形式
        """
        if not html:
            return {}
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return {}

        result: dict[str, list[str]] = {}

        for section in soup.find_all("section"):
            sec_id = section.get("id", "")
            if not sec_id:
                continue

            # 获取章节标题: <h1>/<h2>/<h3> 等
            heading = section.find(
                lambda tag: tag.name in ("h1", "h2", "h3", "h4") and tag.get_text(strip=True)
            )
            if not heading:
                continue
            sec_title = heading.get_text(strip=True)
            if not sec_title:
                continue

            # 在 section 正文中找引用
            cite_keys: set[str] = set()

            # 方式1: <cite> 标签内的 <a class="ltx_ref" href="#bib.xxx">
            for cite in section.find_all("cite"):
                for a in cite.find_all("a", class_="ltx_ref"):
                    href = a.get("href", "")
                    if href.startswith("#"):
                        key = href[1:].strip()
                        if key:
                            cite_keys.add(key)

            # 方式2: 直接 <a class="ltx_ref" href="#bib.xxx">（不在 cite 里）
            for a in section.find_all("a", class_="ltx_ref"):
                href = a.get("href", "")
                if href.startswith("#"):
                    key = href[1:].strip()
                    if key:
                        cite_keys.add(key)

            if cite_keys:
                result[sec_title] = sorted(cite_keys)

        return result

    def fetch_bibliography(self, arxiv_id: str) -> list[dict]:
        """获取指定论文的参考文献列表

        Args:
            arxiv_id: 形如 "2301.00001" 或 "2301.00001v2"

        Returns:
            [{"cite_key": "bib.bib1", "title": "...", "authors": "...", "year": "..."}, ...]
        """
        html = self.fetch_full_text(arxiv_id)
        return self.extract_bibliography(html)


def clear_ar5iv_cache() -> int:
    """清除 ar5iv 本地缓存，返回删除的文件数"""
    if not _CACHE_DIR.exists():
        return 0
    count = 0
    for f in _CACHE_DIR.iterdir():
        if f.suffix == ".html":
            f.unlink()
            count += 1
    logger.info("ar5iv: cleared %d cache files", count)
    return count
