"""Boost 增强 — 经典论文加权 + 高引用加权"""

import logging
from typing import Optional

from src.searcher.models import BoostConfig, ClassicalPaper
from src.searcher.result import SearchResult

logger = logging.getLogger(__name__)


def _build_url_to_paper(classical_papers: list[ClassicalPaper]) -> dict[str, ClassicalPaper]:
    """从经典论文列表构建 URL → ClassicalPaper 映射"""
    url_map: dict[str, ClassicalPaper] = {}
    for paper in classical_papers:
        if paper.url:
            url_map[paper.url] = paper
        if paper.arxivid:
            # 标准 arXiv URL 格式
            arxiv_url = f"https://arxiv.org/abs/{paper.arxivid}"
            url_map[arxiv_url] = paper
            # 也有可能用 abs/ URL 变体
            arxiv_url2 = f"https://arxiv.org/abs/{paper.arxivid}"
            url_map[arxiv_url2] = paper
    return url_map


def apply_boost(
    results: list[SearchResult],
    boost_config: BoostConfig,
    override: Optional[dict[str, float]] = None,
) -> list[SearchResult]:
    """对搜索结果应用 Boost 策略

    Args:
        results: SearchResult 列表
        boost_config: Boost 配置
        override: 可选的 URL → boost_factor 覆盖（用于 Rerank 结果的额外加权）

    Returns:
        Boost 后的 SearchResult 列表（in-place 修改 rank_score）
    """
    if not results:
        return results

    logger.debug("apply_boost: input=%d results", len(results))

    # URL → ClassicalPaper 映射
    url_to_paper = _build_url_to_paper(boost_config.classical_papers)

    boosted = []
    for r in results:
        # 深拷贝避免修改原始结果
        new_result = SearchResult(
            title=r.title,
            url=r.url,
            snippet=r.snippet,
            rank_score=r.rank_score,
            source=r.source,
            category=r.category,
            published_date=r.published_date,
            domain=r.domain,
        )

        boost = 1.0

        # 1. 经典论文 Boost
        if r.url in url_to_paper:
            paper = url_to_paper[r.url]
            boost *= paper.boost_factor
            logger.debug("Boost: url=%s matched classical paper '%s', factor=%.2f",
                         r.url, paper.title, paper.boost_factor)

        # 2. 高引用 Boost（基于 Semantic Scholar citation_count）
        if c := getattr(r, "citation_count", 0):
            if c >= boost_config.high_citation_threshold:
                boost *= boost_config.high_citation_boost
                logger.debug(
                    "Boost: url=%s citation_count=%d >= %d, factor=%.2f",
                    r.url, c, boost_config.high_citation_threshold,
                    boost_config.high_citation_boost,
                )

        # 3. 覆盖加权（来自 Rerank 优化）
        if override and r.url in override:
            boost *= override[r.url]
            logger.debug("Boost: url=%s override factor=%.2f", r.url, override[r.url])

        new_result.rank_score = round(new_result.rank_score * boost, 4)
        boosted.append(new_result)

    logger.debug("apply_boost: output=%d results", len(boosted))
    return boosted