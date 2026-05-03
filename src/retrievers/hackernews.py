"""Hacker News 社区讨论检索器（Algolia API）"""

from urllib.parse import quote

from src.models import ResourceCard
from src.retrievers.base import BaseRetriever

import logging

logger = logging.getLogger(__name__)

HN_ALGOLIA = "https://hn.algolia.com/api/v1"


class HackerNewsRetriever(BaseRetriever):
    """Hacker News 讨论检索器（Algolia 免费搜索 API）"""

    name = "hackernews"

    def search(
        self, query: str, max_results: int | None = None
    ) -> list:
        """HN 不直接返回论文"""
        return []

    def search_resources(
        self, query: str, max_results: int | None = None
    ) -> list[ResourceCard]:
        """搜索 HN 上与给定主题相关的高质量讨论"""
        if max_results is None:
            max_results = 10

        # 按 points 排序，只搜索 story 类型
        url = (
            f"{HN_ALGOLIA}/search"
            f"?query={quote(query)}"
            f"&tags=story"
            f"&hitsPerPage={max_results}"
            f"&numericFilters=points>10"
        )

        logger.info(
            "hackernews: searching discussions '%s' (max=%d)", query, max_results
        )
        try:
            resp = self._get(url)
            data = resp.json()
            return self._parse(data, query)
        except Exception as e:
            logger.warning("hackernews: search failed: %s", e)
            return []

    def search_popular(
        self, query: str, max_results: int = 5
    ) -> list[ResourceCard]:
        """搜索评分最高的讨论"""
        url = (
            f"{HN_ALGOLIA}/search"
            f"?query={quote(query)}"
            f"&tags=story"
            f"&hitsPerPage={max_results}"
            f"&numericFilters=points>50"
        )

        try:
            resp = self._get(url)
            data = resp.json()
            return self._parse(data, query)
        except Exception as e:
            logger.warning("hackernews: popular search failed: %s", e)
            return []

    # ── 解析 ──

    @staticmethod
    def _parse(data: dict, query: str) -> list[ResourceCard]:
        cards: list[ResourceCard] = []
        for hit in data.get("hits", []):
            points = hit.get("points", 0) or 0
            num_comments = hit.get("num_comments", 0) or 0
            title = hit.get("title", "")
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"

            # 过滤低质量
            if points < 5:
                continue

            indicators: list[str] = []
            if points > 50:
                indicators.append("高评分")
            if num_comments > 20:
                indicators.append(f"高讨论度({num_comments}条评论)")

            card = ResourceCard(
                name=title,
                type="discussion",
                url=url,
                description=f"Points: {points} | Comments: {num_comments}",
                stars=points,
                quality_indicators=indicators,
                recommended_use=f"可用于 {query} 综述的社区讨论章节",
                source="hackernews",
            )
            cards.append(card)

        logger.debug("hackernews: parsed %d discussions", len(cards))
        return cards
