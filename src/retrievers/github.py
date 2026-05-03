"""GitHub 开源资源检索器"""

from urllib.parse import quote

from src.config import settings
from src.models import PaperCard, ResourceCard
from src.retrievers.base import BaseRetriever

import logging

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


class GithubRetriever(BaseRetriever):
    """GitHub REST API 检索器（Token 可选，推荐提供以提升速率限制）"""

    name = "github"

    def __init__(self) -> None:
        super().__init__()
        self._token = settings.github_token
        if not self._token:
            logger.warning(
                "GITHUB_TOKEN not set — rate limited to 60 req/hr (unauthenticated)"
            )

    def search(
        self, query: str, max_results: int | None = None
    ) -> list[PaperCard]:
        """GitHub 不直接返回论文，返回空列表"""
        return []

    def search_resources(
        self, query: str, max_results: int | None = None
    ) -> list[ResourceCard]:
        """按关键词搜索 GitHub 仓库"""
        if max_results is None:
            max_results = 20

        headers = {"Accept": "application/vnd.github.v3+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        q = quote(f"{query} in:name,description,readme")
        url = f"{GITHUB_API}/search/repositories?q={q}&per_page={max_results}&sort=stars"

        logger.info("github: searching repos '%s' (max=%d)", query, max_results)
        resp = self._get(url, headers=headers)
        data = resp.json()
        return self._parse_resources(data, query)

    def search_official_implementation(
        self, paper_title: str
    ) -> ResourceCard | None:
        """搜索某篇论文的官方实现"""
        query = f"{paper_title} official implementation"
        resources = self.search_resources(query, max_results=5)
        if resources:
            resources[0].quality_indicators.append("可能为官方实现")
            return resources[0]
        return None

    def search_by_topic(self, topic: str, max_results: int = 10) -> list[ResourceCard]:
        """按 GitHub Topic 搜索"""
        headers = {"Accept": "application/vnd.github.mercy-preview+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        url = f"{GITHUB_API}/search/repositories?q=topic:{quote(topic)}&per_page={max_results}&sort=stars"

        resp = self._get(url, headers=headers)
        data = resp.json()
        return self._parse_resources(data, topic)

    # ── 解析 ──

    @staticmethod
    def _parse_resources(data: dict, query: str) -> list[ResourceCard]:
        cards: list[ResourceCard] = []
        for item in data.get("items", []):
            indicators: list[str] = []
            if item.get("has_pages"):
                indicators.append("有GitHub Pages")
            if item.get("license"):
                indicators.append(f"License: {item['license']['spdx_id']}")
            if item.get("language"):
                indicators.append(f"语言: {item['language']}")

            description = item.get("description") or ""
            top_topics = ", ".join(item.get("topics", [])[:5])

            card = ResourceCard(
                name=item.get("full_name") or item.get("name", ""),
                type="github_repo",
                url=item.get("html_url", ""),
                stars=item.get("stargazers_count", 0),
                description=f"{description} | Topics: {top_topics}",
                owner=item.get("owner", {}).get("login", ""),
                quality_indicators=indicators,
                recommended_use=f"可用于 {query} 相关综述的资源章节",
                source="github",
            )
            cards.append(card)

        logger.debug("github: parsed %d resources", len(cards))
        return cards
