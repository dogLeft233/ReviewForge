"""Google Serper API 检索器 (serper.dev)

提供基于 Google 搜索的实时互联网内容检索。
通过有机搜索结果自动分类：学术/技术类映射为 PaperCard，工具/资源类映射为 ResourceCard。
"""

from datetime import datetime
from urllib.parse import urlparse

from src.config import settings
from src.models import PaperCard, ResourceCard
from src.retrievers.base import BaseRetriever

import logging

logger = logging.getLogger(__name__)

SERPER_ENDPOINT = "https://google.serper.dev/search"

# 搜索内容类型提示，扩展自然语言查询
# serper 支持 type 参数触发专门的搜索体验
SEARCH_TYPES = ("search", "news", "images", "places", "patents")


class SerperRetriever(BaseRetriever):
    """Google Serper API 检索器（需要 SERPER_API_KEY，免费层 2500 次/月）"""

    name = "serper"

    def __init__(self) -> None:
        super().__init__()
        self._api_key = settings.serper_api_key
        if not self._api_key:
            logger.warning(
                "SERPER_API_KEY not set — Serper retriever disabled. "
                "Register at https://serper.dev to get a free API key (2500 free queries/month)."
            )

    # ── 公开接口 ──

    def search(
        self, query: str, max_results: int | None = None
    ) -> list[PaperCard]:
        """Google 搜索 → 提取学术/技术类结果映射为 PaperCard"""
        if not self._api_key:
            logger.warning("serper: skipped — no API key")
            return []

        data = self._serper_search(query)
        results = data.get("organic", [])
        if max_results:
            results = results[:max_results]

        cards: list[PaperCard] = []
        for item in results:
            card = self._organic_to_paper(item, query)
            if card:
                cards.append(card)

        logger.info(
            "serper search '%s': %d results → %d paper cards",
            query, len(results), len(cards),
        )
        return cards

    def search_resources(
        self, query: str, max_results: int | None = None
    ) -> list[ResourceCard]:
        """Google 搜索 → 提取工具/资源类结果映射为 ResourceCard"""
        if not self._api_key:
            logger.warning("serper: skipped — no API key")
            return []

        data = self._serper_search(query)
        results = data.get("organic", [])
        if max_results:
            results = results[:max_results]

        cards: list[ResourceCard] = []
        for item in results:
            card = self._organic_to_resource(item, query)
            if card:
                cards.append(card)

        logger.info(
            "serper resources '%s': %d results → %d resource cards",
            query, len(results), len(cards),
        )
        return cards

    # ── Serper API 调用 ──

    def _serper_search(self, query: str) -> dict:
        """执行 Serper API 搜索，返回完整响应"""
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        payload: dict = {
            "q": query,
            "gl": "us",
            "hl": "en",
        }
        # 尝试自动检测查询中的 type 指示（一般搜索非必要）
        logger.debug("serper: searching %r", query)

        resp = self._client.post(SERPER_ENDPOINT, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

    # ── 映射逻辑 ──

    @staticmethod
    def _organic_to_paper(item: dict, query: str) -> PaperCard | None:
        """将有机搜索结果映射为 PaperCard

        判断标准：结果中含有学术/技术/深度内容特征
        """
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        date_str = item.get("date", "")
        attributes = item.get("attributes", {}) or {}

        if not title or not link:
            return None

        # 提取年份（从日期或链接中的数字）
        year = 0
        if date_str:
            for y in range(2020, 2027):
                if str(y) in date_str:
                    year = y
                    break
        if not year:
            for y in range(2020, 2027):
                if str(y) in link:
                    year = y
                    break

        # 域名提取（作为 venue 的替代标记）
        domain = urlparse(link).netloc

        # 构造 PaperCard （标题 + 摘要模式）
        card = PaperCard(
            title=title,
            year=year,
            venue=domain,
            abstract=snippet,
            url=link,
            source="serper",
            retrieved_at=datetime.now().isoformat(),
        )

        # 填充建议用法
        card.recommended_use = _recommend_use(title, snippet, link)

        return card

    @staticmethod
    def _organic_to_resource(item: dict, query: str) -> ResourceCard | None:
        """将有机搜索结果映射为 ResourceCard

        当结果链接指向工具/平台/框架/数据集类站点时返回 ResourceCard
        """
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")

        if not title or not link:
            return None

        domain = urlparse(link).netloc

        # 资源类型检测
        rtype = _detect_resource_type(domain, title, snippet)

        card = ResourceCard(
            name=title,
            type=rtype,
            url=link,
            description=snippet,
            source="serper",
            retrieved_at=datetime.now().isoformat(),
        )

        indicators: list[str] = []
        if domain:
            indicators.append(f"来源: {domain}")
        if "open source" in snippet.lower():
            indicators.append("开源")
        if "free" in snippet.lower():
            indicators.append("免费")
        if date_str := item.get("date", ""):
            indicators.append(f"更新: {date_str}")
        card.quality_indicators = indicators

        card.recommended_use = _recommend_use(title, snippet, link)

        return card


# ── 辅助函数 ──

_RESOURCE_DOMAINS = {
    "github.com": "github_repo",
    "gitlab.com": "github_repo",
    "pypi.org": "tool",
    "npmjs.com": "tool",
    "hub.docker.com": "tool",
    "kaggle.com": "dataset",
    "huggingface.co": "model",
    "paperswithcode.com": "benchmark",
    "colab.research.google.com": "tool",
    "mlflow.org": "tool",
    "wandb.ai": "tool",
    "docker.com": "tool",
}

_TOOL_KEYWORDS = [
    "framework", "library", "tool", "platform", "sdk", "api",
    "toolkit", "sdk", "engine", "pipeline",
]

_DATASET_KEYWORDS = [
    "dataset", "benchmark", "corpus", "collection",
]

_BENCHMARK_KEYWORDS = [
    "benchmark", "leaderboard", "evaluation", "results",
]


def _detect_resource_type(domain: str, title: str, snippet: str) -> str:
    """启发式检测资源类型"""
    text = f"{title} {snippet}".lower()

    # 域名匹配优先
    for d, rtype in _RESOURCE_DOMAINS.items():
        if d in domain:
            return rtype

    # 关键词匹配
    if any(kw in text for kw in _BENCHMARK_KEYWORDS):
        return "benchmark"
    if any(kw in text for kw in _DATASET_KEYWORDS):
        return "dataset"
    if any(kw in text for kw in _TOOL_KEYWORDS):
        return "tool"

    return "other"


def _recommend_use(title: str, snippet: str, link: str) -> str:
    """生成使用建议标签"""
    text = f"{title} {snippet}".lower()

    if any(kw in text for kw in ("survey", "review", "overview")):
        return "可用于综述背景章节"
    if any(kw in text for kw in ("tutorial", "guide", "getting started")):
        return "可用于入门背景介绍"
    if "blog" in text or "blog" in link.lower():
        return "可作为应用案例参考"
    if "arxiv" in text or "arxiv" in link.lower():
        return "论文引用来源"

    return "可作为补充参考材料"
