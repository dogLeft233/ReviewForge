"""领域知识探索器——在正式检索前建立领域心智模型

核心类放在这里，协调各检索器 + LLM 分析器，输出 DomainProfile。
"""

import asyncio
import logging
from typing import Any

from src.explorer.analyzer import LLMAnalyzer, TermCard
from src.explorer.models import DomainProfile
from src.retrievers.bocha_explorer import BochaRetriever
from src.retrievers.wikipedia_explorer import WikipediaRetriever
from src.retrievers.hackernews import HackerNewsRetriever

logger = logging.getLogger(__name__)


class DomainExplorer:
    """领域知识探索器

    通过多源检索（Hacker News、博查网页搜索、Wikipedia）探测领域知识，
    再由 LLM 分析提取结构化的领域心智模型（DomainProfile）。

    职责边界：
    - ✅ 提供领域初步探索总结（exploration_summary）
    - ✅ 提供关键术语、核心概念、相关主题、代表性工作
    - ❌ 不生成搜索建议查询词（那是 Planner 的职责）
    - ❌ 不评估覆盖度（那是 Planner.evaluate_coverage 的职责）

    用法示例：
        explorer = DomainExplorer()
        profile = explorer.explore("automatic speech recognition")
        print(profile.to_human_readable())
    """

    def __init__(
        self,
        *,
        use_hackernews: bool = True,
        use_bocha: bool = True,
        use_wikipedia: bool = True,
        max_hn_results: int = 8,
        max_bocha_results: int = 8,
        max_wiki_results: int = 5,
        max_terms: int = 15,
        hn_min_points: int = 5,
        enable_llm: bool = True,
        llm_model: str | None = None,
    ) -> None:
        self.use_hackernews = use_hackernews
        self.use_bocha = use_bocha
        self.use_wikipedia = use_wikipedia
        self.max_hn_results = max_hn_results
        self.max_bocha_results = max_bocha_results
        self.max_wiki_results = max_wiki_results
        self.max_terms = max_terms
        self.hn_min_points = hn_min_points
        self.enable_llm = enable_llm
        self._llm_analyzer: LLMAnalyzer | None = None
        if enable_llm:
            self._llm_analyzer = LLMAnalyzer()
            if llm_model:
                self._llm_analyzer._llm.model = llm_model

    def explore(self, query: str) -> DomainProfile:
        """执行完整领域探索流程"""
        logger.info("DomainExplorer: exploring '%s'", query)
        profile = DomainProfile(original_query=query)

        # ── Step 1: 多源并发检索 ──
        results: dict[str, Any] = {}

        async def fetch_all() -> dict[str, Any]:
            tasks = []
            if self.use_hackernews:
                tasks.append(self._async_hackernews(query))
            if self.use_bocha:
                tasks.append(self._async_bocha(query))
            if self.use_wikipedia:
                tasks.append(self._async_wikipedia(query))
            if not tasks:
                return {}
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            out: dict[str, Any] = {}
            idx = 0
            if self.use_hackernews:
                out["hn"] = _safe_result(gathered, idx)
                idx += 1
            if self.use_bocha:
                out["bocha"] = _safe_result(gathered, idx)
                idx += 1
            if self.use_wikipedia:
                out["wiki"] = _safe_result(gathered, idx)
            return out

        try:
            results = asyncio.run(fetch_all())
        except Exception as e:
            logger.warning("DomainExplorer: async fetch failed: %s", e)
            results = {
                "hn": self._sync_hackernews(query),
                "bocha": self._sync_bocha(query),
                "wiki": self._sync_wikipedia(query),
            }

        # ── Step 2: 处理 HN ──
        profile.hn_results = results.get("hn", [])
        profile.hn_discussions = [
            r.get("name") or r.get("title", "")
            for r in profile.hn_results
            if r.get("name") or r.get("title")
        ]
        if profile.hn_discussions:
            profile.hn_insights = self._summarize_hn(
                profile.hn_discussions, profile.hn_results
            )
            if "hackernews" not in profile.sources_used:
                profile.sources_used.append("hackernews")

        # ── Step 3: 处理 Bocha ──
        profile.bocha_results = results.get("bocha", [])
        if profile.bocha_results and "bocha" not in profile.sources_used:
            profile.sources_used.append("bocha")

        # ── Step 4: 处理 Wikipedia ──
        profile.wiki_results = results.get("wiki", [])
        if profile.wiki_results and "wikipedia" not in profile.sources_used:
            profile.sources_used.append("wikipedia")

        # ── Step 5: LLM 分析 ──
        if self.enable_llm and self._llm_analyzer:
            try:
                analysis = self._llm_analyzer.analyze(
                    bocha_results=profile.bocha_results,
                    hn_discussions=profile.hn_discussions,
                    hn_results=profile.hn_results,
                )
                profile.exploration_summary = analysis.exploration_summary
                profile.key_terms = analysis.key_terms[: self.max_terms]
                profile.core_concepts = analysis.core_concepts
                profile.related_topics = analysis.related_topics
                profile.active_work = analysis.active_work
                profile.domain_overview = analysis.domain_overview
                logger.info(
                    "DomainExplorer: LLM done — %d terms, summary=%s chars",
                    len(profile.key_terms),
                    len(profile.exploration_summary),
                )
            except Exception as e:
                logger.warning(
                    "DomainExplorer: LLM analysis failed: %s — "
                    "falling back to raw",
                    e,
                )
                self._fallback_profile(profile)
        else:
            self._fallback_profile(profile)

        logger.info(
            "DomainExplorer: done — sources=%s, terms=%d",
            profile.sources_used,
            len(profile.key_terms),
        )
        return profile

    # ── 异步检索 ──

    @staticmethod
    async def _async_hackernews(query: str) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, DomainExplorer._sync_hackernews, query
        )

    @staticmethod
    async def _async_bocha(query: str) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, DomainExplorer._sync_bocha, query
        )

    @staticmethod
    async def _async_wikipedia(query: str) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, DomainExplorer._sync_wikipedia, query
        )

    # ── 同步检索 ──

    @staticmethod
    def _sync_hackernews(query: str) -> list[dict]:
        try:
            hn = HackerNewsRetriever()
            resources = hn.search_resources(query, max_results=10)
            return [
                {
                    "name": r.name,
                    "url": r.url,
                    "stars": r.stars,
                    "description": r.description,
                }
                for r in resources
                if (r.stars or 0) >= 5
            ]
        except Exception as e:
            logger.warning("HN search failed: %s", e)
            return []

    @staticmethod
    def _sync_bocha(query: str) -> list[dict]:
        try:
            with BochaRetriever() as r:
                return r.search(query, max_results=8)
        except Exception as e:
            logger.warning("Bocha search failed: %s", e)
            return []

    @staticmethod
    def _sync_wikipedia(query: str) -> list[dict]:
        try:
            with WikipediaRetriever(lang="en", max_results=5) as r:
                return r.search(query, max_results=5)
        except Exception as e:
            logger.warning("Wikipedia search failed: %s", e)
            return []

    # ── 降级处理 ──

    def _fallback_profile(self, profile: DomainProfile) -> None:
        """当 LLM 不可用时，用原始结果填充字段"""
        profile.exploration_summary = (
            f"发现 {len(profile.bocha_results)} 条搜索结果，"
            f"{len(profile.hn_discussions)} 条 HN 讨论"
        )

    @staticmethod
    def _summarize_hn(discussions: list[str], results: list[dict]) -> str:
        if not discussions:
            return ""
        points_map = {
            r.get("name", "") or r.get("title", ""): r.get("stars", 0)
            for r in results
        }
        top = sorted(
            discussions[:5],
            key=lambda t: points_map.get(t, 0),
            reverse=True,
        )
        top_str = "；".join(f'"{t}"' for t in top)
        return (
            f"HN 社区近期对相关主题的讨论集中在：{top_str}。"
            f"这些讨论反映该领域 practitioners 主要关注应用落地和工具选型。"
        )


def _safe_result(gathered: list[Any], idx: int) -> Any:
    """从 gather 结果中安全取值（过滤异常）"""
    if idx >= len(gathered):
        return []
    val = gathered[idx]
    if isinstance(val, Exception):
        logger.warning("async task %d raised: %s", idx, val)
        return []
    return val


# ──────────────────────────────────────────────
# 便捷函数
# ──────────────────────────────────────────────


def explore_domain(query: str, **kwargs: Any) -> DomainProfile:
    """一行调用探索器"""
    explorer = DomainExplorer(**kwargs)
    return explorer.explore(query)
