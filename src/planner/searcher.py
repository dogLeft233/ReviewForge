"""WebSearcher — 联网预搜索模块

在 Planner 调用 LLM 规划之前，先搜索真实世界中该主题的现状，
让 LLM 基于实际内容生成子方向，减少幻觉。

当前支持：博查搜索（Bocha Search）— 基于 Node.js 脚本
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

# 博查脚本路径（相对于 workspace/skills）
_BOCHA_SCRIPT = (
    Path.home()
    / ".openclaw"
    / "workspace"
    / "skills"
    / "bocha-search"
    / "scripts"
    / "search.js"
)


@dataclass(slots=True)
class SearchResult:
    """单条搜索结果"""

    title: str = ""
    url: str = ""
    description: str = ""
    summary: str = ""
    site_name: str = ""
    published_date: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SearchResult":
        return cls(
            title=d.get("title", ""),
            url=d.get("url", ""),
            description=d.get("description", ""),
            summary=d.get("summary", "") or d.get("description", ""),
            site_name=d.get("siteName", ""),
            published_date=d.get("publishedDate", "") or "",
        )


@dataclass(slots=True)
class SearchContext:
    """预搜索上下文——注入到 LLM prompt 中使用"""

    topic: str
    results: list[SearchResult] = field(default_factory=list)
    total_estimated: int = 0
    searched_at: float = 0.0

    def is_empty(self) -> bool:
        return len(self.results) == 0

    def to_prompt_block(self) -> str:
        """格式化为可注入 prompt 的文本块"""
        if self.is_empty():
            return "(无搜索结果)"
        lines: list[str] = [
            f"以下是关于「{self.topic}」的联网搜索结果，",
            f"共约 {self.total_estimated} 条结果，取前 {len(self.results)} 条：",
            "",
        ]
        for i, r in enumerate(self.results, 1):
            lines.append(f"  [{i}] {r.title}")
            if r.published_date:
                lines.append(f"      日期: {r.published_date}")
            lines.append(f"      来源: {r.site_name or r.url}")
            lines.append(f"      {r.description}")
            lines.append("")
        return "\n".join(lines)


class WebSearcher:
    """联网搜索器——通过博查 API 搜索主题相关内容"""

    def __init__(
        self,
        script_path: str | Path | None = None,
        max_results: int = 10,
        freshness: str = "noLimit",
    ) -> None:
        self.script_path = Path(script_path or _BOCHA_SCRIPT)
        self.max_results = max_results
        self.freshness = freshness

    def search(self, query: str, count: int | None = None) -> SearchContext:
        """执行一次搜索，返回结构化结果

        Args:
            query: 搜索关键词
            count: 最大结果数（默认 self.max_results）

        Returns:
            SearchContext 对象
        """
        count = count or self.max_results

        if not self.script_path.exists():
            logger.warning(
                "Bocha search script not found at %s",
                self.script_path,
            )
            return SearchContext(topic=query)

        try:
            result = subprocess.run(
                [
                    "node",
                    str(self.script_path),
                    query,
                    "--count", str(min(count, 50)),
                    "--summary",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except FileNotFoundError:
            logger.warning("Node.js not found, cannot search")
            return SearchContext(topic=query)
        except subprocess.TimeoutExpired:
            logger.warning("Bocha search timed out for query: %s", query)
            return SearchContext(topic=query)

        if result.returncode != 0:
            logger.warning(
                "Bocha search failed (code=%d): %s",
                result.returncode,
                result.stderr.strip(),
            )
            return SearchContext(topic=query)

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning("Bocha search returned invalid JSON")
            return SearchContext(topic=query)

        if data.get("type") == "error":
            logger.warning("Bocha search API error: %s", data.get("message"))
            return SearchContext(topic=query)

        raw_results = data.get("results", [])
        results = [SearchResult.from_dict(r) for r in raw_results]

        return SearchContext(
            topic=query,
            results=results,
            total_estimated=data.get("totalResults", len(results)),
            searched_at=time.time(),
        )

    def search_topic(
        self,
        topic: str,
        additional_queries: list[str] | None = None,
    ) -> SearchContext:
        """为主题执行一次或多次搜索，合并结果

        先用原始主题搜索，如果指定了附加查询，也一并搜索并合并。

        Args:
            topic: 原始主题
            additional_queries: 附加搜索（如 "survey"、"最新进展"）

        Returns:
            合并后的 SearchContext
        """
        # 主搜索
        ctx = self.search(topic)

        # 附加搜索
        seen_urls = {r.url for r in ctx.results}
        for aq in (additional_queries or []):
            extra = self.search(f"{topic} {aq}")
            for r in extra.results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    ctx.results.append(r)

        # 去重后截断
        if len(ctx.results) > 50:
            ctx.results = ctx.results[:50]

        ctx.total_estimated = max(ctx.total_estimated, len(ctx.results))
        return ctx
