"""LLM 驱动的领域分析器——将原始检索结果转换为结构化领域知识"""

import json
import logging
from dataclasses import dataclass, field

from src.config import settings
from src.models import PaperCard, ResourceCard

logger = logging.getLogger(__name__)


@dataclass
class LLMClient:
    """轻量 LLM 客户端（SiliconFlow / OpenAI 兼容）"""

    api_key: str = ""
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "Qwen/Qwen3-8B"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = settings.llm_api_key

    def complete(self, system: str, user: str) -> str:
        """调用 LLM 返回文本响应"""
        import httpx

        if not self.api_key:
            raise RuntimeError("LLM_API_KEY not configured")

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            logger.error("LLM API error: %s", e.response.text[:200])
            raise
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise

    def complete_json(self, system: str, user: str) -> dict:
        """调用 LLM 并尝试解析 JSON 响应"""
        text = self.complete(system, user)
        text = text.strip()
        if text.startswith("```"):
            for line in text.splitlines():
                if line.strip() == "json":
                    break
                if line.strip() == "```":
                    continue
                if not line.startswith("```"):
                    text = line
                    break
        start = text.find("{")
        if start < 0:
            start = text.find("[")
        if start >= 0:
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{" or ch == "[":
                    depth += 1
                elif ch == "}" or ch == "]":
                    depth -= 1
                    if depth == 0:
                        text = text[start : i + 1]
                        break
        return json.loads(text)


@dataclass
class TermCard:
    """术语卡片"""

    term: str
    definition: str = ""
    source: str = ""


@dataclass
class DomainAnalysis:
    """LLM 分析结果"""

    exploration_summary: str = ""
    key_terms: list[TermCard] = field(default_factory=list)
    core_concepts: list[str] = field(default_factory=list)
    related_topics: list[str] = field(default_factory=list)
    active_work: str = ""
    domain_overview: str = ""


class LLMAnalyzer:
    """基于大语言模型的领域分析器

    将原始检索结果（Bocha 摘要、Hacker News 讨论）
    交给 LLM 分析，输出结构化的 DomainAnalysis。
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        model: str | None = None,
    ) -> None:
        self._llm = llm_client or LLMClient()
        if model:
            self._llm.model = model

    def analyze(
        self,
        *,
        bocha_results: list[dict] | None = None,
        hn_discussions: list[str] | None = None,
        hn_results: list[dict] | None = None,
        existing_terms: list[TermCard] | None = None,
    ) -> DomainAnalysis:
        """执行 LLM 驱动的领域分析"""
        from src.explorer.prompts import EXPLORATION_PROMPT

        bocha_content = self._format_bocha(bocha_results or [])
        hn_content = self._format_hn(hn_discussions or [], hn_results or [])
        terms_content = self._format_terms(existing_terms or [])

        try:
            raw = self._llm.complete_json(
                system=EXPLORATION_PROMPT.system,
                user=EXPLORATION_PROMPT.user.format(
                    bocha_content=bocha_content,
                    hn_content=hn_content,
                    terms_content=terms_content,
                ),
            )
            return self._parse_analysis(raw)
        except Exception as e:
            logger.warning("LLMAnalyzer: analysis failed: %s — falling back to raw", e)
            return self._fallback(
                bocha_results or [], hn_discussions or [], existing_terms or []
            )

    def _format_bocha(self, results: list[dict]) -> str:
        if not results:
            return "（无博查搜索结果）"
        lines = []
        for r in results[:10]:
            title = r.get("title", "")
            snippet = r.get("description", "") or r.get("summary", "")
            site = r.get("site_name", "")
            if snippet:
                lines.append(f"- [{site}] {title}: {snippet}")
            else:
                lines.append(f"- [{site}] {title}")
        return "\n".join(lines) or "（无博查搜索结果）"

    def _format_hn(
        self, discussions: list[str], results: list[dict]
    ) -> str:
        if not discussions:
            return "（无 Hacker News 讨论）"
        points_map = {r.get("title", ""): r.get("points", 0) for r in results}
        lines = []
        for d in discussions[:10]:
            pts = points_map.get(d, 0)
            lines.append(f"- [{pts} pts] {d}")
        return "\n".join(lines)

    def _format_terms(self, terms: list[TermCard]) -> str:
        if not terms:
            return "（无已发现术语）"
        return "\n".join(
            f"- {t.term}: {t.definition or '(定义未知)'}" for t in terms
        )

    def _parse_analysis(self, raw: dict) -> DomainAnalysis:
        """解析 LLM JSON 输出为 DomainAnalysis"""
        key_terms = []
        for item in raw.get("key_terms", []):
            if isinstance(item, dict):
                key_terms.append(
                    TermCard(
                        term=item.get("term", ""),
                        definition=item.get("definition", ""),
                        source=item.get("source", "llm"),
                    )
                )
            elif isinstance(item, str):
                key_terms.append(TermCard(term=item, source="llm"))

        return DomainAnalysis(
            exploration_summary=raw.get("exploration_summary", ""),
            key_terms=key_terms,
            core_concepts=raw.get("core_concepts", []),
            related_topics=raw.get("related_topics", []),
            active_work=raw.get("active_work", ""),
            domain_overview=raw.get("domain_overview", ""),
        )

    def _fallback(
        self,
        bocha: list[dict],
        hn: list[str],
        terms: list[TermCard],
    ) -> DomainAnalysis:
        """当 LLM 不可用时的降级处理"""
        return DomainAnalysis(
            exploration_summary=(
                f"发现 {len(bocha)} 条搜索结果，{len(hn)} 条 HN 讨论"
            ),
            key_terms=terms,
            core_concepts=[],
            related_topics=[],
            active_work="",
            domain_overview="",
        )
