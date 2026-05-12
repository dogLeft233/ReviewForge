"""Writer Agent — 基于探索结果撰写综述论文（两阶段：规划 → 各章展开）"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.explorer.explorer_report import ExplorerReport
from src.llm import LLM, Message

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WriterConfig:
    """Writer 配置"""

    model: str = "Qwen/Qwen3-8B"
    temperature: float = 0.2
    max_tokens: int = 4096
    body_max_tokens: int = 2000
    timeout_seconds: float = 300.0


@dataclass
class WriterReport:
    """Writer 输出报告

    各阶段共享同一个 report 实例，后写章节可以看到先写章节的输出，
    保证上下文连贯性。
    """

    # ── 规划阶段 ─────────────────────────────────────────
    plan: str = ""  # 全局写作计划（分类体系、各章核心要点）
    title_candidates: list[str] = field(default_factory=list)  # 候选标题（供用户选）

    # ── 各章节（按写作顺序）───────────────────────────────
    title: str = ""
    introduction: str = ""
    body: str = ""
    conclusion: str = ""
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    references: str = ""
    benchmarks: str = ""
    trends: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# 主 Agent
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WriterAgent:
    """综述写作 Agent（两阶段写作，上下文传递）

    接受来自 ExplorerAgent 的三阶段探索结果，
    按照《计算机学报》综述规范撰写完整综述论文。

    写作流程：
        阶段 0（规划） → 生成全局计划（分类体系 + 候选标题 + 各章要点）
        阶段 1（各章） → 按规划展开，各章可引用前阶段输出

    用法:
        from src.llm import LLM
        from src.writer import WriterAgent, WriterConfig

        llm = LLM(api_key="sk-...", model="Qwen/Qwen3-8B")
        config = WriterConfig(max_tokens=4096)
        writer = WriterAgent(llm=llm, config=config)

        report = writer.write(
            topic="LoRA large language model fine-tuning",
            explorer_report=explorer_report,
        )
    """

    llm: LLM = field(repr=False)
    config: WriterConfig = field(default_factory=WriterConfig)

    # ── 入口 ───────────────────────────────────────────────────────────────

    def write(self, topic: str, explorer_report: ExplorerReport) -> WriterReport:
        """执行完整综述写作流程（两阶段：规划 → 各章展开）"""
        report = WriterReport()

        # 阶段 0：规划
        plan_result = self._planning(topic, explorer_report)
        report.plan = plan_result["plan"]
        report.title_candidates = plan_result["title_candidates"]

        # 阶段 1：各章写作（上下文传递）
        # 先设置 plan（所有章节的 format 模板都依赖它）
        report.plan = plan_result["plan"]
        self._write_chapters(topic, explorer_report, report)

        logger.info("Writer 完成！")
        return report

    # ─────────────────────────────────────────────────────────────────────────
    # 阶段 0：规划（Plan）
    # ─────────────────────────────────────────────────────────────────────────

    def _planning(self, topic: str, er: ExplorerReport) -> dict[str, Any]:
        """生成全局写作计划

        规划内容：
        - 候选标题（3个）
        - 分类体系（Taxonomy）—— 将成为正文章节的骨架
        - 各章节核心要点（引言切入点、正文分类依据、结论核心洞察）
        - 关键词方向
        """
        system = _load_prompt("planning_system.txt").format(
            topic=topic,
            overview=(er.stage1_overview or er.stage1_search_results or "")[:2000],
            classics="\n".join(
                f"- **{c.title}** ({c.year}): {c.key_idea or '请补充核心贡献'}"
                for c in er.stage2_classics[:10]
            ),
            sota=(er.stage3_state_of_art or "")[:1000],
            trends=(
                er.stage3_trends
                if isinstance(er.stage3_trends, str)
                else "\n".join(er.stage3_trends or [])
            )[:500],
        )
        user = _load_prompt("planning_user.txt").format(topic=topic)

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

        reply = self.llm.chat(
            external_prompt="你是一名学术写作规划专家，请生成结构化的全局写作计划。",
            messages=messages,
            temperature=0.4,
            max_tokens=800,
        )

        # 解析候选标题
        candidates = []
        for line in reply.split("\n"):
            line = line.strip()
            if line and len(line) > 2:
                # 匹配 "1. 标题" / "1、标题" / "1: 标题" 格式
                if line[0] in "123456789" and "." in line[:3]:
                    parts = line.split(".", 1)
                elif line[0] in "123456789" and "、" in line[:3]:
                    parts = line.split("、", 1)
                else:
                    continue
                title = parts[1].strip().strip("、：: ")
                if title and len(title) < 30:
                    candidates.append(title)

        return {
            "plan": reply,
            "title_candidates": candidates[:3],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 阶段 1：各章写作（上下文传递）
    # ─────────────────────────────────────────────────────────────────────────

    def _write_chapters(
        self,
        topic: str,
        er: ExplorerReport,
        report: WriterReport,
    ) -> None:
        """并发写作各章节（依赖关系：引言∥标题 → 正文 → 结论∥参考文献∥关键词 → 摘要）"""

        # ── 阶段 A：标题 + 引言（无依赖，并发）────────────────
        with ThreadPoolExecutor(max_workers=self.llm._cfg.max_concurrency) as pool:
            f_title = pool.submit(self._write_title, topic, report)
            f_intro = pool.submit(self._write_introduction, topic, er, report)
            report.title = f_title.result()
            report.introduction = f_intro.result()

        # ── 阶段 B：正文（依赖标题+引言）──────────────────────
        report.body = self._write_body(topic, er, report)

        # ── 阶段 C：结论 + 参考文献 + 关键词（均依赖正文，并发）──
        with ThreadPoolExecutor(max_workers=self.llm._cfg.max_concurrency) as pool:
            f_conc = pool.submit(self._write_conclusion, topic, er, report)
            f_ref = pool.submit(self._collect_references, topic, er, report)
            f_kw = pool.submit(self._extract_keywords, topic, er, report)
            report.conclusion = f_conc.result()
            report.references = f_ref.result()
            report.keywords = f_kw.result()

        # ── 阶段 D：摘要（依赖结论+正文）──────────────────────
        report.abstract = self._write_abstract(topic, er, report)

    # ─────────────────────────────────────────────────────────────────────────
    # 各章节实现
    # ─────────────────────────────────────────────────────────────────────────

    def _write_title(self, topic: str, report: WriterReport) -> str:
        system = _load_prompt("title_system.txt").format(topic=topic, plan=report.plan[:2000])
        user = _load_prompt("title_user.txt").format(
            topic=topic,
            plan=report.plan[:2000],  # 规划作为参考上下文
        )

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

        return self.llm.chat(
            external_prompt="你是一名学术写作专家，精通中文期刊标题命名。",
            messages=messages,
            temperature=0.3,
            max_tokens=600,
        )

    def _write_introduction(
        self, topic: str, er: ExplorerReport, report: WriterReport,
    ) -> str:
        system = _load_prompt("introduction_system.txt").format(
            topic=topic,
            plan=report.plan[:2000],  # 引入规划约束
            title=report.title,
            overview=(er.stage1_overview or er.stage1_search_results or "")[:3000],
            classics=(er.stage2_timeline or "\n".join(
                f"- {c.title} ({c.year})" for c in (er.stage2_classics or [])
            ))[:2000],
        )
        user = _load_prompt("introduction_user.txt").format(
            topic=topic,
            title=report.title,  # 标题作为已确定信息传入
        )

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

        return self.llm.chat(
            external_prompt="你是一名学术写作专家，请撰写规范的引言章节。",
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=800,
        )

    def _write_body(
        self, topic: str, er: ExplorerReport, report: WriterReport,
    ) -> str:
        system = _load_prompt("body_system.txt").format(
            topic=topic,
            plan=report.plan[:2000],  # 分类体系作为骨架约束
            title=report.title,
            introduction_preview=report.introduction[:500],  # 承上启下
            overview=(er.stage1_overview or er.stage1_search_results or "")[:3000],
            classics="\n".join(
                f"- **{c.title}** ({c.year}): 请补充核心贡献"
                for c in er.stage2_classics[:15]
            ),
            sota=(er.stage3_state_of_art or "")[:2000],
            benchmarks="\n".join(
                f"- {b.name}" + (f" URL: {b.url}" if b.url else "")
                for b in er.stage3_benchmarks
            )[:1500],
            trends=(
                er.stage3_trends
                if isinstance(er.stage3_trends, str)
                else "\n".join(er.stage3_trends or [])
            )[:1000],
        )
        user = _load_prompt("body_user.txt").format(
            topic=topic,
            overview=(er.stage1_overview or er.stage1_search_results or "")[:3000],
            classics="\n".join(
                f"- **{c.title}** ({c.year}): 请补充核心贡献"
                for c in er.stage2_classics[:15]
            ),
            sota=(er.stage3_state_of_art or "")[:2000],
            benchmarks=("\n".join(
                f"- {b.name}" + (f" URL: {b.url}" if b.url else "")
                for b in er.stage3_benchmarks
            ))[:1500],
            trends=(
                er.stage3_trends
                if isinstance(er.stage3_trends, str)
                else "\n".join(er.stage3_trends or [])
            )[:5],
            title=report.title,
            introduction_summary=report.introduction[:400],
        )

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

        return self.llm.chat(
            external_prompt="你是一名学术写作专家，请撰写综述的核心章节。",
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.body_max_tokens,
        )

    def _write_conclusion(
        self, topic: str, er: ExplorerReport, report: WriterReport,
    ) -> str:
        system = _load_prompt("conclusion_system.txt").format(
            topic=topic,
            plan=report.plan[:1500],
            title=report.title,
            introduction_summary=report.introduction[:300],
            body_preview=report.body[:500],
            body_summary=report.body[:500],
            trends=(
                er.stage3_trends
                if isinstance(er.stage3_trends, str)
                else "\n".join(er.stage3_trends or [])
            )[:1000],
        )
        user = _load_prompt("conclusion_user.txt").format(
            topic=topic,
            title=report.title,
            body_summary=report.body[:500],
            trends=(
                er.stage3_trends
                if isinstance(er.stage3_trends, str)
                else "\n".join(er.stage3_trends or [])
            )[:1000],
        )

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

        return self.llm.chat(
            external_prompt="你是一名学术写作专家，请撰写结论章节。",
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=600,
        )

    def _collect_references(
        self, topic: str, er: ExplorerReport, report: WriterReport,
    ) -> str:
        classics_text = "\n".join(
            f"- {c.title} ({c.year})"
            for c in er.stage2_classics[:20]
        )

        system = _load_prompt("references_system.txt").format(
            classics=classics_text,
        )

        user = _load_prompt("references_user.txt").format(
            topic=topic,
            classics=classics_text or "（无经典论文数据）",
        )

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

        return self.llm.chat(
            external_prompt="你是一名学术写作专家，请按《计算机学报》格式整理参考文献。",
            messages=messages,
            temperature=0.2,
            max_tokens=300,
        )

    def _write_abstract(
        self, topic: str, er: ExplorerReport, report: WriterReport,
    ) -> str:
        system = _load_prompt("abstract_system.txt").format(
            topic=topic,
            title=report.title,
            body_preview=report.body[:500],
            conclusion_preview=report.conclusion[:300],
            introduction=(er.stage1_overview or er.stage1_search_results or "")[:2000],
            classics="\n".join(
                f"- {c.title} ({c.year})" for c in er.stage2_classics[:5]
            ),
            sota=(er.stage3_state_of_art or "")[:1000],
        )
        user = _load_prompt("abstract_user.txt").format(
            topic=topic,
            title=report.title,
            introduction=(er.stage1_overview or er.stage1_search_results or "")[:2000],
            classics="\n".join(
                f"- {c.title} ({c.year})" for c in er.stage2_classics[:5]
            ),
            sota=(er.stage3_state_of_art or "")[:1000],
            body_summary=report.body[:500],
            conclusion_preview=report.conclusion[:300],
        )

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

        return self.llm.chat(
            external_prompt="你是一名学术写作专家，请按《计算机学报》摘要规范撰写。",
            messages=messages,
            temperature=0.2,
            max_tokens=60,
        )

    def _extract_keywords(
        self, topic: str, er: ExplorerReport, report: WriterReport,
    ) -> list[str]:
        concepts_text = "\n".join((er.stage1_concepts or [])[:10])
        system = _load_prompt("keywords_system.txt").format(
            concepts=concepts_text,
        )
        user = _load_prompt("keywords_user.txt").format(
            topic=topic,
        )

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

        reply = self.llm.chat(
            external_prompt="你是一名学术写作专家，请提取5-7个关键词。",
            messages=messages,
            temperature=0.2,
            max_tokens=50,
        )

        keywords = []
        for line in reply.split("\n"):
            line = line.strip().strip("-*、.。")
            if line and len(line) < 15:
                keywords.append(line)
        return keywords[:7]
