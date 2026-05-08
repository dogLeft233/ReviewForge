"""Explorer Agent — 领域探索主逻辑"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.explorer.explorer_report import ClassicWork, ExplorerReport
from src.llm import Message
from src.seacher.tools.searcher_tools import web_search

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt 加载
# ─────────────────────────────────────────────────────────────────────────────

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ─────────────────────────────────────────────────────────────────────────────
# 主 Agent
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExplorerAgent:
    """领域探索 Agent — 三阶段搜索 + 综合报告

    用法:
        from src.llm import LLM
        from src.explorer import ExplorerAgent

        llm = LLM(model="Qwen/Qwen3-8B")
        explorer = ExplorerAgent(llm=llm)
        report = explorer.run("automatic speech recognition")
    """

    llm: Any = field(repr=False)
    verbose: bool = False
    max_chars_per_fetch: int = 4000

    def run(self, topic: str) -> ExplorerReport:
        """执行完整三阶段探索"""
        report = ExplorerReport(topic=topic)
        total_queries = 0

        # Stage 1
        logger.info("[Stage 1] 探索领域概况: %s", topic)
        stage1_result = self._run_stage1(topic)
        report.stage1_overview = stage1_result["overview"]
        report.stage1_concepts = stage1_result["concepts"]
        report.stage1_search_results = stage1_result["raw"]
        total_queries += stage1_result["query_count"]

        # Stage 2
        logger.info("[Stage 2] 探索经典工作: %s", topic)
        stage2_result = self._run_stage2(topic)
        report.stage2_classics = stage2_result["classics"]
        report.stage2_timeline = stage2_result["timeline"]
        report.stage2_search_results = stage2_result["raw"]
        total_queries += stage2_result["query_count"]

        # Stage 3
        logger.info("[Stage 3] 探索前沿进展: %s", topic)
        stage3_result = self._run_stage3(topic)
        report.stage3_benchmarks = stage3_result["benchmarks"]
        report.stage3_state_of_art = stage3_result["sota"]
        report.stage3_trends = stage3_result["trends"]
        report.stage3_search_results = stage3_result["raw"]
        total_queries += stage3_result["query_count"]

        # Synthesis
        logger.info("[Synthesis] 生成下游报告")
        report.downstream_report = self._synthesize(topic, report)
        report.total_queries = total_queries

        logger.info("Explorer 完成！共执行 %d 次搜索", total_queries)
        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1
    # ─────────────────────────────────────────────────────────────────────────

    def _run_stage1(self, topic: str) -> dict[str, Any]:
        system = _load_prompt("stage1_system.txt").format(topic=topic)
        user = _load_prompt("stage1_user.txt").format(topic=topic)

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

        reply = self.llm.chat(
            external_prompt="你是一名学术研究员，擅长搜索和分析。",
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
        )

        # 用 src.llm 的 web_search 补充
        search_results, qcount = self._do_search(topic, "overview")
        synthesis_messages = messages + [
            Message(role="assistant", content=reply + "\n\n--- 补充搜索结果 ---\n" + search_results),
            Message(role="user", content="基于以上材料，请整理出结构化的领域概况，包含：定义、核心问题、主流方法、核心概念术语列表（带简明解释）。"),
        ]

        final_reply = self.llm.chat(
            external_prompt="你是一名学术研究员，请整理信息。",
            messages=synthesis_messages,
            temperature=0.2,
            max_tokens=800,
        )

        concepts = self._extract_concepts(final_reply)

        return {
            "overview": final_reply,
            "concepts": concepts,
            "raw": reply + "\n\n" + search_results,
            "query_count": qcount,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2
    # ─────────────────────────────────────────────────────────────────────────

    def _run_stage2(self, topic: str) -> dict[str, Any]:
        system = _load_prompt("stage2_system.txt").format(topic=topic)
        user = _load_prompt("stage2_user.txt").format(topic=topic)

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

        reply = self.llm.chat(
            external_prompt="你是一名学术研究员，擅长搜索和分析。",
            messages=messages,
            temperature=0.3,
            max_tokens=1200,
        )

        search_results, qcount = self._do_search(topic, "classics")

        synthesis_messages = messages + [
            Message(role="assistant", content=reply + "\n\n--- 补充搜索结果 ---\n" + search_results),
            Message(role="user", content="基于以上材料，请整理出该领域的经典论文列表和时间线。每个里程碑工作请列出：标题、作者/年份/会议、核心贡献、对领域的影响。"),
        ]

        final_reply = self.llm.chat(
            external_prompt="你是一名学术研究员，请整理信息。",
            messages=synthesis_messages,
            temperature=0.2,
            max_tokens=1000,
        )

        classics = self._parse_classics(final_reply)
        timeline = self._extract_timeline(final_reply)

        return {
            "classics": classics,
            "timeline": timeline,
            "raw": reply + "\n\n" + search_results,
            "query_count": qcount,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3
    # ─────────────────────────────────────────────────────────────────────────

    def _run_stage3(self, topic: str) -> dict[str, Any]:
        from src.explorer.explorer_report import BenchmarkEntry

        system = _load_prompt("stage3_system.txt").format(topic=topic)
        user = _load_prompt("stage3_user.txt").format(topic=topic)

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

        reply = self.llm.chat(
            external_prompt="你是一名学术研究员，擅长搜索和分析。",
            messages=messages,
            temperature=0.3,
            max_tokens=1200,
        )

        search_results, qcount = self._do_search(topic, "benchmarks")

        synthesis_messages = messages + [
            Message(role="assistant", content=reply + "\n\n--- 补充搜索结果 ---\n" + search_results),
            Message(role="user", content="基于以上材料，请整理出该领域的 Benchmark 和 Leaderboard 列表。每个请列出：名称、URL、评估指标、测试数据集、当前最佳成绩。同时列出近期 SOTA 方法和 3-5 个发展趋势。"),
        ]

        final_reply = self.llm.chat(
            external_prompt="你是一名学术研究员，请整理信息。",
            messages=synthesis_messages,
            temperature=0.2,
            max_tokens=1000,
        )

        benchmarks = self._parse_benchmarks(final_reply)
        sota, trends = self._parse_sota_and_trends(final_reply)

        return {
            "benchmarks": benchmarks,
            "sota": sota,
            "trends": trends,
            "raw": reply + "\n\n" + search_results,
            "query_count": qcount,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Synthesis
    # ─────────────────────────────────────────────────────────────────────────

    def _synthesize(self, topic: str, report: ExplorerReport) -> str:
        stage1 = report.stage1_overview or report.stage1_search_results
        stage2 = report.stage2_timeline or report.stage2_search_results
        if not stage2:
            stage2 = "\n".join(
                f"- {c.title} ({c.year})" for c in report.stage2_classics
            )

        stage3_benchmarks_str = ""
        for b in report.stage3_benchmarks:
            stage3_benchmarks_str += f"\n- **{b.name}** URL: {b.url}" if b.url else f"\n- **{b.name}**"

        system = _load_prompt("synthesis_system.txt").format(topic=topic)
        user = _load_prompt("synthesis_user.txt").format(
            topic=topic,
            stage1=stage1[:3000],
            stage2=stage2[:2000],
            stage3=(report.stage3_state_of_art + stage3_benchmarks_str)[:3000],
        )

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

        reply = self.llm.chat(
            external_prompt="你是一名学术写作助手，请生成结构化报告。",
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )

        return reply

    # ─────────────────────────────────────────────────────────────────────────
    # 搜索辅助
    # ─────────────────────────────────────────────────────────────────────────

    def _do_search(self, topic: str, query_type: str) -> tuple[str, int]:
        """使用 src.llm.web_search 执行搜索，返回 (formatted_results, query_count)"""
        queries = {
            "overview": [f"{topic} overview introduction", f"{topic} core concepts"],
            "classics": [f"{topic} history evolution classic papers", f"{topic} seminal work"],
            "benchmarks": [f"{topic} leaderboard benchmark", f"{topic} state of the art 2024"],
        }
        q_list = queries.get(query_type, [topic])

        sections = []
        count = 0
        for q in q_list[:2]:
            results = web_search(q, count=8)
            count += 1
            if results and not _is_error(results):
                sections.append(f"### 搜索: {q}\n{self._format_results(results)}")

        return "\n\n".join(sections), count

    def _format_results(self, results: list) -> str:
        lines = []
        for r in results[:8]:
            title = r.get("title", "")
            url = r.get("url", "")
            desc = r.get("description", "")[:200]
            lines.append(f"- [{title}]({url})\n  {desc}")
        return "\n".join(lines) if lines else "（无结果）"

    # ─────────────────────────────────────────────────────────────────────────
    # 解析辅助
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_concepts(self, text: str) -> list[str]:
        import re
        concepts = []
        for line in text.split("\n"):
            m = re.match(r"^\s*[-*•]\s*(.+?)(?:：|:)\s*(.+)", line)
            if m:
                concepts.append(f"{m.group(1).strip()}: {m.group(2).strip()}")
            elif re.match(r"^\s*[-*•]\s*[\u4e00-\u9fa5a-zA-Z]", line):
                concepts.append(line.lstrip("-*• ").strip())
        return concepts[:10]

    def _parse_classics(self, text: str) -> list[ClassicWork]:
        import re
        classics: list[ClassicWork] = []
        for line in text.split("\n"):
            m = re.search(
                r"\*\*(.+?)\*\*.*?\((\d{4})\)|(.+?)\s*\((\d{4})\)|(\d{4}).*?[-–]\s*(.+)",
                line,
            )
            if m:
                title = m.group(1) or m.group(3) or m.group(6)
                year_str = m.group(2) or m.group(4)
                year = int(year_str) if year_str else 0
                if title and year:
                    classics.append(ClassicWork(title=title.strip(), year=year))
        return classics[:20]

    def _extract_timeline(self, text: str) -> str:
        lines = text.split("\n")
        timeline_lines = [
            l for l in lines
            if any(k in l.lower() for k in ["时间线", "timeline", "阶段", "20", "201", "202"])
        ]
        return "\n".join(timeline_lines[:30])

    def _parse_benchmarks(self, text: str) -> list:
        from src.explorer.explorer_report import BenchmarkEntry
        import re
        benchmarks = []
        for line in text.split("\n"):
            url_match = re.search(r"https?://[^\s\)（）]+", line)
            if url_match:
                url = re.sub(r"[),.;，。]+$", "", url_match.group(0))
                name = line.split("[")[1].split("]")[0] if "[" in line else line[:60]
                benchmarks.append(BenchmarkEntry(name=name, url=url, description=line[:200]))
        return benchmarks[:10]

    def _parse_sota_and_trends(self, text: str) -> tuple[str, list[str]]:
        import re
        trends = []
        sota_lines = []
        for line in text.split("\n"):
            if any(k in line.lower() for k in ["sota", "最佳", "state-of-the-art", "current best"]):
                sota_lines.append(line.strip())
            if any(k in line for k in ["趋势", "trend", "方向", "direction"]):
                m = re.match(r"^\s*[-*\d\.]+\s*(.+)", line)
                if m:
                    trends.append(m.group(1).strip())
        return "\n".join(sota_lines[:5]), trends[:5]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _is_error(results) -> bool:
    """判断 web_search 返回是否表示错误"""
    if not isinstance(results, list):
        return True
    if results and isinstance(results[0], dict) and "error" in results[0]:
        return True
    return False
