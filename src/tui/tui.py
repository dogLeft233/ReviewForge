"""
ReviewForge TUI — 交互式终端界面

使用 Rich 构建，特点：
- 实时日志面板（所有 logger 输出同步显示）
- 分步骤进度指示
- 清晰展示 plan → report → outline 全流程

工作流：
  [1] 输入主题
  [2] plan_retrieval     → 显示检索规划
  [3] execute_plan       → 显示论文/资源统计
  [4] curate             → 显示分类洞察
  [5] generate_outline   → 显示细纲（含各节点覆盖状态）
  [6] 补搜（如需要）      → 显示新增论文
  [7] 展示最终结果

用法：
  python -m src.tui              # 交互模式
  python -m src.tui --topic "xxx" # 单次运行后退出
  python -m src.tui --debug       # 开启 debug 日志
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


# ── Console 全局实例 ──────────────────────────────────────────────
console = Console()


# ── 步骤定义 ──────────────────────────────────────────────────────
class Step(Enum):
    IDLE = "空闲"
    PLAN = "规划检索"
    EXECUTE = "执行检索"
    CURATE = "策展分析"
    OUTLINE = "生成细纲"
    SUPPLEMENT = "补搜"
    DONE = "完成"


# ── App 状态 ──────────────────────────────────────────────────────
@dataclass
class SourceResult:
    """单次检索源的执行结果"""
    source: str           # 源名称：arxiv, github, etc.
    query: str            # 实际执行的查询
    papers_count: int     # 论文数量
    resources_count: int  # 资源数量
    error: str | None = None  # 错误信息（如有）

    def to_display(self) -> str:
        if self.error:
            return f"❌ {self.source}: {self.error}"
        paper_str = f"{self.papers_count}篇论文" if self.papers_count > 0 else ""
        resource_str = f"{self.resources_count}个资源" if self.resources_count > 0 else ""
        parts = [p for p in [paper_str, resource_str] if p]
        detail = ", ".join(parts) if parts else "0结果"
        return f"✅ 从 {self.source} 找到了 {detail}"


class AppState:
    step: Step = Step.IDLE
    topic: str = ""
    step_detail: str = ""
    progress: float = 0.0  # 0.0 ~ 1.0

    # 各步骤产出
    plan_text: str = ""
    report_lines: list[str] = []
    outline_text: str = ""
    logs: list[str] = []
    
    # 新增：按源的检索结果
    source_results: list[SourceResult] = field(default_factory=list)
    # 补搜结果
    supplement_results: list[SourceResult] = field(default_factory=list)

    def reset(self) -> None:
        self.step = Step.IDLE
        self.topic = ""
        self.step_detail = ""
        self.progress = 0.0
        self.plan_text = ""
        self.report_lines = []
        self.outline_text = ""
        self.logs = []
        self.source_results = []
        self.supplement_results = []


state = AppState()


# ── 日志重定向到 TUI ───────────────────────────────────────────────
class UITailHandler(logging.Handler):
    """把 log 记录到 state.logs（供日志面板显示）"""

    def emit(self, record: logging.LogRecord) -> None:
        level = record.levelname
        msg = self.format(record)
        # 截断太长消息
        if len(msg) > 200:
            msg = msg[:200] + "…"
        import time
        t = time.strftime("%H:%M:%S", time.localtime(record.created))
        state.logs.append(f"[{t}] [{level}] {msg}")
        # 只保留最近 500 条
        if len(state.logs) > 500:
            state.logs = state.logs[-500:]


tail_handler = UITailHandler()
tail_handler.setFormatter(logging.Formatter("%(message)s"))


# ── 布局构建 ───────────────────────────────────────────────────────
def make_layout() -> Layout:
    """构建三区布局：Header / Main / LogPanel"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="main"),
        Layout(name="logs", size=10),
    )
    return layout


def build_header() -> Panel:
    """顶部状态栏"""
    step_icons = {
        Step.IDLE: "⏸",
        Step.PLAN: "🔍",
        Step.EXECUTE: "⚡",
        Step.CURATE: "🧠",
        Step.OUTLINE: "📋",
        Step.SUPPLEMENT: "🔄",
        Step.DONE: "✅",
    }
    icon = step_icons.get(state.step, "•")
    title = f"{icon} [{state.step.value}]"

    if state.topic:
        title += f" | 主题：{state.topic}"

    bar = _progress_bar(state.progress)

    content = Text.assemble(
        f"[bold cyan]ReviewForge TUI[/bold cyan]   {title}\n",
        bar,
    )
    return Panel(content, border_style="cyan", title="[状态栏]")


def _progress_bar(pct: float, width: int = 60) -> str:
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[cyan]{bar}[/cyan] {pct * 100:.0f}%"


def build_main() -> Group:
    """主区域：根据当前步骤显示不同面板"""
    panels = []

    # 步骤说明
    step_descriptions = {
        Step.IDLE: "输入主题开始",
        Step.PLAN: "LLM 分析主题，生成多组检索查询和策略",
        Step.EXECUTE: "按路由决策并发执行各检索器，获取论文+资源",
        Step.CURATE: "分类统计 + 洞察提炼 + 质量评估",
        Step.OUTLINE: "LLM 基于检索结果生成逻辑细纲，评估各节点覆盖度",
        Step.SUPPLEMENT: "根据缺口查询补充检索结果",
        Step.DONE: "全部完成",
    }

    desc = step_descriptions.get(state.step, "")
    if state.step_detail:
        desc = state.step_detail

    panels.append(Panel(
        f"[dim]{desc}[/dim]",
        border_style="white",
        title="[当前步骤]",
        height=3,
    ))

    if state.step == Step.PLAN and state.plan_text:
        panels.append(Panel(
            state.plan_text[:2000],
            title="[检索规划预览]",
            border_style="green",
        ))

    if state.report_lines:
        tbl = Table(show_header=True, header_style="bold magenta", border_style="magenta")
        tbl.add_column("指标", style="cyan", width=20)
        tbl.add_column("数值", style="white")
        for line in state.report_lines:
            parts = line.split("|")
            if len(parts) == 2:
                tbl.add_row(parts[0].strip(), parts[1].strip())
        panels.append(Panel(tbl, title="[策展报告]", border_style="magenta"))

    # 显示每个源的检索结果（EXECUTE 阶段）
    if state.source_results and state.step in (Step.EXECUTE, Step.CURATE, Step.OUTLINE, Step.DONE):
        src_lines = [r.to_display() for r in state.source_results]
        panels.append(Panel(
            "\n".join(src_lines),
            title="[各源检索结果]",
            border_style="green",
        ))

    # 显示补搜结果（SUPPLEMENT 阶段）
    if state.supplement_results and state.step in (Step.SUPPLEMENT, Step.DONE):
        supp_lines = [r.to_display() for r in state.supplement_results]
        panels.append(Panel(
            "\n".join(supp_lines),
            title="[补搜结果]",
            border_style="blue",
        ))

    if state.outline_text:
        panels.append(Panel(
            state.outline_text[:3000],
            title="[细纲预览]",
            border_style="yellow",
        ))

    return Group(*panels)


def build_logs() -> Panel:
    """底部日志面板"""
    if not state.logs:
        lines = ["[dim]（日志将在工作流运行时显示）[/dim]"]
    else:
        # 显示最近 20 条
        lines = state.logs[-20:]
    return Panel(
        "\n".join(lines),
        title="[实时日志]",
        border_style="bright_black",
        style="dim",
    )


def refresh(layout: Layout) -> None:
    layout["header"].update(build_header())
    layout["main"].update(build_main())
    layout["logs"].update(build_logs())


def _build_source_results(report) -> None:
    """从报告结果中提取各源检索统计，存入 state.source_results"""
    from collections import defaultdict
    
    source_counts: dict[str, dict] = defaultdict(lambda: {"papers": 0, "resources": 0})
    
    # 统计论文来源
    for p in report.papers:
        src = p.source or "unknown"
        source_counts[src]["papers"] += 1
    
    # 统计资源来源
    for r in report.resources:
        src = r.source or "unknown"
        source_counts[src]["resources"] += 1
    
    # 转换为 SourceResult 列表
    state.source_results = []
    for src, counts in sorted(source_counts.items()):
        if counts["papers"] > 0 or counts["resources"] > 0:
            state.source_results.append(SourceResult(
                source=src,
                query="",
                papers_count=counts["papers"],
                resources_count=counts["resources"],
            ))


def _build_supplement_results(supp_report) -> None:
    """从补搜报告中提取各源检索统计，存入 state.supplement_results"""
    from collections import defaultdict
    
    source_counts: dict[str, dict] = defaultdict(lambda: {"papers": 0, "resources": 0})
    
    # 遍历每个查询的结果报告
    for query, sub_report in supp_report.result_map.items():
        for p in sub_report.papers:
            src = p.source or "unknown"
            source_counts[src]["papers"] += 1
        for r in sub_report.resources:
            src = r.source or "unknown"
            source_counts[src]["resources"] += 1
    
    # 转换为 SourceResult 列表
    state.supplement_results = []
    for src, counts in sorted(source_counts.items()):
        if counts["papers"] > 0 or counts["resources"] > 0:
            state.supplement_results.append(SourceResult(
                source=src,
                query="",
                papers_count=counts["papers"],
                resources_count=counts["resources"],
            ))


# ── 工作流 ─────────────────────────────────────────────────────────

async def run_workflow(topic: str, llm_api_key: Optional[str] = None) -> None:
    """执行完整工作流，各步骤实时更新 TUI"""
    from src.planner import Planner
    from src.executor import RetrieverManager
    from src.curation.curator import Curator

    state.topic = topic
    state.step = Step.PLAN
    state.progress = 0.05
    refresh(_layout)

    # ── Step 1: Plan Retrieval ──────────────────────────────────────
    state.step_detail = "LLM 正在分析主题、生成检索规划…"
    state.plan_text = ""
    refresh(_layout)

    try:
        planner_kwargs = {}
        if llm_api_key:
            planner_kwargs["llm_client"] = None  # will use env

        planner = Planner()
        plan = planner.plan_retrieval(topic)
    except Exception as e:
        logging.error(f"规划失败: {e}")
        state.step_detail = f"❌ 规划失败：{e}"
        state.progress = 0
        refresh(_layout)
        return

    # 格式化 plan 显示
    plan_lines = []
    plan_lines.append(f"[bold]主题[/bold]：{plan.topic}")
    plan_lines.append(f"[bold]核心概念[/bold]：{', '.join(plan.topic_analysis.get('core_concepts', []))}")
    plan_lines.append(f"[bold]子方向[/bold]：{', '.join(plan.topic_analysis.get('sub_directions', []))}")
    plan_lines.append(f"[bold]查询数[/bold]：{len(plan.queries)}")
    for i, q in enumerate(plan.queries):
        plan_lines.append(f"  [{i+1}] {q.query} → {q.target_sources} (优先:{q.priority})")

    state.plan_text = "\n".join(plan_lines)
    state.step_detail = f"✅ 规划完成，共 {len(plan.queries)} 条查询"
    state.progress = 0.2
    refresh(_layout)

    # ── Step 2: Execute Plan ───────────────────────────────────────
    state.step = Step.EXECUTE
    state.step_detail = "并发执行各检索器，获取论文和资源…"
    refresh(_layout)

    try:
        manager = RetrieverManager()
        # run() 是同步入口，内部管理事件循环
        report = await manager.execute_plan(plan, routing_mode="target_sources")
        
        # 统计各源结果（用于显示"从xxx找到了N个资源"）
        _build_source_results(report)
    except Exception as e:
        logging.error(f"检索执行失败: {e}")
        state.step_detail = f"❌ 检索失败：{e}"
        state.progress = 0
        refresh(_layout)
        return

    # 统计报告
    report_lines = [
        f"论文总数 | {report.total_papers}",
        f"经典论文（引用>50） | {report.classic_count}",
        f"前沿论文（2023+） | {report.frontier_count}",
        f"GitHub 资源 | {report.github_count}",
        f"Benchmark | {report.benchmark_count}",
        f"数据集 | {report.dataset_count}",
        f"执行查询数 | {len(report.queries)}",
    ]
    state.report_lines = report_lines
    state.step_detail = f"✅ 检索完成，获得 {report.total_papers} 篇论文、{len(report.resources)} 个资源"
    state.progress = 0.4
    refresh(_layout)

    # ── Step 3: Curation ────────────────────────────────────────────
    state.step = Step.CURATE
    state.step_detail = "策展分析：分类、统计、洞察提炼…"
    refresh(_layout)

    try:
        curator = Curator()
        report = curator.curate(report)
        eval_result = curator.evaluate(report)
    except Exception as e:
        logging.error(f"策展失败: {e}")
        eval_result = "未知"

    state.report_lines.extend([
        f"策展评级 | {eval_result}",
        f"分类数 | {len(report.categories)}",
    ])
    if report.insights:
        state.report_lines.append(f"洞察 | {report.insights[0][:50]}")
    if report.missing_directions:
        state.report_lines.append(f"缺口方向 | {', '.join(report.missing_directions[:3])}")

    state.step_detail = f"✅ 策展完成，评级：{eval_result}"
    state.progress = 0.6
    refresh(_layout)

    # ── Step 4: Generate Outline ───────────────────────────────────
    state.step = Step.OUTLINE
    state.step_detail = "LLM 生成综述细纲，评估各节点覆盖度…"
    state.outline_text = ""
    refresh(_layout)

    try:
        papers_summary = report.to_papers_summary()
        resources_summary = report.to_resources_summary()
        outline = planner.generate_outline(topic, papers_summary, resources_summary)
    except Exception as e:
        logging.error(f"细纲生成失败: {e}")
        state.step_detail = f"❌ 细纲生成失败：{e}"
        state.progress = 0
        refresh(_layout)
        return

    outline_lines = []
    outline_lines.append(f"[bold]摘要[/bold]：{outline.abstract[:200]}")
    outline_lines.append(f"[bold]整体覆盖度[/bold]：{outline.overall_coverage:.0%}")
    outline_lines.append(f"[bold]章节数[/bold]：{len(outline.sections)}")
    outline_lines.append("")
    for s in outline.sections:
        status_icon = {"sufficient": "✅", "partial": "⚠️", "insufficient": "❌", "unknown": "❓"}.get(s.coverage_status, "?")
        outline_lines.append(f"{status_icon} [{s.id}] {s.title}  {s.coverage_status}")
        if s.description:
            outline_lines.append(f"    └ {s.description[:60]}")
        if s.supplementary_queries:
            outline_lines.append(f"    └ 补搜: {', '.join(s.supplementary_queries[:2])}")
        for c in s.child_sections:
            cs_icon = {"sufficient": "✅", "partial": "⚠️", "insufficient": "❌", "unknown": "❓"}.get(c.coverage_status, "?")
            outline_lines.append(f"    {cs_icon} [{c.id}] {c.title}")

    state.outline_text = "\n".join(outline_lines)
    state.step_detail = f"✅ 细纲生成完成"
    state.progress = 0.8
    refresh(_layout)

    # ── Step 5: Supplement (if needed) ─────────────────────────────
    if outline.needs_supplement:
        state.step = Step.SUPPLEMENT
        
        # 使用详细版接口获取带源策略的补搜查询
        all_gap_queries, section_map_detailed = outline.collect_gap_queries_detailed()
        
        # 格式化补搜计划用于显示
        gap_plan_lines = [f"[bold]补搜计划（共 {len(all_gap_queries)} 条查询）[/bold]"]
        for sec_id, sec_gap_queries in section_map_detailed.items():
            sec_title = sec_gap_queries[0].section_title if sec_gap_queries else ""
            gap_plan_lines.append(f"  [bold]{sec_id}[/bold] {sec_title}:")
            for gq in sec_gap_queries:
                srcs = ', '.join(gq.target_sources) if gq.target_sources else '全源'
                gap_plan_lines.append(f"    • {gq.query} → {srcs}")
        state.outline_text = "\n".join(gap_plan_lines)
        
        state.step_detail = f"需要补搜 {len(all_gap_queries)} 条查询…"
        refresh(_layout)

        try:
            supp_report = await manager.supplementary_search_detailed(
                all_gap_queries, report, section_map_detailed
            )
            state.report_lines.extend([
                f"补搜新增论文 | {supp_report.total_new_papers}",
                f"补搜新增资源 | {supp_report.total_new_resources}",
                f"补搜耗时 | {supp_report.duration_seconds:.1f}秒",
            ])
            
            # 统计补搜各源结果
            _build_supplement_results(supp_report)
            refresh(_layout)  # 立即刷新显示补搜结果
        except Exception as e:
            logging.error(f"补搜失败: {e}")

    # ── Done ───────────────────────────────────────────────────────
    state.step = Step.DONE
    state.step_detail = "全部完成 ✅"
    state.progress = 1.0
    state.report_lines.append("")
    state.report_lines.append("[bold green]✅ 工作流完成！[/bold green]")
    refresh(_layout)


# ── 主入口 ─────────────────────────────────────────────────────────

def _setup_logging(debug: bool = False) -> None:
    """配置日志：Rich 屏幕输出 + UITailHandler 到日志面板"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if not any(isinstance(h, UITailHandler) for h in root_logger.handlers):
        root_logger.addHandler(tail_handler)

    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    if not any(isinstance(h, RichHandler) for h in root_logger.handlers):
        root_logger.addHandler(rich_handler)


def run_interactive() -> None:
    """交互模式：反复输入主题"""
    console.print(Rule("[bold cyan]ReviewForge TUI[/bold cyan]  ·  交互式论文检索与细纲生成", style="cyan"))
    console.print()

    global _layout
    _layout = make_layout()
    refresh(_layout)

    console.print("[dim]💡 输入主题并按回车开始，输入 [bold]quit[/bold] 退出[/dim]")
    console.print()

    with Live(_layout, console=console, refresh_per_second=8, transient=False) as live:
        while True:
            # 停止 Live 以允许终端输入
            live.stop()
            console.print("[cyan]▶[/cyan] 主题：", end="")
            topic = console.input().strip()

            if topic.lower() in ("quit", "exit", "q"):
                console.print("[dim]再见！[/dim]")
                break

            if not topic:
                console.print("[red]⚠ 主题不能为空，请输入有效主题名称[/red]")
                continue

            state.reset()
            state.topic = topic

            # 更新布局后重新启动 Live
            _layout["header"].update(Panel(
                Text.assemble(f"[bold cyan]ReviewForge TUI[/bold cyan]   🔍 [规划检索] | 主题：{topic}"),
                border_style="cyan",
            ))
            _layout["main"].update(Panel("[dim]LLM 正在分析主题、生成检索规划…[/dim]", border_style="white", title="[当前步骤]"))
            _layout["logs"].update(build_logs())
            live.start()

            asyncio.run(run_workflow(topic))

            console.print()
            console.print("[dim]" + "─" * 60 + "[/dim]")
            console.print("[dim]💡 输入新主题继续，或 [bold]quit[/bold] 退出[/dim]")
            console.print()


def run_once(topic: str) -> None:
    """单次模式：运行一次工作流后退出（--topic 参数用）"""
    global _layout
    _layout = make_layout()
    refresh(_layout)

    with Live(_layout, console=console, refresh_per_second=4, transient=False) as live:
        state.topic = topic
        _layout["header"].update(Panel(
            Text.assemble(f"[bold cyan]ReviewForge TUI[/bold cyan]   🔍 [规划检索] | 主题：{topic}"),
            border_style="cyan",
        ))
        live.refresh()
        asyncio.run(run_workflow(topic))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="reviewforge-tui",
        description="ReviewForge TUI — 交互式论文检索与细纲生成",
    )
    parser.add_argument(
        "--topic", "-t", type=str, default=None,
        help="直接指定主题，运行单次工作流后退出（非交互模式）",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="开启 debug 日志（显示详细调试信息）",
    )
    args = parser.parse_args()

    _setup_logging(debug=args.debug)

    if args.topic:
        run_once(args.topic)
    else:
        run_interactive()


_layout: Optional[Layout] = None


if __name__ == "__main__":
    main()