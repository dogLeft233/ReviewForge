"""ReviewForge Scholar Agent Streamlit entry point.

Offline mode:
    streamlit run ui/app.py --offline

Online mode:
    streamlit run ui/app.py --online
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

# Make `src.*` importable when streamlit is invoked from the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.logging_config import DEFAULT_LOG_LEVEL, LOG_LEVELS, get_logger, set_log_level  # noqa: E402
from src.visualizer.loader import load_demo_data, load_json  # noqa: E402
from src.visualizer.schema import VisualizationData  # noqa: E402
from ui import online_workflow  # noqa: E402
from ui.views import (  # noqa: E402
    benchmark,
    chat_agent,
    frontier,
    knowledge_graph,
    links,
    method_map,
    overview,
    timeline,
)

RunMode = Literal["offline", "online"]

logger = get_logger(__name__)


st.set_page_config(
    page_title="ReviewForge Scholar Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _run_mode() -> RunMode:
    args = set(sys.argv[1:])
    if "--offline" in args and "--online" in args:
        st.error("只能选择一种运行模式：--offline 或 --online。")
        st.stop()
    if "--online" in args or "online" in args:
        return "online"
    return "offline"


def _load_offline_data() -> tuple[VisualizationData, str]:
    """Returns (data, source_label). Falls back to demo on any error."""

    uploaded = st.session_state.get("uploaded_file")
    if uploaded is None:
        return load_demo_data(), "内置 Demo (ASR)"
    try:
        return load_json(uploaded), f"上传文件：{uploaded.name}"
    except Exception as exc:
        st.sidebar.error(f"读取失败：{exc}")
        st.sidebar.info("已自动回退到内置 Demo 数据。")
        return load_demo_data(), "内置 Demo (回退)"


def _log_level_control() -> None:
    log_level = st.sidebar.selectbox(
        "日志级别",
        options=LOG_LEVELS,
        index=LOG_LEVELS.index(st.session_state.get("log_level", DEFAULT_LOG_LEVEL)),
        help="调整日志详细程度；DEBUG 会输出请求/响应详情",
    )
    if log_level != st.session_state.get("log_level"):
        st.session_state["log_level"] = log_level
        set_log_level(log_level)
        st.sidebar.success(f"日志级别已调整为：{log_level}")


def _empty_sidebar(mode: RunMode) -> None:
    st.sidebar.header("控制面板")
    st.sidebar.caption(f"运行模式：{mode}")
    st.sidebar.info("在线模式会在生成 visualization_data.json 后显示可视化筛选器。")
    st.sidebar.divider()
    _log_level_control()


def _sidebar(
    data: VisualizationData,
    source_label: str,
    *,
    mode: RunMode,
    allow_upload: bool,
) -> dict:
    st.sidebar.header("控制面板")
    st.sidebar.caption(f"运行模式：{mode}")
    if allow_upload:
        st.sidebar.file_uploader(
            "上传 visualization_data.json",
            type=["json"],
            key="uploaded_file",
            help="不上传则使用内置 ASR Demo 数据",
        )
    st.sidebar.caption(f"数据来源：{source_label}")
    st.sidebar.markdown(f"### 当前领域\n**{data.topic}**")

    st.sidebar.divider()

    categories = sorted({m.category for m in data.methods if m.category})
    selected_categories = st.sidebar.multiselect(
        "方法类别筛选",
        options=categories,
        default=categories,
        help="影响 Timeline 与 Method 详情的展示",
    )

    year_pool = (
        [e.year for e in data.timeline if e.year]
        + [p.year for p in data.papers if p.year]
        + [b.year for b in data.benchmarks if b.year]
    )
    if year_pool:
        ymin, ymax = min(year_pool), max(year_pool)
        if ymin == ymax:
            ymax = ymin + 1
        year_range = st.sidebar.slider("年份范围", ymin, ymax, (ymin, ymax))
    else:
        year_range = None

    st.sidebar.divider()
    _log_level_control()

    return {
        "selected_categories": selected_categories,
        "year_range": year_range,
    }


def _render_visualization_tabs(data: VisualizationData, filters: dict) -> None:
    tabs = st.tabs([
        "Overview",
        "Timeline",
        "Method Map",
        "Frontier",
        "Benchmark",
        "Knowledge Graph",
        "Ask Agent",
        "Links",
    ])

    with tabs[0]:
        overview.render(data)
    with tabs[1]:
        tl_categories = sorted({e.category for e in data.timeline if e.category})
        sel = filters["selected_categories"] or []
        applied = [c for c in tl_categories if not sel or c in sel] or tl_categories
        timeline.render(data, year_range=filters["year_range"], selected_categories=applied)
    with tabs[2]:
        method_map.render(data)
    with tabs[3]:
        frontier.render(data)
    with tabs[4]:
        benchmark.render(data)
    with tabs[5]:
        knowledge_graph.render(data)
    with tabs[6]:
        chat_agent.render(data)
    with tabs[7]:
        links.render(data)


def _render_offline_app() -> None:
    st.title("ReviewForge Scholar Agent")
    st.caption("Offline 模式：上传或加载已有 visualization_data.json 进行可视化分析。")

    data, source_label = _load_offline_data()
    filters = _sidebar(data, source_label, mode="offline", allow_upload=True)
    _render_visualization_tabs(data, filters)


def _render_online_app() -> None:
    existing_data = st.session_state.get("online_visualization_data")
    if isinstance(existing_data, VisualizationData):
        with st.sidebar:
            data, source_label = online_workflow.render(compact=True)
        if not isinstance(data, VisualizationData):
            st.rerun()

        filters = _sidebar(data, source_label, mode="online", allow_upload=False)
        st.title(data.topic or "ReviewForge Scholar Agent")
        st.caption("Online 模式可视化结果")
        _render_visualization_tabs(data, filters)
        return

    st.title("ReviewForge Scholar Agent")
    st.caption("Online 模式：从领域请求开始，自动完成检索、解析、增强和可视化展示。")

    data, source_label = online_workflow.render(compact=False)
    if data is None:
        _empty_sidebar("online")
        return

    filters = _sidebar(data, source_label, mode="online", allow_upload=False)
    st.title(data.topic or "ReviewForge Scholar Agent")
    st.caption("Online 模式可视化结果")
    _render_visualization_tabs(data, filters)


def main() -> None:
    mode = _run_mode()
    if mode == "online":
        _render_online_app()
    else:
        _render_offline_app()


main()
