"""ReviewForge Scholar Agent — Streamlit MVP entry point.

Run from project root:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src.*` importable when streamlit is invoked from the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.visualizer.loader import load_demo_data, load_json  # noqa: E402
from src.visualizer.schema import VisualizationData  # noqa: E402
from ui.views import (  # noqa: E402
    ask_agent,
    benchmark,
    frontier,
    knowledge_graph,
    links,
    method_map,
    overview,
    timeline,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────


st.set_page_config(
    page_title="ReviewForge Scholar Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_data() -> tuple[VisualizationData, str]:
    """Returns (data, source_label). Falls back to demo on any error."""
    uploaded = st.session_state.get("uploaded_file")
    if uploaded is None:
        return load_demo_data(), "内置 Demo (ASR)"
    try:
        return load_json(uploaded), f"上传文件：{uploaded.name}"
    except Exception as exc:  # surface error in sidebar, fall back to demo
        st.sidebar.error(f"读取失败：{exc}")
        st.sidebar.info("已自动回退到内置 Demo 数据。")
        return load_demo_data(), "内置 Demo (回退)"


def _sidebar(data: VisualizationData, source_label: str) -> dict:
    st.sidebar.header("⚙️ 控制面板")
    st.sidebar.file_uploader(
        "上传 visualization_data.json",
        type=["json"],
        key="uploaded_file",
        help="不上传则使用内置 ASR Demo 数据",
    )
    st.sidebar.caption(f"📦 数据来源：{source_label}")
    st.sidebar.markdown(f"### 当前领域\n**{data.topic}**")

    st.sidebar.divider()

    # Method category filter (used by Method Map / Frontier views indirectly)
    categories = sorted({m.category for m in data.methods if m.category})
    selected_categories = st.sidebar.multiselect(
        "方法类别筛选",
        options=categories,
        default=categories,
        help="影响 Timeline 与 Method 详情的展示",
    )

    # Year range from timeline + papers + benchmarks
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

    return {
        "selected_categories": selected_categories,
        "year_range": year_range,
    }


def main() -> None:
    st.title("ReviewForge Scholar Agent")
    st.caption("帮助学者快速理解一个研究领域的发展脉络、技术地图与前沿趋势")

    data, source_label = _load_data()
    filters = _sidebar(data, source_label)

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
        # Timeline category filter merges method categories + timeline categories.
        tl_categories = sorted({e.category for e in data.timeline if e.category})
        # If the user filtered method categories, also restrict timeline if any overlap.
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
        ask_agent.render(data)
    with tabs[7]:
        links.render(data)


main()
