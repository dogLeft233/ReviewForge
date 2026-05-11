"""Timeline tab — Plotly scatter showing领域发展脉络."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.visualizer.schema import VisualizationData


def render(
    data: VisualizationData,
    year_range: tuple[int, int] | None = None,
    selected_categories: list[str] | None = None,
) -> None:
    st.subheader("🕰️ 领域发展脉络")

    events = data.timeline
    if not events:
        st.info("当前数据中没有 timeline 事件。")
        return

    rows = [
        {
            "year": e.year,
            "title": e.title or "(无标题)",
            "category": e.category or "未分类",
            "description": e.description or "",
        }
        for e in events
    ]
    df = pd.DataFrame(rows)

    if year_range is not None:
        lo, hi = year_range
        df = df[(df["year"] >= lo) & (df["year"] <= hi)]
    if selected_categories:
        df = df[df["category"].isin(selected_categories)]

    if df.empty:
        st.warning("当前筛选条件下没有匹配的事件。")
        return

    df = df.sort_values("year")
    fig = px.scatter(
        df,
        x="year",
        y="title",
        color="category",
        hover_data={"description": True, "year": True, "title": False, "category": True},
        size_max=18,
    )
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color="white")))
    fig.update_layout(
        height=max(400, 40 * len(df) + 120),
        title="按年份排列的代表性事件",
        xaxis_title="年份",
        yaxis_title="事件",
        legend_title="类别",
        margin=dict(l=10, r=10, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 事件明细"):
        st.dataframe(df, use_container_width=True, hide_index=True)
