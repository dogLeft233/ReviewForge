"""Benchmark tab — table + Plotly score-over-year trend chart."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.visualizer.schema import VisualizationData


def render(data: VisualizationData) -> None:
    st.subheader("📊 Benchmark / SOTA")

    if not data.benchmarks:
        st.info("当前数据中没有 benchmark 信息。")
        return

    df = pd.DataFrame([b.model_dump() for b in data.benchmarks])

    st.caption("⚠️ 注意：不同 metric 优劣方向不同（例如 WER 越低越好，Accuracy 越高越好）。请结合 metric 列阅读。")

    st.markdown("#### 全部记录")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### 趋势图（按年份）")
    plot_df = df[df["year"] > 0].copy()
    if plot_df.empty:
        st.warning("benchmark 没有可用的 year 字段，无法绘制趋势图。")
        return

    color_field = "dataset" if plot_df["dataset"].nunique() > 1 else "model"
    fig = px.scatter(
        plot_df,
        x="year",
        y="score",
        color=color_field,
        symbol="metric",
        hover_data={"model": True, "dataset": True, "metric": True, "score": True, "year": True},
        size_max=20,
    )
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color="white")))
    fig.update_layout(
        height=480,
        title="Benchmark 分数随年份变化（请结合 metric 判断高低优劣）",
        xaxis_title="年份",
        yaxis_title="score",
        margin=dict(l=10, r=10, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
