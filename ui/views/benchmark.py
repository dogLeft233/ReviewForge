"""Benchmark tab - table + Plotly score-over-year trend chart."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.visualizer.schema import VisualizationData


def render(data: VisualizationData) -> None:
    st.subheader("Benchmark / SOTA")

    if not data.benchmarks:
        st.info("No benchmark information in the current data.")
        return

    df = pd.DataFrame([b.model_dump() for b in data.benchmarks])

    st.caption("Different metrics have different directions, for example lower WER is better while higher accuracy is better.")

    st.markdown("#### Records")
    st.dataframe(df, use_container_width=True, hide_index=True)

    linked = [b for b in data.benchmarks if b.url]
    if linked:
        st.markdown("#### Benchmark Links")
        for b in linked:
            label = " / ".join(part for part in (b.model, b.dataset, b.metric) if part) or b.url
            st.markdown(f"- [{label}]({b.url})")

    st.markdown("#### Trend By Year")
    plot_df = df[df["year"] > 0].copy()
    if plot_df.empty:
        st.warning("No usable year field for benchmark trend plotting.")
        return

    color_field = "dataset" if plot_df["dataset"].nunique() > 1 else "model"
    fig = px.scatter(
        plot_df,
        x="year",
        y="score",
        color=color_field,
        symbol="metric",
        hover_data={"model": True, "dataset": True, "metric": True, "score": True, "year": True, "url": True},
        size_max=20,
    )
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color="white")))
    fig.update_layout(
        height=480,
        title="Benchmark scores over time",
        xaxis_title="Year",
        yaxis_title="Score",
        margin=dict(l=10, r=10, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
