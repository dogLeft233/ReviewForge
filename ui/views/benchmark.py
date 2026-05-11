"""Benchmark tab - compact benchmark catalog."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.visualizer.schema import VisualizationData


def render(data: VisualizationData) -> None:
    st.subheader("Benchmark")

    if not data.benchmarks:
        st.info("No benchmark information in the current data.")
        return

    rows = []
    for benchmark in data.benchmarks:
        name = benchmark.model or benchmark.dataset or "(unnamed benchmark)"
        rows.append(
            {
                "benchmark": name,
                "dataset": benchmark.dataset,
                "description": benchmark.description,
                "metric": benchmark.metric,
                "url": benchmark.url,
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("url"),
        },
    )
