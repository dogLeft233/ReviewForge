"""Frontier tab — card grid of前沿趋势."""

from __future__ import annotations

import streamlit as st

from src.visualizer.schema import VisualizationData


def render(data: VisualizationData) -> None:
    st.subheader("🚀 前沿趋势")

    if not data.frontiers:
        st.info("当前数据中没有 frontier 信息。")
        return

    method_lookup = {m.id: m.name for m in data.methods}
    paper_lookup = {p.id: p.title for p in data.papers}

    cols_per_row = 2
    for row_start in range(0, len(data.frontiers), cols_per_row):
        cols = st.columns(cols_per_row)
        for offset, col in enumerate(cols):
            idx = row_start + offset
            if idx >= len(data.frontiers):
                continue
            f = data.frontiers[idx]
            with col:
                with st.container(border=True):
                    st.markdown(f"### {f.name or '(未命名趋势)'}")
                    if f.description:
                        st.write(f.description)
                    if f.importance:
                        st.markdown(f"**为什么重要**：{f.importance}")
                    if f.related_methods:
                        names = [method_lookup.get(mid, mid) for mid in f.related_methods]
                        st.markdown("**相关方法**：" + ", ".join(f"`{n}`" for n in names))
                    if f.related_papers:
                        names = [paper_lookup.get(pid, pid) for pid in f.related_papers]
                        st.markdown("**相关论文**：" + "； ".join(names))
