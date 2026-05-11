"""Frontier tab - card grid of research trends."""

from __future__ import annotations

import streamlit as st

from src.visualizer.schema import VisualizationData


def render(data: VisualizationData) -> None:
    st.subheader("Frontier Trends")

    if not data.frontiers:
        st.info("No frontier information in the current data.")
        return

    method_lookup = {m.id: m.name for m in data.methods}
    paper_lookup = {p.id: p for p in data.papers}

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
                    st.markdown(f"### {f.name or '(unnamed trend)'}")
                    if f.description:
                        st.write(f.description)
                    if f.importance:
                        st.markdown(f"**Importance**: {f.importance}")
                    if f.related_methods:
                        names = [method_lookup.get(mid, mid) for mid in f.related_methods]
                        st.markdown("**Related Methods**: " + ", ".join(f"`{n}`" for n in names))
                    if f.related_papers:
                        links = []
                        for pid in f.related_papers:
                            paper = paper_lookup.get(pid)
                            if paper and paper.url:
                                links.append(f"[{paper.title}]({paper.url})")
                            elif paper:
                                links.append(paper.title)
                            else:
                                links.append(pid)
                        st.markdown("**Related Papers**: " + " · ".join(links))
