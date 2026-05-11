"""Overview tab."""

from __future__ import annotations

import streamlit as st

from src.visualizer.schema import VisualizationData


def render(data: VisualizationData) -> None:
    st.subheader(data.topic)
    if data.overview.definition:
        st.markdown(f"> {data.overview.definition}")
    else:
        st.info("No domain definition provided yet.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Key Concepts", len(data.overview.key_concepts))
    c2.metric("Timeline Events", len(data.timeline))
    c3.metric("Papers", len(data.papers))
    c4.metric("Benchmarks", len(data.benchmarks))
    c5.metric("Links", len(data.resources))

    st.divider()

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Core Questions")
        if data.overview.core_questions:
            for q in data.overview.core_questions:
                st.markdown(f"- {q}")
        else:
            st.caption("No core questions yet.")
    with right:
        st.markdown("#### Key Concepts")
        if data.overview.key_concepts:
            chips = " ".join(f"`{c}`" for c in data.overview.key_concepts)
            st.markdown(chips)
        else:
            st.caption("No key concepts yet.")

    st.divider()
    st.markdown("#### Representative Papers")
    if not data.papers:
        st.info("No papers in the current data.")
    else:
        for paper in sorted(data.papers, key=lambda p: p.year or 0, reverse=True):
            header = f"[{paper.year or '-'}] {paper.title or '(untitled)'}"
            with st.expander(header):
                meta_bits = []
                if paper.authors:
                    meta_bits.append(f"**Authors**: {paper.authors}")
                if paper.venue:
                    meta_bits.append(f"**Venue**: {paper.venue}")
                if meta_bits:
                    st.markdown(" · ".join(meta_bits))
                if paper.summary:
                    st.write(paper.summary)
                if paper.url:
                    st.link_button("Open paper", paper.url)

    st.divider()
    st.markdown("#### Papers, Code, and Open Resources")
    if not data.resources:
        st.caption("No paper/project links were extracted yet.")
        return

    for resource in data.resources:
        label = resource.name or resource.url
        with st.container(border=True):
            st.markdown(f"**[{label}]({resource.url})**")
            st.caption(resource.type)
            if resource.description:
                st.write(resource.description)
