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
            selected = st.radio(
                "Key concept",
                options=data.overview.key_concepts,
                horizontal=True,
                label_visibility="collapsed",
            )
            explanation = data.overview.key_concept_explanations.get(selected, "")
            if explanation:
                st.write(explanation)
            else:
                st.caption("No explanation available yet.")
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
