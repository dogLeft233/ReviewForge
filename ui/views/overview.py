"""Overview tab."""

from __future__ import annotations

import streamlit as st

from src.visualizer.schema import VisualizationData


def render(data: VisualizationData) -> None:
    st.subheader(f"📌 {data.topic}")
    if data.overview.definition:
        st.markdown(f"> {data.overview.definition}")
    else:
        st.info("尚未提供领域定义。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("关键概念", len(data.overview.key_concepts))
    c2.metric("发展事件", len(data.timeline))
    c3.metric("代表论文", len(data.papers))
    c4.metric("Benchmark", len(data.benchmarks))

    st.divider()

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 🎯 核心问题")
        if data.overview.core_questions:
            for q in data.overview.core_questions:
                st.markdown(f"- {q}")
        else:
            st.caption("（暂无核心问题）")
    with right:
        st.markdown("#### 🏷️ 关键概念")
        if data.overview.key_concepts:
            chips = " ".join(f"`{c}`" for c in data.overview.key_concepts)
            st.markdown(chips)
        else:
            st.caption("（暂无关键概念）")

    st.divider()
    st.markdown("#### 📚 代表论文")
    if not data.papers:
        st.info("当前数据中没有论文。")
        return

    for paper in sorted(data.papers, key=lambda p: p.year or 0, reverse=True):
        header = f"[{paper.year or '—'}] {paper.title or '(无标题)'}"
        with st.expander(header):
            meta_bits = []
            if paper.authors:
                meta_bits.append(f"**作者**：{paper.authors}")
            if paper.venue:
                meta_bits.append(f"**会议/期刊**：{paper.venue}")
            if meta_bits:
                st.markdown(" · ".join(meta_bits))
            if paper.summary:
                st.write(paper.summary)
            if paper.url:
                st.markdown(f"🔗 [原文链接]({paper.url})")
