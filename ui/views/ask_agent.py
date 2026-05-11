"""Ask Agent tab — fake agent that summarizes the visualization data."""

from __future__ import annotations

import streamlit as st

from src.visualizer.schema import VisualizationData


def fake_agent_answer(question: str, data: VisualizationData) -> str:
    """简易关键词路由 + 数据摘要。后续可替换为真实 RAG / LLM Agent。"""
    q = (question or "").lower().strip()

    parts: list[str] = [f"### 关于 **{data.topic}** 的回答"]

    def _section_overview() -> None:
        if data.overview.definition:
            parts.append(f"**领域定义**：{data.overview.definition}")
        if data.overview.core_questions:
            parts.append("**核心问题**：")
            parts.extend(f"- {q}" for q in data.overview.core_questions[:5])

    def _section_timeline() -> None:
        if not data.timeline:
            return
        sorted_events = sorted(data.timeline, key=lambda e: e.year)
        parts.append("**发展主线**（按时间）：")
        for e in sorted_events:
            parts.append(f"- {e.year}：{e.title}（{e.category or '未分类'}）")

    def _section_methods() -> None:
        if not data.methods:
            return
        parts.append("**关键技术方法**：")
        for m in data.methods[:8]:
            parts.append(f"- **{m.name}**（{m.category}）— {m.description}")

    def _section_papers() -> None:
        if not data.papers:
            return
        parts.append("**代表论文**：")
        for p in sorted(data.papers, key=lambda p: p.year or 0, reverse=True)[:6]:
            parts.append(f"- [{p.year}] {p.title} — {p.authors}（{p.venue}）")

    def _section_frontiers() -> None:
        if not data.frontiers:
            return
        parts.append("**当前前沿趋势**：")
        for f in data.frontiers:
            parts.append(f"- **{f.name}** — {f.description}")

    def _section_benchmarks() -> None:
        if not data.benchmarks:
            return
        parts.append("**Benchmark 概览**：")
        for b in data.benchmarks[:8]:
            parts.append(f"- {b.model} on {b.dataset}：{b.metric}={b.score}（{b.year}）")

    if not q:
        _section_overview()
        _section_timeline()
        _section_frontiers()
    elif any(k in q for k in ("发展", "历史", "脉络", "timeline", "history", "evolution")):
        _section_timeline()
    elif any(k in q for k in ("方法", "技术", "method", "approach", "technique")):
        _section_methods()
    elif any(k in q for k in ("论文", "文章", "paper", "publication")):
        _section_papers()
    elif any(k in q for k in ("前沿", "趋势", "frontier", "trend", "future")):
        _section_frontiers()
    elif any(k in q for k in ("benchmark", "sota", "性能", "指标", "score")):
        _section_benchmarks()
    elif any(k in q for k in ("定义", "什么是", "概述", "overview", "definition", "what is")):
        _section_overview()
    else:
        _section_overview()
        _section_methods()
        _section_frontiers()

    if len(parts) == 1:
        parts.append("（当前数据中没有可用于回答的内容。）")
    return "\n\n".join(parts)


def render(data: VisualizationData) -> None:
    st.subheader("💬 Ask Agent")
    st.caption("当前为基于本地 visualization_data 的简易回答；后续可替换为真实 RAG / LLM Agent。")

    examples = [
        "这个领域的发展主线是什么？",
        "有哪些关键技术方法？",
        "代表论文有哪些？",
        "目前的前沿趋势是什么？",
        "benchmark 表现如何？",
    ]
    cols = st.columns(len(examples))
    if "ask_agent_question" not in st.session_state:
        st.session_state["ask_agent_question"] = ""
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["ask_agent_question"] = ex

    question = st.text_input("提问", key="ask_agent_question", placeholder="例如：这个领域的发展主线是什么？")

    if st.button("提交", type="primary"):
        with st.spinner("Agent 思考中…"):
            answer = fake_agent_answer(question, data)
        st.markdown(answer)
    elif question:
        st.markdown(fake_agent_answer(question, data))
