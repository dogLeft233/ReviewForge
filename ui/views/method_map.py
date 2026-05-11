"""Method Map tab — Graphviz tree (topic → category → method) + detail table."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.visualizer.schema import VisualizationData


def _escape(text: str) -> str:
    return text.replace('"', '\\"')


def _build_dot(topic: str, methods) -> str:
    lines = ["digraph G {", '  rankdir=LR;', '  node [shape=box, style="rounded,filled", fontname="Helvetica"];']
    lines.append(f'  topic [label="{_escape(topic)}", fillcolor="#1f77b4", fontcolor="white"];')

    by_cat: dict[str, list] = {}
    for m in methods:
        by_cat.setdefault(m.category or "未分类", []).append(m)

    for ci, (cat, ms) in enumerate(by_cat.items()):
        cat_id = f"cat_{ci}"
        lines.append(f'  {cat_id} [label="{_escape(cat)}", fillcolor="#ffbb78"];')
        lines.append(f"  topic -> {cat_id};")
        for mi, m in enumerate(ms):
            method_id = f"m_{ci}_{mi}"
            lines.append(f'  {method_id} [label="{_escape(m.name or m.id)}", fillcolor="#aec7e8"];')
            lines.append(f"  {cat_id} -> {method_id};")

    lines.append("}")
    return "\n".join(lines)


def render(data: VisualizationData) -> None:
    st.subheader("🗺️ 技术分类地图")

    if not data.methods:
        st.info("当前数据中没有 method。")
        return

    dot = _build_dot(data.topic, data.methods)
    st.graphviz_chart(dot, use_container_width=True)

    st.markdown("#### 方法详情")
    rows = [
        {
            "name": m.name,
            "category": m.category,
            "description": m.description,
            "pros": "; ".join(m.pros),
            "cons": "; ".join(m.cons),
        }
        for m in data.methods
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
