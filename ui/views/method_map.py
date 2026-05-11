"""Method Map tab - Graphviz tree plus method details."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.visualizer.schema import VisualizationData


def _escape(text: str) -> str:
    return text.replace('"', '\\"')


def _build_dot(topic: str, methods) -> str:
    lines = ["digraph G {", "  rankdir=LR;", '  node [shape=box, style="rounded,filled", fontname="Helvetica"];']
    lines.append(f'  topic [label="{_escape(topic)}", fillcolor="#1f77b4", fontcolor="white"];')

    by_cat: dict[str, list] = {}
    for m in methods:
        by_cat.setdefault(m.category or "Uncategorized", []).append(m)

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
    st.subheader("Method Map")

    if not data.methods:
        st.info("No methods in the current data.")
        return

    dot = _build_dot(data.topic, data.methods)
    st.graphviz_chart(dot, use_container_width=True)

    paper_lookup = {p.id: p for p in data.papers}
    resources_by_method: dict[str, list] = {}
    for resource in data.resources:
        for method_id in resource.related_methods:
            resources_by_method.setdefault(method_id, []).append(resource)

    st.markdown("#### Method Details")
    rows = [
        {
            "name": m.name,
            "category": m.category,
            "description": m.description,
            "pros": "; ".join(m.pros),
            "cons": "; ".join(m.cons),
            "papers": len(m.papers),
        }
        for m in data.methods
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for method in data.methods:
        if not method.papers and not resources_by_method.get(method.id):
            continue
        with st.expander(f"Links for {method.name or method.id}"):
            for pid in method.papers:
                paper = paper_lookup.get(pid)
                if paper and paper.url:
                    st.markdown(f"- Paper: [{paper.title}]({paper.url})")
                elif paper:
                    st.markdown(f"- Paper: {paper.title}")
            for resource in resources_by_method.get(method.id, []):
                st.markdown(f"- {resource.type}: [{resource.name}]({resource.url})")
