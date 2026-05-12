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

    complete_methods = [
        method for method in data.methods
        if method.description and method.papers
    ]

    if not complete_methods:
        st.info("No methods in the current data.")
        return

    dot = _build_dot(data.topic, complete_methods)
    st.graphviz_chart(dot, use_container_width=True)

    paper_lookup = {p.id: p for p in data.papers}
    resources_by_paper: dict[str, list] = {}
    for resource in data.resources:
        for paper_id in resource.related_papers:
            resources_by_paper.setdefault(paper_id, []).append(resource)

    st.markdown("#### Method Details")
    rows = [
        {
            "name": m.name,
            "category": m.category,
            "description": m.description,
            "pros": "\n".join(f"- {item}" for item in m.pros),
            "cons": "\n".join(f"- {item}" for item in m.cons),
            "papers": len(m.papers),
        }
        for m in complete_methods
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for method in complete_methods:
        with st.expander(f"Links for {method.name or method.id}"):
            st.markdown("##### Representative Papers")
            has_paper_link = False
            for pid in method.papers:
                paper = paper_lookup.get(pid)
                if paper and paper.url:
                    has_paper_link = True
                    st.markdown(f"- [{paper.title}]({paper.url})")
                elif paper:
                    resource = next((r for r in resources_by_paper.get(pid, []) if r.url), None)
                    if resource:
                        has_paper_link = True
                        st.markdown(f"- [{paper.title}]({resource.url})")
                    else:
                        st.markdown(f"- {paper.title}")
            if not has_paper_link:
                st.caption("No linked representative papers yet.")
            st.divider()
            st.caption("All literature links are also collected on the Links tab.")
