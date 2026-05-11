"""Links tab - centralized papers, code, datasets, and open resources."""

from __future__ import annotations

import streamlit as st

from src.visualizer.schema import VisualizationData


def render(data: VisualizationData) -> None:
    st.subheader("Links")

    if not data.resources and not any(p.url for p in data.papers):
        st.info("No paper/project links were extracted yet.")
        return

    method_lookup = {m.id: m.name for m in data.methods}
    paper_lookup = {p.id: p.title for p in data.papers}

    by_type: dict[str, list] = {}
    seen_urls: set[str] = set()
    for paper in data.papers:
        if not paper.url or paper.url in seen_urls:
            continue
        seen_urls.add(paper.url)
        by_type.setdefault("paper", []).append({
            "name": paper.title,
            "url": paper.url,
            "description": paper.summary,
            "related": [],
        })

    for resource in data.resources:
        if resource.url and resource.url in seen_urls:
            continue
        if resource.url:
            seen_urls.add(resource.url)
        related = []
        related.extend(method_lookup.get(mid, mid) for mid in resource.related_methods)
        related.extend(paper_lookup.get(pid, pid) for pid in resource.related_papers)
        by_type.setdefault(resource.type or "resource", []).append({
            "name": resource.name,
            "url": resource.url,
            "description": resource.description,
            "related": related,
        })

    for resource_type in sorted(by_type):
        st.markdown(f"#### {resource_type}")
        for resource in by_type[resource_type]:
            label = resource["name"] or resource["url"] or "(unnamed resource)"
            with st.container(border=True):
                if resource["url"]:
                    st.markdown(f"**[{label}]({resource['url']})**")
                else:
                    st.markdown(f"**{label}**")
                if resource["description"]:
                    st.write(resource["description"])

                if resource["related"]:
                    st.caption("Related: " + " · ".join(resource["related"]))
