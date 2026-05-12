"""Frontier tab - card grid of research trends."""

from __future__ import annotations

import streamlit as st

from src.visualizer.schema import VisualizationData


def _is_recent_resource(resource) -> bool:
    text = f"{resource.name} {resource.description} {resource.url}".lower()
    return (
        any(year in text for year in ("2024", "2025", "2026"))
        or "arxiv.org/abs/24" in text
        or "arxiv.org/abs/25" in text
        or "arxiv.org/abs/26" in text
    )


def _is_recent_or_open_source(frontier, paper_lookup: dict, resources: list) -> bool:
    recent_paper = any(
        (paper := paper_lookup.get(pid)) is not None and 2024 <= paper.year <= 2026
        for pid in frontier.related_papers
    )
    if recent_paper:
        return True

    text = f"{frontier.name} {frontier.description} {frontier.importance}".lower()
    if any(token in text for token in ("open-source", "open source", "github", "2024", "2025", "2026")):
        return True

    for resource in resources:
        resource_text = f"{resource.name} {resource.description} {resource.url}".lower()
        name_hit = frontier.name.lower() in resource_text or resource.name.lower() in text
        if resource.type in {"github_repo", "model_or_space"} and name_hit:
            return True
        if resource.type == "paper" and name_hit and _is_recent_resource(resource):
            return True
    return False


def render(data: VisualizationData) -> None:
    st.subheader("Frontier Trends")

    if not data.frontiers:
        st.info("No frontier information in the current data.")
        return

    method_lookup = {m.id: m.name for m in data.methods}
    paper_lookup = {p.id: p for p in data.papers}
    resources_by_method: dict[str, list] = {}
    resources_by_paper: dict[str, list] = {}
    for resource in data.resources:
        for method_id in resource.related_methods:
            resources_by_method.setdefault(method_id, []).append(resource)
        for paper_id in resource.related_papers:
            resources_by_paper.setdefault(paper_id, []).append(resource)
    frontiers = [
        frontier for frontier in data.frontiers
        if _is_recent_or_open_source(frontier, paper_lookup, data.resources)
    ]

    if not frontiers:
        st.info("No source-backed open-source projects or 2024-2026 frontier papers yet.")
        return

    cols_per_row = 2
    for row_start in range(0, len(frontiers), cols_per_row):
        cols = st.columns(cols_per_row)
        for offset, col in enumerate(cols):
            idx = row_start + offset
            if idx >= len(frontiers):
                continue
            f = frontiers[idx]
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
                    resource_links = []
                    for method_id in f.related_methods:
                        resource_links.extend(resources_by_method.get(method_id, []))
                    for paper_id in f.related_papers:
                        resource_links.extend(resources_by_paper.get(paper_id, []))
                    frontier_text = f"{f.name} {f.description}".lower()
                    resource_links.extend(
                        resource for resource in data.resources
                        if resource.name.lower() in frontier_text or f.name.lower() in f"{resource.name} {resource.description}".lower()
                    )
                    seen_urls = set()
                    resource_lines = []
                    for resource in resource_links:
                        if not resource.url or resource.url in seen_urls:
                            continue
                        seen_urls.add(resource.url)
                        resource_lines.append(f"[{resource.name}]({resource.url})")
                    if resource_lines:
                        st.markdown("**Sources**: " + " · ".join(resource_lines))
