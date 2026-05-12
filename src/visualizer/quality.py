"""Quality checks for VisualizationData.

The UI tolerates empty fields, but the adapter benefits from a compact signal
that says whether each tab has enough material to be useful.
"""

from __future__ import annotations

from typing import Any

from src.adapter.schema import VisualizationData


def _ratio(done: int, total: int) -> float:
    return round(done / total, 3) if total else 0.0


def assess_visualization_data(data: VisualizationData) -> dict[str, Any]:
    papers = data.papers
    methods = data.methods
    benchmarks = data.benchmarks
    frontiers = data.frontiers
    resources = data.resources

    paper_urls = sum(1 for p in papers if p.url)
    paper_years = sum(1 for p in papers if p.year)
    linked_methods = sum(1 for m in methods if m.papers)
    benchmark_scores = sum(1 for b in benchmarks if b.score)
    frontier_links = sum(1 for f in frontiers if f.related_methods or f.related_papers)

    return {
        "overview_ready": bool(
            data.overview.definition
            and data.overview.key_concepts
            and data.overview.core_questions
        ),
        "timeline_events": len(data.timeline),
        "papers": {
            "count": len(papers),
            "with_year_ratio": _ratio(paper_years, len(papers)),
            "with_url_ratio": _ratio(paper_urls, len(papers)),
        },
        "methods": {
            "count": len(methods),
            "linked_to_papers_ratio": _ratio(linked_methods, len(methods)),
        },
        "benchmarks": {
            "count": len(benchmarks),
            "with_score_ratio": _ratio(benchmark_scores, len(benchmarks)),
        },
        "frontiers": {
            "count": len(frontiers),
            "linked_ratio": _ratio(frontier_links, len(frontiers)),
        },
        "resources": {
            "count": len(resources),
            "with_url_ratio": _ratio(sum(1 for r in resources if r.url), len(resources)),
        },
        "graph": {
            "nodes": len(data.graph.nodes),
            "edges": len(data.graph.edges),
        },
        "needs_research": len(data.needs_research),
    }
