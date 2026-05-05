"""测试 executor RetrieverManager 统一实现"""

import asyncio
import sys

sys.path.insert(0, "/mnt/e/Documents/ReviewForge")

from src.executor import RetrieverManager
from src.models import CurationReport, PaperCard, ResourceCard


async def test_search_all():
    """测试 search_all 单查询全源检索"""
    print("\n=== test_search_all ===")
    with RetrieverManager() as mgr:
        report = await mgr.search_all("deep learning image classification", max_results=5)
        print(f"topic: {report.topic}")
        print(f"papers: {len(report.papers)}")
        print(f"resources: {len(report.resources)}")
        for p in report.papers[:3]:
            print(f"  - [{p.source}] {p.title[:60]} ( cites={p.citation_count} )")
        assert len(report.papers) > 0, "should get some papers"
        print("PASS")


async def test_supplementary_search():
    """测试 supplementary_search 补搜"""
    print("\n=== test_supplementary_search ===")

    # 构建已有的 CurationReport
    existing_papers = [
        PaperCard(
            title="Deep Residual Learning for Image Recognition",
            url="https://arxiv.org/abs/1512.03385",
            authors=["He et al."],
            year=2015,
            citation_count=200000,
            source="arxiv",
        )
    ]
    existing = CurationReport(
        topic="image classification",
        papers=existing_papers,
        resources=[],
        total_papers=1,
        classic_count=1,
        frontier_count=0,
        github_count=0,
        benchmark_count=0,
        dataset_count=0,
        queries=["image classification"],
    )

    gap_queries = [
        "vision transformer ViT 2023",
        "CLIP multimodal learning",
    ]
    section_map = {
        "sec-1": ["vision transformer ViT 2023"],
        "sec-2": ["CLIP multimodal learning"],
    }

    with RetrieverManager() as mgr:
        supp = await mgr.supplementary_search(
            gap_queries,
            existing_report=existing,
            section_query_map=section_map,
            max_results=3,
        )
        print(f"queries_executed: {supp.queries_executed}")
        print(f"total_new_papers: {supp.total_new_papers}")
        print(f"total_new_resources: {supp.total_new_resources}")
        print(f"new_classics: {supp.new_classics}")
        print(f"new_frontiers: {supp.new_frontiers}")
        print(f"sections_improved: {supp.sections_improved}")
        print(f"duration: {supp.duration_seconds:.2f}s")
        for q, r in supp.result_map.items():
            print(f"  query={q!r} → {len(r.papers)} papers")
        assert supp.total_new_papers >= 0, "should count papers"
        assert supp.duration_seconds > 0, "should measure time"
        print("PASS")


async def test_execute_plan():
    """测试 execute_plan（需要 RetrievalPlan）"""
    print("\n=== test_execute_plan ===")
    from src.planner.schemas import QuerySpec, RetrievalPlan

    plan = RetrievalPlan(
        topic="image classification",
        queries=[
            QuerySpec(
                query="deep learning image classification",
                language="en",
                target_sources=["arxiv", "semantic_scholar"],
            ),
        ],
    )

    with RetrieverManager() as mgr:
        report = await mgr.execute_plan(plan, routing_mode="target_sources")
        print(f"papers: {len(report.papers)}")
        print(f"resources: {len(report.resources)}")
        print(f"queries: {len(report.queries)}")
        print("PASS")


async def main():
    print("Testing unified RetrieverManager (executor.py)")
    await test_search_all()
    await test_supplementary_search()
    await test_execute_plan()
    print("\n✅ All tests passed")


if __name__ == "__main__":
    asyncio.run(main())
