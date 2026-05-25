#!/usr/bin/env python3
"""最小化实验脚本 — 多 Topic 对比 + BFS 消融

用法:
    python scripts/run_experiments.py          # 跑全部实验
    python scripts/run_experiments.py --dry-run  # 预览会做什么

API 调用预算：3 次 pipeline run（~50 次 LLM 调用）
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ["PYTHONPATH"] = str(_ROOT / "src")

from src.core import CoreConfig, ReviewForge
from src.llm import LLM
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger("run_experiments")

RESULTS_DIR = _ROOT / "experiment_results"


def collect_metrics(topic: str, tmp_dir: Path) -> dict:
    """从已保存的 pipeline 输出中提取指标（零 API 消耗）"""
    metrics = {"topic": topic}

    # ── 从 step3 JSON 获取耗时和基本统计 ──
    step3_path = tmp_dir / "step3_writer_done.json"
    if step3_path.exists():
        data = json.loads(step3_path.read_text(encoding="utf-8"))
        metrics["explorer_elapsed_s"] = data.get("explorer_elapsed_seconds", 0)
        metrics["searcher_elapsed_s"] = data.get("searcher_elapsed_seconds", 0)
        metrics["writer_elapsed_s"] = data.get("writer_elapsed_seconds", 0)
        metrics["total_elapsed_s"] = (
            metrics["explorer_elapsed_s"]
            + metrics["searcher_elapsed_s"]
            + metrics["writer_elapsed_s"]
        )

        # Explorer 统计
        er = data.get("explorer_report", {})
        metrics["explorer_total_queries"] = er.get("total_queries", 0)
        metrics["classics_count"] = len(er.get("stage2_classics", []))
        metrics["benchmarks_count"] = len(er.get("stage3_benchmarks", []))
        metrics["concepts_count"] = len(er.get("stage1_concepts", []))

        # Searcher 统计
        papers = data.get("searcher_papers", [])
        metrics["papers_found"] = len(papers)
        if papers:
            scores = [p.get("select_score", 0) for p in papers if p.get("select_score")]
            if scores:
                metrics["reranker_score_max"] = round(max(scores), 4)
                metrics["reranker_score_min"] = round(min(scores), 4)
                metrics["reranker_score_mean"] = round(sum(scores) / len(scores), 4)
            depths = [p.get("depth", 0) for p in papers]
            for d in sorted(set(depths)):
                metrics[f"papers_depth_{d}"] = depths.count(d)

        # Writer 统计
        wr = data.get("writer_report", {})
        if isinstance(wr, dict):
            metrics["has_title"] = bool(wr.get("title"))
            metrics["has_abstract"] = bool(wr.get("abstract"))
            metrics["has_body"] = bool(wr.get("body"))
            metrics["keywords_count"] = len(wr.get("keywords", []))

    # ── 从 visualization_data.json 获取完整性指标 ──
    viz_path = tmp_dir / "visualization_data.json"
    if viz_path.exists():
        viz = json.loads(viz_path.read_text(encoding="utf-8"))
        metrics["needs_research_count"] = len(viz.get("needs_research", []))
        metrics["viz_methods_count"] = len(viz.get("methods", []))
        metrics["viz_papers_count"] = len(viz.get("papers", []))
        metrics["viz_timeline_count"] = len(viz.get("timeline", []))
        metrics["viz_frontiers_count"] = len(viz.get("frontiers", []))
        metrics["viz_benchmarks_count"] = len(viz.get("benchmarks", []))
        graph = viz.get("graph", {})
        metrics["graph_nodes"] = len(graph.get("nodes", []))
        metrics["graph_edges"] = len(graph.get("edges", []))

    return metrics


def run_one(llm: LLM, topic: str, bfs_enabled: bool, label: str) -> dict:
    """跑一次 pipeline，返回收集的指标"""
    logger.info("=" * 60)
    logger.info("▶ 开始: %s (BFS=%s)", label, bfs_enabled)
    t0 = time.time()

    config = CoreConfig(
        bfs_enabled=bfs_enabled,
        bfs_expand_layers=2,
        bfs_search_queries_count=5,
        bfs_search_papers_count=3,
        bfs_expand_papers_count=15,
        bfs_rerank_top_n=100,
        verbose=False,
    )
    rf = ReviewForge(llm=llm, config=config)
    result = rf.run(topic)

    elapsed = time.time() - t0
    logger.info("✓ 完成: %s (%.0fs)", label, elapsed)

    # 收集指标
    slug = _slugify(topic)
    tmp_dir = _ROOT / "tmp" / slug
    metrics = collect_metrics(topic, tmp_dir)
    metrics["label"] = label
    metrics["bfs_enabled"] = bfs_enabled
    metrics["wall_elapsed_s"] = round(elapsed, 1)

    return metrics


def _slugify(topic: str) -> str:
    import re
    slug = re.sub(r"[\s\-]+", "_", topic.strip())
    slug = re.sub(r"[^\w一-龥_]", "", slug)
    return slug[:40] or "untitled"


def main():
    parser = argparse.ArgumentParser(description="ReviewForge 最小化实验")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    args = parser.parse_args()

    # ── 实验计划 ──
    experiments = [
        # (topic, bfs_enabled, label)
        ("LoRA fine-tuning", True, "LoRA-Full"),
        ("Graph Neural Networks", True, "GNN-Full"),
        ("LoRA fine-tuning", False, "LoRA-NoBFS"),
    ]

    if args.dry_run:
        print("\n实验计划（共 3 次 pipeline run）：")
        for topic, bfs, label in experiments:
            print(f"  {label}: topic='{topic}', BFS={bfs}")
        print(f"\n预计 LLM 调用：~50 次")
        return

    # ── 初始化 ──
    llm = LLM(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        max_concurrency=settings.llm_max_concurrency,
        timeout_seconds=300.0,
        max_tokens=4096,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for topic, bfs, label in experiments:
        try:
            metrics = run_one(llm, topic, bfs, label)
            all_results.append(metrics)
            # 每完成一个就保存（防止后续失败丢失数据）
            out_path = RESULTS_DIR / "results.json"
            out_path.write_text(
                json.dumps(all_results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("✗ 失败: %s — %s", label, e, exc_info=True)
            all_results.append({"label": label, "topic": topic, "error": str(e)})

    # ── 汇总输出 ──
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    keys = [
        "label", "total_elapsed_s", "papers_found",
        "needs_research_count", "reranker_score_mean",
        "viz_methods_count", "viz_benchmarks_count",
        "graph_nodes", "graph_edges",
    ]
    print(f"{'Label':<16} {'Time':>8} {'Papers':>8} {'Gaps':>6} {'Rerank':>7} "
          f"{'Methods':>8} {'BMs':>5} {'Nodes':>6} {'Edges':>6}")
    print("-" * 90)
    for r in all_results:
        if "error" in r:
            print(f"{r['label']:<16} {'ERROR: ' + r['error'][:50]}")
            continue
        print(f"{r.get('label',''):<16} "
              f"{r.get('total_elapsed_s',0):>7.0f}s "
              f"{r.get('papers_found',0):>8} "
              f"{r.get('needs_research_count',0):>6} "
              f"{r.get('reranker_score_mean',0):>6.3f} "
              f"{r.get('viz_methods_count',0):>8} "
              f"{r.get('viz_benchmarks_count',0):>5} "
              f"{r.get('graph_nodes',0):>6} "
              f"{r.get('graph_edges',0):>6}")

    print(f"\n结果已保存: {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
