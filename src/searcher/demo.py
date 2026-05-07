#!/usr/bin/env python3
"""MetaSearcher 调试体验脚本

用法:
    # DEBUG 模式（显示详细日志）
    python -m src.searcher.demo

    # 自定义查询
    python -m src.searcher.demo "diffusion model image generation"

环境变量（可选）:
    SILICONFLOW_API_KEY  — SiliconFlow API Key（Rerank 用）
    SERPER_API_KEY       — Serper API Key（可选）
    GITHUB_TOKEN         — GitHub Token（可选，限速 60/小时）
"""

import argparse
import asyncio
import logging
import sys


def setup_logging(level: int = logging.DEBUG) -> None:
    """配置调试日志格式"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Adapter 层的 retriever 也打印 DEBUG
    logging.getLogger("src.retrievers").setLevel(logging.DEBUG)


async def main(query: str, max_results: int) -> None:
    from src.searcher.meta import MetaSearcher

    print(f"\n{'='*60}")
    print(f"  MetaSearcher Demo — query: '{query}'")
    print(f"{'='*60}\n")

    searcher = MetaSearcher()
    ctx = await searcher.search(query, max_results=max_results)

    print(f"\n[结果摘要]")
    print(f"  Papers:    {len(ctx.papers)} 条")
    print(f"  Resources: {len(ctx.resources)} 条")
    print(f"  News:      {len(ctx.news)} 条")

    if ctx.papers:
        print(f"\n[论文 Top 3]")
        for i, p in enumerate(ctx.papers[:3], 1):
            print(f"  {i}. {p.title}")
            print(f"     {p.url} (score={p.rank_score:.4f})")

    if ctx.resources:
        print(f"\n[资源 Top 3]")
        for i, r in enumerate(ctx.resources[:3], 1):
            print(f"  {i}. {r.title}")
            print(f"     {r.url}")

    if ctx.news:
        print(f"\n[新闻 Top 3]")
        for i, n in enumerate(ctx.news[:3], 1):
            print(f"  {i}. {n.title}")
            print(f"     {n.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaSearcher Debug Demo")
    parser.add_argument("query", nargs="?", default="transformer architecture attention mechanism",
                        help="搜索主题（默认: transformer architecture attention mechanism）")
    parser.add_argument("-n", "--max-results", type=int, default=5,
                        help="每个分类最大结果数（默认: 5）")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="只显示结果，不显示 DEBUG 日志")
    args = parser.parse_args()

    level = logging.WARNING if args.quiet else logging.DEBUG
    setup_logging(level=level)

    asyncio.run(main(args.query, args.max_results))
