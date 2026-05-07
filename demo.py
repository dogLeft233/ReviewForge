#!/usr/bin/env python3
"""
ReviewForge Searcher Demo
=========================
体验多源学术搜索：Arxiv / HuggingFace / GitHub / HackerNews / Bocha

用法:
    python demo.py "RAG"                        # 默认搜索
    python demo.py "LLM" --debug                # 开启 DEBUG 日志
    python demo.py "transformer" --max 20        # 自定义结果数量
    python demo.py "diffusion model" --level info  # 设置日志级别

环境变量（可选）:
    SILICONFLOW_API_KEY   - 硅基流动 API Key（默认使用内置额度）
    GITHUB_TOKEN          - GitHub Token（提升 GitHub 搜索限流）
    SERPER_API_KEY        - Serper API Key（开启 Google 搜索）
"""

import argparse
import asyncio
import logging
import sys
import os

from src.searcher.meta import MetaSearcher
from src.searcher.models import DomainProfile, ClassicalPaper

# ─── 日志级别映射 ───────────────────────────────────────────
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info":  logging.INFO,
    "warn":  logging.WARNING,
    "error": logging.ERROR,
}


def setup_logger(level: str) -> logging.Logger:
    lvl = LOG_LEVELS.get(level.lower(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("searcher")


def print_paper(p, i):
    badge = "📄"
    title = p.title[:72] if p.title else "(无标题)"
    url = p.url[:45] if p.url else ""
    score = f"{p.rank_score:.3f}"
    src = f"@{p.source}" if p.source else ""
    snippet = p.snippet[:80] if p.snippet else ""
    print(f"  {badge} #{i+1:02d} {score} | {title}")
    if url:
        print(f"       {url}")
    if snippet:
        print(f"       └─ {snippet[:75]}")


def print_resource(r, i):
    badge = "🔗"
    title = r.title[:64] if r.title else "(无标题)"
    score = f"{r.rank_score:.3f}"
    src = f"@{r.source}" if r.source else ""
    url = r.url[:50] if r.url else ""
    snippet = r.snippet[:60] if r.snippet else ""
    print(f"  {badge} #{i+1:02d} {score} | {title} {src}")
    if url:
        print(f"       {url}")
    if snippet:
        print(f"       └─ {snippet[:65]}")


def print_news(n, i):
    badge = "📰"
    title = n.title[:68] if n.title else "(无标题)"
    score = f"{n.rank_score:.3f}"
    src = f"@{n.source}" if n.source else ""
    url = n.url[:50] if n.url else ""
    snippet = n.snippet[:65] if n.snippet else ""
    print(f"  {badge} #{i+1:02d} {score} | {title} {src}")
    if url:
        print(f"       {url}")
    if snippet:
        print(f"       └─ {snippet[:70]}")


async def run(query: str, max_results: int, log_level: str):
    logger = setup_logger(log_level)
    logger.debug(f"搜索启动: query={query!r}, max_results={max_results}")

    profile = DomainProfile(
        topic=query,
        core_concepts=[query],
        classical_papers=[],
    )

    print(f"\n{'='*60}")
    print(f"🔍 搜索: {query}")
    print(f"{'='*60}")

    searcher = MetaSearcher()
    ctx = await searcher.search(query, domain_profile=profile, max_results=max_results)

    # ─── 论文 ──────────────────────────────────────────────
    print(f"\n📄 论文 ({len(ctx.papers)} 条)")
    print("-" * 60)
    if ctx.papers:
        for i, p in enumerate(ctx.papers):
            print_paper(p, i)
    else:
        print("  (无结果，可能因为 Arxiv 限流或网络问题)")

    # ─── 资源 ──────────────────────────────────────────────
    print(f"\n🔗 开源资源 ({len(ctx.resources)} 条)")
    print("-" * 60)
    if ctx.resources:
        for i, r in enumerate(ctx.resources):
            print_resource(r, i)
    else:
        print("  (无结果，GitHub/HuggingFace 需配置 token 或网络问题)")

    # ─── 新闻 ──────────────────────────────────────────────
    print(f"\n📰 新闻/帖子 ({len(ctx.news)} 条)")
    print("-" * 60)
    if ctx.news:
        for i, n in enumerate(ctx.news):
            print_news(n, i)
    else:
        print("  (无结果)")

    # ─── 状态摘要 ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("📊 搜索完成")
    print(f"   论文: {len(ctx.papers)}  资源: {len(ctx.resources)}  新闻: {len(ctx.news)}")
    total = len(ctx.papers) + len(ctx.resources) + len(ctx.news)
    print(f"   总计: {total} 条结果")
    print(f"{'='*60}\n")

    logger.debug("搜索完成，退出")


def main():
    parser = argparse.ArgumentParser(
        description="ReviewForge Searcher 体验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", default="RAG", help="搜索关键词（默认: RAG）")
    parser.add_argument("--max", "-n", type=int, default=8, help="每个来源的最大结果数（默认: 8）")
    parser.add_argument(
        "--debug", "-d", action="store_true", help="开启 DEBUG 日志（等同于 --level debug）"
    )
    parser.add_argument(
        "--level",
        "-l",
        choices=["debug", "info", "warn", "error"],
        default="info",
        help="日志级别（默认: info）",
    )
    args = parser.parse_args()

    log_level = "debug" if args.debug else args.level
    asyncio.run(run(args.query, args.max, log_level))


if __name__ == "__main__":
    main()
