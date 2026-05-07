#!/usr/bin/env python3
"""
ReviewForge Searcher Demo
=========================
体验多源学术搜索：Arxiv / HuggingFace / GitHub / HackerNews / Bocha

用法:
    python demo.py "RAG"                        # 默认搜索
    python demo.py "LLM" --debug                # 开启 DEBUG 日志
    python demo.py "transformer" -p 20 -r 10 -N 5  # 自定义各分类数量
    python demo.py "ASR" --explore               # 先探索领域再搜索
    python demo.py "diffusion model" --level info  # 设置日志级别

环境变量（可选）:
    SILICONFLOW_API_KEY   - 硅基流动 API Key（默认使用内置额度）
    GITHUB_TOKEN          - GitHub Token（提升 GitHub 搜索限流）
    SERPER_API_KEY        - Serper API Key（开启 Google 搜索）
    BOCHA_API_KEY         - Bocha API Key（开启中文网页搜索）
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


async def run(query: str, max_papers: int, max_resources: int, max_news: int, log_level: str, explore: bool = False):
    logger = setup_logger(log_level)
    logger.debug(f"搜索启动: query={query!r}, papers={max_papers} resources={max_resources} news={max_news}, explore={explore}")

    # ─── 可选：Explorer 先探索领域 ────────────────────────
    searcher_profile = None
    if explore:
        print(f"\n{'='*60}")
        print(f"🔎 领域探索: {query}")
        print(f"{'='*60}")
        try:
            from src.explorer.explorer import DomainExplorer

            explorer = DomainExplorer(enable_llm=True)
            explorer_profile = explorer.explore(query)

            print(f"  核心概念: {', '.join(explorer_profile.core_concepts[:5])}")
            print(f"  相关主题: {', '.join(explorer_profile.related_topics[:5])}")
            if explorer_profile.key_terms:
                print(f"  关键术语: {len(explorer_profile.key_terms)} 个")
            print(f"  来源: {', '.join(explorer_profile.sources_used)}")
            print()

            # 转换为 searcher DomainProfile
            searcher_profile = DomainProfile(
                topic=explorer_profile.original_query,
                core_concepts=explorer_profile.core_concepts,
                related_fields=explorer_profile.related_topics,
                search_hints=explorer_profile.hn_discussions,
            )
            logger.info("Explorer 完成，profile: core_concepts=%d, related=%d",
                        len(searcher_profile.core_concepts), len(searcher_profile.related_fields))
        except Exception as e:
            logger.warning("Explorer 失败: %s，直接搜索", e)
            searcher_profile = DomainProfile(topic=query, core_concepts=[query])
    else:
        searcher_profile = DomainProfile(topic=query, core_concepts=[query])

    print(f"\n{'='*60}")
    print(f"🔍 搜索: {query}")
    print(f"{'='*60}")

    searcher = MetaSearcher()
    ctx = await searcher.search(
        query,
        domain_profile=searcher_profile,
        max_papers=max_papers,
        max_resources=max_resources,
        max_news=max_news,
    )

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
    parser.add_argument("--papers", "-p", type=int, default=30, help="论文最大数量（默认: 30）")
    parser.add_argument("--resources", "-r", type=int, default=30, help="资源最大数量（默认: 30）")
    parser.add_argument("--news", "-N", type=int, default=30, help="新闻最大数量（默认: 30）")
    parser.add_argument(
        "--debug", "-d", action="store_true", help="开启 DEBUG 日志（等同于 --level debug）"
    )
    parser.add_argument(
        "--explore", "-e", action="store_true", help="先使用 Explorer 探索领域（增加领域先验知识）"
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
    asyncio.run(run(args.query, args.papers, args.resources, args.news, log_level, explore=args.explore))


if __name__ == "__main__":
    main()
