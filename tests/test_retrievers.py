#!/usr/bin/env python3
"""检索器测试程序

测试 src/retrievers 包下所有数据源检索器的基本功能。
运行方式（从项目根目录）:
    uv run python tests/test_retrievers.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "")

from src.retrievers import ArxivRetriever, GithubRetriever, HuggingFaceRetriever
from src.retrievers.rate_limits import RateLimiter, get_source_config
from src.retrievers.exceptions import (
    RetrieverError,
    RateLimitError,
    AuthenticationError,
)

PASS = 0
FAIL = 0


def ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"  ✅ {msg}")


def ng(msg: str) -> None:
    global FAIL
    FAIL += 1
    print(f"  ❌ {msg}")


# ─────────────────────────────────────────────
# 1. RateLimiter
# ─────────────────────────────────────────────


def test_rate_limiter() -> None:
    """测试 RateLimiter 基础功能"""
    print("\n═══ RateLimiter ═══")

    # 间隔 0 → 不阻塞
    t0 = time.perf_counter()
    limiter = RateLimiter(min_interval_seconds=0.0)
    limiter.wait()
    limiter.wait()
    elapsed = time.perf_counter() - t0
    ok(f"间隔=0: 两次 wait 耗时 {elapsed:.3f}s (应 < 0.01s)")
    if elapsed > 0.05:
        ng("间隔=0 时不应阻塞")

    # 间隔 0.1
    limiter2 = RateLimiter(min_interval_seconds=0.1)
    t0 = time.perf_counter()
    limiter2.wait()
    limiter2.wait()
    elapsed2 = time.perf_counter() - t0
    ok(f"间隔=0.1: 两次 wait 耗时 {elapsed2:.3f}s (应 > 0.1s)")


# ─────────────────────────────────────────────
# 2. SourceRateLimit 配置
# ─────────────────────────────────────────────


def test_rate_limit_config() -> None:
    """测试各数据源的限速配置"""
    print("\n═══ 限速配置 ═══")

    sources = ["arxiv", "github", "huggingface", "semantic_scholar", "dblp"]
    for name in sources:
        cfg = get_source_config(name)
        ok(f"{name}: interval={cfg.min_interval_seconds}s, burst={cfg.burst}")

    # 不存在的 source → 宽松默认
    default = get_source_config("nonexistent_source")
    ok(f"fallback: interval={default.min_interval_seconds}s, burst={default.burst}")

    # RateLimiter 可以通过 min_interval setter 调整
    limiter = RateLimiter(1.0)
    assert limiter.min_interval == 1.0
    limiter.min_interval = 0.5
    ok(f"min_interval setter: 1.0 → {limiter.min_interval}")


# ─────────────────────────────────────────────
# 3. ArxivRetriever
# ─────────────────────────────────────────────


def test_arxiv() -> None:
    """Arxiv 论文检索测试"""
    print("\n═══ ArxivRetriever ═══")

    try:
        with ArxivRetriever() as arxiv:
            papers = arxiv.search("LoRA fine-tuning", max_results=5)
    except Exception as e:
        ng(f"搜索失败: {e}")
        return

    ok(f"获取 {len(papers)} 篇论文")

    if not papers:
        ng("返回为空")
        return

    for i, p in enumerate(papers[:3], 1):
        checks = []
        if p.title:
            checks.append("title")
        if p.url:
            checks.append("url")
        if p.authors:
            checks.append("authors")
        if p.abstract:
            checks.append("abstract")
        ok(f"  [{i}] {p.title[:50]}… ({', '.join(checks)})")

    # 验证数据完整性
    p = papers[0]
    ok(f"  标题: {p.title[:60]}…")
    ok(f"  作者: {p.authors[0] if p.authors else 'N/A'}")
    ok(f"  年份: {p.year}")
    ok(f"  URL: {p.url[:70]}…" if len(p.url) > 70 else f"  URL: {p.url}")
    ok(f"  引用: {p.citation_count}")


# ─────────────────────────────────────────────
# 4. HuggingFaceRetriever
# ─────────────────────────────────────────────


def test_huggingface() -> None:
    """HuggingFace 资源检索测试"""
    print("\n═══ HuggingFaceRetriever ═══")

    try:
        with HuggingFaceRetriever() as hf:
            resources = hf.search_resources("text classification", max_results=5)
    except Exception as e:
        ng(f"搜索失败: {e}")
        return

    ok(f"获取 {len(resources)} 个资源")

    if not resources:
        ng("返回为空")
        return

    for i, r in enumerate(resources[:3], 1):
        checks = []
        if r.name:
            checks.append("name")
        if r.url:
            checks.append("url")
        if r.description:
            checks.append("desc")
        ok(f"  [{i}] {r.name} ({', '.join(checks)})")

    r = resources[0]
    ok(f"  名称: {r.name}")
    ok(f"  ⭐Stars: {r.stars}")
    ok(f"  类型: {r.type}")

    # HuggingFace search_papers 返回空列表（hf 只 search_resources）
    papers = hf.search("test", max_results=3)
    ok(f"  search_papers: 返回 {len(papers)} 篇（应 = 0）")


# ─────────────────────────────────────────────
# 5. GithubRetriever
# ─────────────────────────────────────────────


def test_github() -> None:
    """GitHub 仓库检索测试"""
    print("\n═══ GithubRetriever ═══")

    try:
        with GithubRetriever() as gh:
            repos = gh.search_resources("LLM fine-tuning", max_results=5)
    except Exception as e:
        ng(f"搜索失败: {e}")
        return

    ok(f"获取 {len(repos)} 个仓库")

    if not repos:
        ng("返回为空")
        return

    for i, r in enumerate(repos[:3], 1):
        checks = []
        if r.name:
            checks.append("name")
        if r.url:
            checks.append("url")
        if r.stars:
            checks.append("stars")
        ok(f"  [{i}] {r.name} ⭐{r.stars} ({', '.join(checks)})")

    r = repos[0]
    ok(f"  名称: {r.name}")
    ok(f"  URL: {r.url}")
    ok(f"  ⭐Stars: {r.stars}")

    # GitHub search_papers → 空（只会返回 resources）
    papers = gh.search("test", max_results=3)
    ok(f"  search_papers: 返回 {len(papers)} 篇（应 = 0）")


# ─────────────────────────────────────────────
# 6. BaseRetriever 异常层次结构
# ─────────────────────────────────────────────


def test_exception_hierarchy() -> None:
    """异常继承结构验证"""
    print("\n═══ 异常层次 ═══")

    try:
        raise RateLimitError("test rate limit")
    except RetrieverError:
        ok("RateLimitError 继承自 RetrieverError")

    try:
        raise AuthenticationError("test auth")
    except RetrieverError:
        ok("AuthenticationError 继承自 RetrieverError")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 50)
    print("  ReviewForge — Retrievers 测试")
    print("=" * 50)

    test_rate_limiter()
    test_rate_limit_config()
    test_exception_hierarchy()
    test_arxiv()
    test_huggingface()
    test_github()

    print(f"\n{'=' * 50}")
    print(f"  结果: ✅ {PASS} 通过 | ❌ {FAIL} 失败")
    print(f"{'=' * 50}")

    sys.exit(1 if FAIL else 0)
