#!/usr/bin/env python3
"""Web 工具测试程序

测试 src/tools 包下 web_search（博查搜索）和 web_fetch（网页抓取）功能。
运行方式（从项目根目录）:
    uv run python tests/test_tools.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

sys.path.insert(0, "")

from src.tools import web_fetch, WebFetchResult
from src.tools.exceptions import (
    ConfigurationError,
    FetchError,
)
from src.config import settings

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


INFO = lambda msg: print(f"  ℹ️  {msg}")


# ─────────────────────────────────────────────
# 1. web_search — Bocha 搜索
# ─────────────────────────────────────────────


def test_web_search() -> None:
    """Bocha 搜索测试"""
    print("\n═══ web_search (Bocha) ═══")

    key = getattr(settings, "bocha_api_key", "")
    if not key:
        ng("BOCHA_API_KEY 未配置，跳过搜索测试")
        return

    INFO(f"API key: {key[:8]}…{key[-4:]}")

    # 使用修复后的 web_search 函数
    from src.tools.web_search import web_search as ws

    results = ws("LLM reasoning survey 2025", count=5)

    ok(f"返回 {len(results)} 条结果")

    for i, r in enumerate(results[:3], 1):
        ok(f"  [{i}] {r.title}")
        INFO(f"        URL: {r.url}")
        INFO(f"        Desc: {r.description[:60]}…")

    # 验证数据完整性
    if results:
        first = results[0]
        ok(f"  首条标题: {first.title}")
        ok(f"  首条 URL: {first.url}")
        ok(f"  网站: {first.site_name}")
        ok(f"  description 不空: {bool(first.description)}")

    INFO("")
    INFO("已修复: web_search.py 改用 POST + JSON body 调用 Bocha API")


# ─────────────────────────────────────────────
# 2. web_fetch — 网页抓取
# ─────────────────────────────────────────────


def test_web_fetch() -> None:
    """网页抓取测试"""
    print("\n═══ web_fetch ═══")

    # 2a. 正常 HTML 页面
    print("  ── 正常页面 ──")
    try:
        result = web_fetch("https://httpbin.org/html", max_chars=1000, timeout=15)
        ok(f"状态码: {result.status_code}")
        ok(f"标题: {result.title}")
        content_len = len(result.content)
        ok(f"内容长度: {content_len} 字符")
        if content_len < 50:
            ng("内容过短，提取可能失败")
        else:
            ok(f"内容预览: {result.content[:80]}…")
    except Exception as e:
        ng(f"抓取失败: {e}")

    # 2b. 返回结构完整性
    print("  ── 返回结构 ──")
    try:
        result = web_fetch("https://httpbin.org/anything", max_chars=500, timeout=15)
        assert isinstance(result, WebFetchResult)
        ok(f"类型: WebFetchResult")
        ok(f"url: {result.url}")
        ok(f"title: {result.title}")
        ok(f"status_code: {result.status_code}")
        ok(f"content_type: {result.content_type}")
    except Exception as e:
        ng(f"结构验证失败: {e}")

    # 2c. 404 错误
    print("  ── 错误处理 ──")
    try:
        web_fetch("https://httpbin.org/status/404", timeout=10)
        ng("404 应抛出 FetchError")
    except FetchError as e:
        ok(f"404 正确抛出 FetchError: {e}")
    except Exception as e:
        ng(f"404 抛出类型不符合预期: {type(e).__name__}: {e}")

    # 2d. 无效 URL
    try:
        web_fetch("not-a-url", timeout=5)
        ng("无效 URL 应抛出 ConfigurationError")
    except ConfigurationError:
        ok("无效 URL 正确抛出 ConfigurationError")
    except Exception as e:
        ng(f"无效 URL 异常类型不符: {type(e).__name__}: {e}")

    # 2e. arXiv 论文页
    print("  ── arXiv 论文 ──")
    try:
        result = web_fetch(
            "https://arxiv.org/abs/1706.03762",
            max_chars=3000,
            timeout=20,
        )
        ok(f"状态码: {result.status_code}")
        ok(f"标题: {result.title}")
        ok(f"内容: {len(result.content)} 字符")
        if "Attention" in result.content[:500]:
            ok("正文包含 Attention (预期内容)")
        else:
            ng("正文不包含 Attention，可能 URL 结构变了")
    except FetchError as e:
        ng(f"arXiv 抓取失败: {e}")
    except Exception as e:
        ng(f"arXiv 异常: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────
# 3. web_fetch — PDF 检测
# ─────────────────────────────────────────────


def test_web_fetch_pdf() -> None:
    """PDF 检测"""
    print("\n═══ web_fetch PDF ═══")
    try:
        result = web_fetch(
            "https://arxiv.org/pdf/1706.03762.pdf",
            max_chars=500,
            timeout=20,
        )
        ok(f"PDF 标题: {result.title}")
        ok(f"content_type: {result.content_type}")
        ok(f"内容 (头部标记): {result.content[:80]}…" if result.content else ng("内容为空"))
    except Exception as e:
        ng(f"PDF 抓取异常: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────
# 4. web_fetch — HTTP 超时/错误
# ─────────────────────────────────────────────


def test_web_fetch_errors() -> None:
    """更多错误场景"""
    print("\n═══ web_fetch 错误场景 ═══")

    # 500 error
    try:
        web_fetch("https://httpbin.org/status/500", timeout=10)
        ng("500 应抛出异常")
    except FetchError:
        ok("500 正确抛出 FetchError")
    except Exception as e:
        INFO(f"500 异常: {type(e).__name__}: {e}")

    # 非 HTTP URL
    try:
        web_fetch("ftp://example.com/file", timeout=5)
        ng("非 HTTP URL 应抛出 ConfigurationError")
    except ConfigurationError:
        ok("非 HTTP URL 正确抛出 ConfigurationError")
    except Exception as e:
        ng(f"异常类型不符: {type(e).__name__}")


# ─────────────────────────────────────────────
# 5. 异常层次结构
# ─────────────────────────────────────────────


def test_exceptions() -> None:
    """tools.exceptions 异常层次"""
    print("\n═══ tools 异常层次 ═══")

    from src.tools.exceptions import (
        WebToolError,
        SearchError,
        FetchError,
        RateLimitError as ToolsRateLimitError,
        ConfigurationError as ToolsConfigError,
    )

    ok("SearchError → WebToolError: "
       + str(issubclass(SearchError, WebToolError)))
    ok("FetchError → WebToolError: "
       + str(issubclass(FetchError, WebToolError)))
    ok("RateLimitError → WebToolError: "
       + str(issubclass(ToolsRateLimitError, WebToolError)))
    ok("ConfigurationError → WebToolError: "
       + str(issubclass(ToolsConfigError, WebToolError)))


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 50)
    print("  ReviewForge — Tools 测试")
    print("=" * 50)

    test_exceptions()
    test_web_search()
    test_web_fetch()
    test_web_fetch_pdf()
    test_web_fetch_errors()

    print(f"\n{'=' * 50}")
    print(f"  结果: ✅ {PASS} 通过 | ❌ {FAIL} 失败")
    print(f"{'=' * 50}")

    sys.exit(1 if FAIL else 0)
