"""
Qwen/Qwen3-8B 工具调用能力测试

测试 SiliconFlow 上 Qwen3-8B 能否胜任 ReviewForge 各环节：
1. 基础 API 连通性
2. 函数调用（Function Calling）能力 — 最关键
3. 结构化 JSON 输出
4. 各领域问答（检索策略、论文分析、策展评估）
"""

import json
import os
import sys
import time
from typing import Any

from openai import OpenAI

# ── 配置 ──

API_KEY = "sk-grnzvqmqpizcjwszwfuyirfhwocbopgwhcibkitmrpsoauye"
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "Qwen/Qwen3-8B"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

PASS = 0
FAIL = 0
TOTAL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL, TOTAL
    TOTAL += 1
    if ok:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")
        if detail:
            for line in detail.strip().split("\n"):
                print(f"     {line}")


# ═══════════════════════════════════════════════
# 1. 基础连通性测试
# ═══════════════════════════════════════════════

def test_basic_connectivity() -> None:
    print("\n" + "=" * 60)
    print("1️⃣  基础连通性测试")
    print("=" * 60)

    # 1.1 简单问答
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "回复'Hello'"}],
            temperature=0.1,
            max_tokens=50,
        )
        reply = resp.choices[0].message.content or ""
        check("简单问答", len(reply) > 0, f"回复为空: {reply!r}")
    except Exception as e:
        check(f"简单问答: {e}", False)

    # 1.2 超时设置
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "1+1=?"}],
            temperature=0.1,
            max_tokens=50,
            timeout=30,
        )
        reply = resp.choices[0].message.content or ""
        check("超时参数", len(reply) > 0)
    except Exception as e:
        check(f"超时参数: {e}", False)

    # 1.3 Token 计数
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "回复'A'"}],
            temperature=0.1,
            max_tokens=10,
        )
        usage = resp.usage
        check(
            "Token 计数",
            usage is not None
            and usage.prompt_tokens > 0
            and usage.completion_tokens > 0,
            f"usage={usage}",
        )
    except Exception as e:
        check(f"Token 计数: {e}", False)

    # 1.4 中文能力
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "请用中文说明什么是大数据处理"}],
            temperature=0.3,
            max_tokens=200,
        )
        reply = resp.choices[0].message.content or ""
        has_chinese = any("\u4e00" <= c <= "\u9fff" for c in reply)
        check("中文能力", has_chinese and len(reply) > 20, reply[:100])
    except Exception as e:
        check(f"中文能力: {e}", False)


# ═══════════════════════════════════════════════
# 2. 函数调用 (Function Calling) 测试
# ═══════════════════════════════════════════════

def test_function_calling() -> None:
    print("\n" + "=" * 60)
    print("2️⃣  函数调用 (Function Calling) 测试")
    print("=" * 60)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_papers",
                "description": "按关键词检索学术论文",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索关键词"},
                        "max_results": {
                            "type": "integer",
                            "description": "最大返回数量",
                        },
                        "source": {
                            "type": "string",
                            "enum": ["arxiv", "semantic_scholar", "dblp"],
                            "description": "检索来源",
                        },
                    },
                    "required": ["query", "source"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_github",
                "description": "搜索 GitHub 上的开源项目",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "sort": {
                            "type": "string",
                            "enum": ["stars", "forks", "updated"],
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "evaluate_report",
                "description": "评估综述报告质量",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "total_papers": {"type": "integer"},
                        "classic_papers": {"type": "integer"},
                        "frontier_papers": {"type": "integer"},
                        "has_github": {"type": "boolean"},
                        "has_benchmark": {"type": "boolean"},
                    },
                    "required": ["total_papers", "classic_papers", "frontier_papers"],
                },
            },
        },
    ]

    # 2.1 基础函数调用 — 应该调用 search_papers
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "帮我从 arXiv 上搜索关于大数据处理的最新论文",
                }
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=300,
        )
        msg = resp.choices[0].message
        has_tc = msg.tool_calls is not None and len(msg.tool_calls) > 0
        if has_tc:
            tc = msg.tool_calls[0]
            name_ok = tc.function.name == "search_papers"
            args = json.loads(tc.function.arguments)
            args_ok = "query" in args and args.get("source") == "arxiv"
            check(
                "自动选择 search_papers",
                name_ok and args_ok,
                f"name={tc.function.name}, args={tc.function.arguments}",
            )
        else:
            check("自动选择 search_papers", False, "未触发函数调用")
    except Exception as e:
        check(f"函数调用: {e}", False)

    # 2.2 强制函数调用
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "搜索 GitHub 上关于 big data 的项目",
                }
            ],
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "search_github"},
            },
            temperature=0.1,
            max_tokens=300,
        )
        msg = resp.choices[0].message
        has_tc = msg.tool_calls is not None and len(msg.tool_calls) > 0
        if has_tc:
            name_ok = msg.tool_calls[0].function.name == "search_github"
            args = json.loads(msg.tool_calls[0].function.arguments)
            args_ok = "query" in args
            check(
                "强制调用 search_github",
                name_ok and args_ok,
                f"args={msg.tool_calls[0].function.arguments}",
            )
        else:
            check("强制调用 search_github", False, "未触发函数调用")
    except Exception as e:
        check(f"强制调用: {e}", False)

    # 2.3 多工具调用 — 复杂查询需要多个工具协作
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "我需要写一份大数据处理综述。先找一些经典论文和前沿论文，"
                        "再看看 GitHub 上有没有相关项目。"
                    ),
                }
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=500,
        )
        msg = resp.choices[0].message
        num_tc = len(msg.tool_calls) if msg.tool_calls else 0
        check("多工具调用", num_tc >= 1, f"调用了 {num_tc} 个工具")
    except Exception as e:
        check(f"多工具调用: {e}", False)

    # 2.4 带上下文的工具调用 — 基于已检索结果的后续决策
    try:
        # 先模拟一次工具调用结果
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "我们已有 30 篇论文（10 篇高引用经典, 8 篇 2023年后前沿），"
                        "以及 5 个 GitHub 项目和 3 个 benchmark。请评估报告质量。"
                    ),
                }
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=300,
        )
        msg = resp.choices[0].message
        has_tc = msg.tool_calls is not None and len(msg.tool_calls) > 0
        if has_tc:
            tc = msg.tool_calls[0]
            fn_ok = tc.function.name == "evaluate_report"
            args = json.loads(tc.function.arguments)
            args_ok = (
                args.get("total_papers") == 30
                and args.get("classic_papers") == 10
            )
            check(
                "上下文驱动的工具选择",
                fn_ok and args_ok,
                f"name={tc.function.name}, args={tc.function.arguments}",
            )
        else:
            check("上下文驱动的工具选择", False, "未触发函数调用")
    except Exception as e:
        check(f"上下文驱动: {e}", False)


# ═══════════════════════════════════════════════
# 3. 结构化输出测试
# ═══════════════════════════════════════════════

def test_structured_output() -> None:
    print("\n" + "=" * 60)
    print("3️⃣  结构化 JSON 输出测试")
    print("=" * 60)

    # 3.1 JSON 模式
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {
                    "role": "user",
                    "content": (
                        "提取论文信息: 《A Survey of Big Data Processing》"
                        " by Zhang et al., VLDB 2023. "
                        "请以 {\"title\": ..., \"authors\": ..., \"venue\": ..., \"year\": ...} 格式返回"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=300,
        )
        content = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(content)
            has_title = "title" in parsed
            has_authors = "authors" in parsed
            check(
                "JSON 模式输出",
                has_title and has_authors,
                f"parsed keys={list(parsed.keys())}",
            )
        except json.JSONDecodeError:
            check("JSON 模式输出", False, f"非法 JSON: {content[:200]}")
    except Exception as e:
        check(f"JSON 模式: {e}", False)


# ═══════════════════════════════════════════════
# 4. 领域能力测试
# ═══════════════════════════════════════════════

def test_domain_capabilities() -> None:
    print("\n" + "=" * 60)
    print("4️⃣  领域能力测试")
    print("=" * 60)

    tests = [
        (
            "检索策略规划",
            "为了写'大数据处理技术综述'，需要搜索哪些关键词？请列出 5 个最相关的搜索查询。",
            lambda r: any(kw in r.lower() for kw in ["mapreduce", "spark", "hadoop", "flink", "data", "big", "distributed", "processing", "survey", "stream", "batch"]),
        ),
        (
            "论文分类",
            "将以下论文分类为 '经典' 或 '前沿': "
            "Dean & Ghemawat, MapReduce (2004); "
            "Zaharia et al., Spark (2012); "
            "Carbone et al., Flink (2015); "
            "某篇2024年的LLM-on-big-data论文。",
            lambda r: any(kw in r.lower() for kw in ["经典", "classic", "前沿", "frontier"]),
        ),
        (
            "技术概念解释",
            "用中文解释什么是 '数据倾斜 (data skew)' 以及它在大数据处理中为什么是核心问题",
            lambda r: any(kw in r.lower() for kw in ["数据倾斜", "skew", "不均匀", "倾斜"]),
        ),
        (
            "综述结构设计",
            "列出大数据处理技术综述论文的典型章节结构（用中文）",
            lambda r: any(kw in r.lower() for kw in ["引言", "介绍", "章节", "结构", "分类", "相关工作", "未来", "挑战"]),
        ),
    ]

    for name, prompt, checker in tests:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "你是大数据领域的专家，请用中文回答。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            reply = resp.choices[0].message.content or ""
            ok = checker(reply)
            check(name, ok, reply[:120])
        except Exception as e:
            check(f"{name}: {e}", False)


# ═══════════════════════════════════════════════
# 5. 长文本处理测试（Qwen3-8B 上下文窗口）
# ═══════════════════════════════════════════════

def test_long_context() -> None:
    print("\n" + "=" * 60)
    print("5️⃣  长文本处理测试")
    print("=" * 60)

    # 构造约 4k token 的输入
    paragraph = "大数据处理是计算机科学的重要领域。" * 200
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": f"{paragraph}\n\n以上文本提到了什么主题？请用一句话回答。"}
            ],
            temperature=0.1,
            max_tokens=100,
            timeout=60,
        )
        reply = resp.choices[0].message.content or ""
        check("长文本(4k+ tokens)", len(reply) > 0, reply[:100])
    except Exception as e:
        check(f"长文本: {e}", False)


# ═══════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════

def main() -> int:
    print("=" * 60)
    print(f"  Qwen/Qwen3-8B 工具调用能力测试")
    print(f"  模型: {MODEL}")
    print(f"  API: {BASE_URL}")
    print("=" * 60)

    test_basic_connectivity()
    test_function_calling()
    test_structured_output()
    test_domain_capabilities()
    test_long_context()

    print("\n" + "=" * 60)
    print(f"  结果: {PASS}/{TOTAL} 通过, {FAIL} 失败")
    if FAIL == 0:
        print("  🎉 全部通过！Qwen3-8B 可用于 ReviewForge")
    else:
        print(f"  ⚠️  {FAIL} 个测试失败，请注意")
    print("=" * 60)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
