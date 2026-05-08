"""Searcher — 核心搜索能力封装（多轮对话 + 工具调用）"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 嵌入模型工作原理相关搜索词
EMBEDDING_SEARCH_QUERIES = [
    "text embedding semantic search how does it work",
    "embedding model query optimization best practices",
]


def _parse_tool_calls(text: str) -> list[tuple[str, str]]:
    """从模型回复中解析工具调用

    支持两种格式：
    1. 「tool_call」web_search("query")「/tool_call」
    2. web_search("query")

    Returns:
        [(tool_name, args_str), ...]
    """
    results = []

    # 格式 1：<tool_call>tool_name("args")</tool_call>
    pattern1 = re.compile(r'<tool_call>\s*(\w+)\s*\(\s*"([^"]*)"\s*\)\s*</tool_call>', re.DOTALL)
    for m in pattern1.finditer(text):
        results.append((m.group(1), m.group(2)))

    # 格式 2：纯文本 tool_name("args")
    if not results:
        pattern2 = re.compile(r'\b(\w+)\s*\(\s*"([^"]*)"\s*\)', re.DOTALL)
        # 排除内置函数名
        for m in pattern2.finditer(text):
            name = m.group(1)
            if name not in ("print", "len", "str", "int", "float", "list", "dict", "tuple", "set"):
                results.append((name, m.group(2)))

    return results


def _execute_tool(tool_name: str, args_str: str) -> str:
    """执行单个工具调用，返回结果文本"""
    import json

    # 从 args_str 解析 kwargs（如 count=5）
    kwargs = {}
    if args_str:
        try:
            # 尝试解析为 JSON（支持复杂参数）
            kwargs = json.loads("{" + args_str + "}")
        except Exception:
            # 回退：把整个 args_str 当作 query
            kwargs = {"query": args_str}

    logger.info("执行工具: %s kwargs=%s", tool_name, kwargs)

    # 延迟导入避免循环
    try:
        from searcher_tools import web_search, web_fetch
    except ImportError:
        # 从包内导入
        from . import searcher_tools

        return searcher_tools.execute(tool_name, kwargs)

    if tool_name == "web_search":
        query = kwargs.get("query", "")
        count = kwargs.get("count", 5)
        results = web_search(query, count=count)
        return _format_search_results(results)
    elif tool_name == "web_fetch":
        url = kwargs.get("url", "")
        results = web_fetch(url, max_chars=3000)
        return _format_fetch_result(results)
    else:
        return f"[unknown tool: {tool_name}]"


def _format_search_results(results: list) -> str:
    if not results:
        return "（无搜索结果）"
    lines = []
    for r in results[:5]:  # 最多取 5 条
        title = r.get("title", "")
        url = r.get("url", "")
        desc = r.get("description", "")
        lines.append(f"- {title}\n  {url}\n  {desc[:200]}")
    return "\n".join(lines)


def _format_fetch_result(result: dict) -> str:
    content = result.get("content", "")
    title = result.get("title", "")
    if not content:
        return f"（抓取失败: {title}）"
    return f"# {title}\n\n{content[:3000]}"


def search_arxiv_keywords(question: str, llm: Any, max_turns: int = 8) -> str:
    """给定研究问题，生成语义查询描述 + arXiv 搜索关键词（多轮工具调用）

    流程：
    1. 将 system_prompt + user_prompt 发送给 LLM
    2. LLM 可能调用 web_search/web_fetch（通过文本生成工具调用指令）
    3. 我们解析指令，执行工具，将结果注入下一轮对话
    4. 重复直到 LLM 输出最终结果（不再调用工具）

    参数:
        question: 用户的研究问题
        llm: LLM 实例（支持 chat 方法）
        max_turns: 最大对话轮数（防止无限循环）

    返回:
        LLM 最终生成的关键词建议文本
    """
    from llm import Message
    import seacher.prompts as prompts_mod
    SYSTEM_PROMPT = prompts_mod.SYSTEM_PROMPT
    USER_PROMPT_TEMPLATE = prompts_mod.USER_PROMPT_TEMPLATE

    user_prompt = USER_PROMPT_TEMPLATE.format(user_question=question)

    # 对话历史（用于多轮）
    messages = [Message(role="user", content=user_prompt)]

    logger.info("开始多轮搜索关键词生成，问题: %s", question[:50])

    for turn in range(1, max_turns + 1):
        logger.info("第 %d 轮对话", turn)

        # 调用 LLM（带 system prompt）
        reply = llm.chat(
            external_prompt=SYSTEM_PROMPT,
            messages=messages,
        )

        logger.debug("LLM 回复长度: %d", len(reply))

        # 检查是否还有工具调用
        tool_calls = _parse_tool_calls(reply)
        if not tool_calls:
            logger.info("LLM 已停止工具调用，最终结果得到")
            messages.append(Message(role="assistant", content=reply))
            return reply

        # 有工具调用 → 追加 LLM 回复 + 执行工具 + 追加结果
        messages.append(Message(role="assistant", content=reply))

        for tool_name, args_str in tool_calls:
            logger.info("解析到工具调用: %s(%s)", tool_name, args_str)
            tool_result = _execute_tool(tool_name, args_str)
            messages.append(Message(role="user", content=f"[{tool_name} 返回结果]\n{tool_result}"))

    # 超过最大轮数，返回最后一轮结果
    logger.warning("达到最大轮数 %d，终止", max_turns)
    return messages[-1].content if messages else ""