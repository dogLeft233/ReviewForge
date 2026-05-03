"""LLM 客户端适配器——对外部 API 的统一封装

设计原则：
- 适配 OpenAI 兼容的 API（SiliconFlow / OpenAI / 任意兼容端点）
- 支持聊天补全与函数调用
- 支持 JSON 提取（不依赖 response_format 参数）
- 内置重试与超时控制
"""

import json
import logging
import re
import time
from typing import Any

from os import environ

from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM 调用客户端"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.api_key = api_key or settings.llm_api_key or environ.get("LLM_API_KEY", "")
        self.base_url = base_url or settings.llm_base_url
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = timeout or settings.llm_timeout_seconds
        self.max_retries = max_retries or settings.llm_max_retries

        if not self.api_key:
            raise ValueError(
                "LLM API key not configured. "
                "Set LLM_API_KEY environment variable or pass api_key to constructor."
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    # ── 公开方法 ──

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> str:
        """聊天补全，返回文本内容

        Args:
            messages: 消息列表 [{"role": "...", "content": "..."}]
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认 max_tokens
            tools: 工具定义列表（可选）
            tool_choice: 工具选择策略（可选）

        Returns:
            模型回复文本。如果是工具调用，返回 JSON 格式的工具调用摘要。
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature if temperature is not None else self.temperature,
                    "max_tokens": max_tokens or self.max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice

                resp = self._client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message

                # 工具调用优先
                if msg.tool_calls:
                    return json.dumps(
                        {
                            "tool_calls": [
                                {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                }
                                for tc in msg.tool_calls
                            ]
                        },
                        ensure_ascii=False,
                    )

                return msg.content or ""

            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt, self.max_retries, e,
                )
                if attempt < self.max_retries:
                    time.sleep(1.5 * attempt)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries: {last_error}"
        )

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """聊天补全并提取 JSON 输出

        不使用 response_format（SiliconFlow 上该参数有兼容问题），
        而是用 system prompt 引导模型输出 JSON，然后从文本中提取。
        """
        content = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._extract_json(content)

    def chat_with_function(
        self,
        messages: list[dict[str, str]],
        function_name: str,
        function_description: str,
        parameters_schema: dict[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """使用函数调用模式获取结构化输出

        比 chat_json 更可靠，因为函数调用的参数解析由 API 保证为有效 JSON。
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": function_description,
                    "parameters": parameters_schema,
                },
            }
        ]

        content = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": function_name},
            },
        )

        # chat 方法对工具调用返回了 JSON 字符串
        try:
            parsed = json.loads(content)
            calls = parsed.get("tool_calls", [])
            if calls:
                return json.loads(calls[0]["arguments"])
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

        # 如果 chat 返回了普通文本，尝试直接提取 JSON
        return self._extract_json(content)

    # ── 内部方法 ──

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        """从 LLM 回复中提取 JSON 对象

        处理模型在 JSON 前后添加解释文字的情况，
        以及 SiliconFlow 上 response_format 返回空对象的问题。
        """
        if not content or not content.strip():
            raise ValueError("Empty LLM response, cannot extract JSON")

        # 1. 尝试直接解析整个内容
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass

        # 2. 尝试找到第一个 { ... } 或 [ ... ] 块
        for bracket in ("{}", "[]"):
            start_bracket, end_bracket = bracket[0], bracket[1]
            depth = 0
            json_start = -1
            for i, ch in enumerate(content):
                if ch == start_bracket:
                    if depth == 0:
                        json_start = i
                    depth += 1
                elif ch == end_bracket:
                    depth -= 1
                    if depth == 0 and json_start >= 0:
                        candidate = content[json_start:i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            continue

        # 3. 尝试正则匹配
        json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Cannot extract valid JSON from LLM response "
            f"(content length={len(content)}, preview={content[:200]!r})"
        )
