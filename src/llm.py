"""LLM 维护器——统一模型调用，集成 web_search / web_fetch 工具"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from src.retrievers.ar5iv import Ar5ivRetriever
from src.logging_config import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMConfig:
    """模型调用配置——由外部传入"""

    model: str
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout_seconds: float = 120.0
    max_retries: int = 2
    max_concurrency: int = 4  # 全局 LLM 并发上限
    thinking: str | None = None  # "low" | "medium" | "high" | "none" | None


# ──────────────────────────────────────────────────────────────
# 工具调用提示词
# ──────────────────────────────────────────────────────────────


def _build_tool_use_prompt() -> str:
    """内置的工具调用提示词——告知 LLM 可用工具及调用规范"""

    return """## 可用工具

### web_search — 网页搜索
```
web_search(query: str, *, count: int = 10, freshness: str | None = None, summary: bool = True) -> list[WebSearchResult]
```
- **query**: 搜索关键词，支持中英文
- **count**: 返回结果数量（1-50）
- **freshness**: 时间过滤 `"oneDay" | "oneWeek" | "oneMonth" | "oneYear"`
- **summary**: 是否包含网页摘要
- **返回**: `WebSearchResult` 列表，每条含 `title` / `url` / `description` / `siteName`

### web_fetch — 页面抓取
```
web_fetch(url: str, *, max_chars: int = 5000, timeout: float = 30.0) -> WebFetchResult
```
- **url**: 目标 URL，支持 arXiv / GitHub / 普通网页
- **max_chars**: 正文最大字符数（默认 5000，避免 context 溢出）
- **timeout**: 请求超时（秒）
- **返回**: `WebFetchResult`，含 `title` / `content`（提取正文） / `status_code`

### explorer_overview — 领域探索第一阶段
```
explorer_overview(topic: str) -> dict[str, Any]
```
- **topic**: 要探索的学术领域或主题
- **返回**: `{overview: str, concepts: list[str], topic: str}`
- 用于当用户询问某领域的基本介绍、发展历史、核心概念时调用

### run_explorer — 完整领域探索（Stage 1+2+3）
```
run_explorer(topic: str) -> str
```
- **topic**: 要探索的研究领域或主题
- **返回**: Markdown 格式的完整探索报告，包含领域概况、经典论文和 Benchmark
- 执行完整的三阶段探索（比 explorer_overview 更全面，但耗时更长）

## 调用规范

1. **需要实时信息时**，优先 `web_search`，获取 URL 后再 `web_fetch` 抓取详情
2. **搜索结果足够时不必再抓取**——`web_search` 的 `description` 字段已含摘要
3. **每次最多抓取 2-3 个 URL**，选最相关的
4. **正文截断到 3000-5000 字**，足够 LLM 理解又不溢出
5. **PDF 链接**：`web_fetch` 对 PDF 返回二进制描述，LLM 可据此决定是否下载

## 调用示例

**搜索 + 抓取流程：**
1. `web_search("LoRA fine-tuning 2025 survey", count=5, freshness="oneYear")`
2. 从结果中选择相关的 URL
3. `web_fetch("https://arxiv.org/abs/2301.00001", max_chars=3000)`
4. 将内容注入 LLM context 继续回答

## 注意事项
- 搜索用 Bocha API（需 BOCHA_API_KEY）
- 抓取受站点速率限制，arXiv 建议间隔 3s 以上
- 所有工具调用错误会在结果中标记，LLM 应尝试降级处理
"""


# ──────────────────────────────────────────────────────────────
# 异常
# ──────────────────────────────────────────────────────────────


class LLMError(Exception):
    """LLM 调用基异常"""


@dataclass(frozen=True)
class APIError(LLMError):
    """API 调用失败（认证、超时、模型错误等）"""

    status_code: int | None = None
    detail: str = ""

    def __str__(self) -> str:
        base = f"APIError(status={self.status_code})"
        if self.detail:
            base += f", detail={self.detail}"
        return base


@dataclass(frozen=True)
class RateLimitError(LLMError):
    """触发速率限制"""


# ──────────────────────────────────────────────────────────────
# 消息格式
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Message:
    """对话消息"""

    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass(slots=True)
class ToolCallResult:
    """工具调用结果——注入回 LLM"""

    tool_name: str
    args: dict[str, Any]
    raw_result: Any
    error: str = ""


# ──────────────────────────────────────────────────────────────
# 主类
# ──────────────────────────────────────────────────────────────


class LLM:
    """LLM 统一调用器——模型信息由外部传入，内置工具调用 prompt

    用法:
        llm = LLM(
            api_key="sk-...",
            model="Qwen/Qwen3-8B",
            base_url="https://api.siliconflow.cn/v1",
        )
        reply = llm.chat("解释 LoRA 的原理")

        # 带外部 prompt（会拼在工具调用 prompt 之前）
        reply = llm.chat(
            external_prompt="你是一个论文评审专家，专注于创新性评估",
            messages=[Message(role="user", content="评价这篇论文...")],
        )
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        *,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        max_concurrency: int = 4,
        thinking: str | None = None,  # 思考深度控制
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not model:
            raise ValueError("model is required")

        self._cfg = LLMConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_concurrency=max_concurrency,
            thinking=thinking,
        )
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._tool_prompt = _build_tool_use_prompt()
        self._ar5iv = Ar5ivRetriever()

        logger.debug("LLM init: model=%s base=%s", model, self._base_url)

    # ── 对话接口 ────────────────────────────────────────────

    def chat(
        self,
        external_prompt: str = "",
        messages: list[Message] | None = None,
        *,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: str | None = None,  # 思考深度控制
    ) -> str:
        """发送对话请求，返回模型文本回复

        参数:
            external_prompt: 外部传入的初始 prompt（拼在工具调用 prompt 之前）
            messages: 对话历史（用于多轮对话）
            system_prompt: 额外系统 prompt（追加在工具调用 prompt 之后）
            temperature: 采样温度（None=默认）
            max_tokens: 最大 token 数（None=默认）

        返回:
            模型回复文本
        """
        t0 = time.perf_counter()

        effective_temp = temperature if temperature is not None else self._cfg.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self._cfg.max_tokens

        payload, response_url = self._build_payload(
            external_prompt=external_prompt,
            messages=messages,
            system_prompt=system_prompt,
            thinking=thinking,
        )

        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                tt = time.perf_counter()
                reply_text = self._send_request(
                    payload=payload,
                    url=response_url,
                    timeout=self._cfg.timeout_seconds,
                )
                elapsed = time.perf_counter() - t0
                logger.info(
                    "[LLM] ✓ chat done in %.2fs (attempt %d, req %.0fms)",
                    elapsed, attempt, (time.perf_counter() - tt) * 1000,
                )
                return reply_text
            except RateLimitError:
                wait = 2.0 * attempt
                logger.warning("LLM rate-limited, waiting %.1fs", wait)
                time.sleep(wait)
            except APIError as e:
                if e.status_code in (429, 500, 502, 503, 504):
                    wait = 2.0 * attempt
                    logger.warning("LLM %s, retrying in %.1fs", e, wait)
                    time.sleep(wait)
                else:
                    raise

        raise APIError(
            status_code=None,
            detail=f"All {self._cfg.max_retries} attempts failed",
        )

    def chat_with_tools(
        self,
        external_prompt: str = "",
        messages: list[Message] | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_turns: int = 5,
    ) -> tuple[str, list[ToolCallResult]]:
        """对话 + 工具调用（支持多轮 tool calling）

        实现 ReAct 循环：LLM 返回 tool_calls 时实际执行，并将结果注入下一轮对话。
        达到 max_turns 限制时停止。

        参数:
            external_prompt: 外部传入的初始 prompt
            messages: 对话历史
            tools: 工具规格列表（OpenAI function calling 格式）
            system_prompt: 额外系统 prompt
            temperature: 采样温度
            max_tokens: 最大 token 数
            max_turns: 最大工具调用轮数（防止无限循环）

        返回:
            (最终回复文本, 已执行的工具调用结果列表)
        """
        import json

        # 构建初始消息列表
        all_messages: list[dict[str, Any]] = []
        if external_prompt:
            all_messages.append({"role": "system", "content": external_prompt})
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        if messages:
            for msg in messages:
                all_messages.append({"role": msg.role, "content": msg.content})

        effective_temp = temperature if temperature is not None else self._cfg.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self._cfg.max_tokens

        turn = 0
        tool_results: list[ToolCallResult] = []

        while turn < max_turns:
            turn += 1

            # 构建请求 payload
            payload: dict[str, Any] = {
                "model": self._cfg.model,
                "messages": all_messages,
                "temperature": effective_temp,
                "max_tokens": effective_max_tokens,
            }
            if tools:
                payload["tools"] = tools

            url = f"{self._base_url}/chat/completions"

            # 发送请求
            t0 = time.perf_counter()
            response_text = self._send_request(payload, url, self._cfg.timeout_seconds)
            elapsed = time.perf_counter() - t0
            logger.info("[LLM] chat_with_tools turn %d done in %.2fs", turn, elapsed)

            # 解析 tool_calls
            parsed = self._parse_tool_calls_from_response(response_text)

            if not parsed:
                # 没有工具调用，直接返回
                return response_text, tool_results

            # 执行工具调用
            for tc in parsed:
                tool_name = tc["function"]["name"]
                arguments = tc["function"]["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                logger.info("[Tool Call] %s(%s)", tool_name, arguments)

                tool_result_str = self._execute_tool(tool_name, arguments)
                tool_results.append(
                    ToolCallResult(
                        tool_name=tool_name,
                        args=arguments,
                        raw_result=tool_result_str,
                    )
                )

                # 重新构造带 tool_calls 的 assistant 消息（不依赖 response_text 原始字符串）
                assistant_msg = {"role": "assistant"}
                # 提取 reasoning_content（DeepSeek 多轮对话必须回传）
                try:
                    data = json.loads(response_text)
                    if isinstance(data, dict) and data.get("reasoning_content"):
                        assistant_msg["reasoning_content"] = data["reasoning_content"]
                except (json.JSONDecodeError, ValueError):
                    pass
                # 从 parsed 中重建 function_call 结构
                if tc.get("id"):
                    assistant_msg["tool_calls"] = [
                        {"id": t["id"], "type": "function", "function": t["function"]}
                        for t in parsed
                    ]
                all_messages.append(assistant_msg)
                # 将工具结果追加为 tool 消息
                all_messages.append({
                    "role": "tool",
                    "content": tool_result_str,
                    "tool_call_id": tc.get("id", ""),
                })

        # 达到最大轮次，返回最后结果
        logger.warning("[LLM] max_turns (%d) reached, returning last response", max_turns)
        return response_text, tool_results

    def _parse_tool_calls_from_response(self, response_text: str) -> list[dict[str, Any]]:
        """从 LLM 响应文本中解析 tool_calls（JSON 块）"""
        import json
        import re

        # 尝试从响应中提取 ```json ... ``` 块
        json_blocks = re.findall(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if not json_blocks:
            # 检查直接 JSON 解析（MiniMax/SiliconFlow 返回格式）
            try:
                data = json.loads(response_text)
                if isinstance(data, dict) and "tool_calls" in data:
                    return data["tool_calls"]
                # 也支持直接返回 tool_calls 列表（某些 API 格式）
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "function" in data[0]:
                    return data
            except json.JSONDecodeError:
                pass

        for block in json_blocks:
            try:
                data = json.loads(block)
                if isinstance(data, dict) and "tool_calls" in data:
                    return data["tool_calls"]
            except json.JSONDecodeError:
                continue

        # MiniMax 等模型使用 [TOOL_CALL]...[/TOOL_CALL] 格式
        tool_call_blocks = re.findall(
            r'\[TOOL_CALL\](.*?)\[/TOOL_CALL\]', response_text, re.DOTALL
        )
        all_parsed_calls: list[dict[str, Any]] = []
        import uuid
        for block in tool_call_blocks:
            # 提取 tool name
            tool_match = re.search(r'^\s*\{?\s*tool\s*[=:]\s*"([^"]+)"', block, re.MULTILINE)
            if not tool_match:
                tool_match = re.search(r'^\s*\{?\s*"tool"\s*[=:]\s*"([^"]+)"', block, re.MULTILINE)
            if not tool_match:
                continue
            tool_name = tool_match.group(1)

            # 提取 args（支持 --key "value" 和 --key value 格式）
            args_str = block[tool_match.end():]
            args = {}
            # 匹配 --key "value" 或 --key 'value'
            kv_matches = re.findall(r'--(\w+)\s+"([^"]*)"', args_str)
            for k, v in kv_matches:
                try:
                    args[k] = json.loads(v)
                except json.JSONDecodeError:
                    args[k] = v
            # 匹配 --key value（无引号）
            kv_matches2 = re.findall(r'--(\w+)\s+([^(?:\s|"|\'|,)])+', args_str)
            for k, v in kv_matches2:
                v = v.strip().rstrip(',')
                try:
                    args[k] = json.loads(v)
                except json.JSONDecodeError:
                    args[k] = v

            tc_id = uuid.uuid4().hex[:24]
            all_parsed_calls.append({
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False)
                }
            })

        if all_parsed_calls:
            return all_parsed_calls

        return []

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """根据工具名执行对应工具，返回格式化的结果字符串"""
        import json

        if tool_name == "web_search":
            from src.tools.web_search import web_search as ws
            logger.debug("[web_search] query=%s count=%s freshness=%s",
                         arguments["query"], arguments.get("count", 10), arguments.get("freshness"))
            results = ws(
                query=arguments["query"],
                count=arguments.get("count", 10),
                freshness=arguments.get("freshness"),
            )
            if not results:
                return "No results found."
            lines = []
            for r in results:
                lines.append(f"- **{r.title}** ({r.site_name or 'unknown'})")
                lines.append(f"  URL: {r.url}")
                if r.description:
                    lines.append(f"  Summary: {r.description}")
                lines.append("")
            return "\n".join(lines)

        elif tool_name == "web_fetch":
            from src.tools.web_fetch import web_fetch as wf
            try:
                result = wf(
                    url=arguments["url"],
                    max_chars=arguments.get("max_chars", 5000),
                    timeout=arguments.get("timeout", 30.0),
                )
                if result.status_code != 200:
                    return f"Failed to fetch {arguments['url']}: status {result.status_code}"
                return f"Title: {result.title or arguments['url']}\nContent: {result.content[:3000]}"
            except Exception as e:
                return f"Failed to fetch {arguments['url']}: {type(e).__name__}: {e}"

        elif tool_name == "run_explorer":
            from src.agent.skills.core_skill import run_explorer
            return run_explorer(topic=arguments["topic"])

        elif tool_name == "run_explorer_async":
            import threading
            from src.agent.skills.core_skill import run_explorer_async
            topic = arguments["topic"]
            thread = threading.Thread(target=run_explorer_async, kwargs={"topic": topic})
            thread.daemon = True
            thread.start()
            return f"⏳ 领域探索任务已启动（topic: {topic}），UI 将自动更新进度..."

        elif tool_name == "run_searcher":
            from src.agent.skills.core_skill import run_searcher
            return run_searcher(topic=arguments["topic"])

        elif tool_name == "run_pipeline":
            from src.agent.skills.core_skill import run_pipeline
            return run_pipeline(topic=arguments["topic"])

        elif tool_name == "run_pipeline_async":
            import threading
            from src.agent.skills.core_skill import run_pipeline_async
            topic = arguments["topic"]
            thread = threading.Thread(target=run_pipeline_async, kwargs={"topic": topic})
            thread.daemon = True
            thread.start()
            return f"⏳ 完整流水线任务已启动（topic: {topic}），后台执行中..."

        else:
            return f"Unknown tool: {tool_name}"

    # ── 构建请求 ─────────────────────────────────────────────

    def _build_payload(
        self,
        external_prompt: str,
        messages: list[Message] | None,
        system_prompt: str,
        thinking: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        """拼接 system prompt（含工具说明）+ 用户消息，返回 payload 和 URL"""

        system_parts: list[str] = []
        if external_prompt:
            system_parts.append(external_prompt)

        system_parts.append(self._tool_prompt)

        if system_prompt:
            system_parts.append(system_prompt)

        system_content = "\n\n".join(system_parts)

        # 构建 messages
        all_messages: list[dict[str, str]] = []
        if system_content:
            all_messages.append({"role": "system", "content": system_content})

        if messages:
            for msg in messages:
                all_messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": self._cfg.model,
            "messages": all_messages,
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }

        # 思考深度控制：优先使用调用时传入的值，其次使用配置默认值
        # SiliconFlow 使用 thinking_budget 参数
        effective_thinking = thinking if thinking is not None else self._cfg.thinking
        if effective_thinking is not None:
            payload["thinking_budget"] = int(effective_thinking)

        logger.debug("[LLM] Request payload: model=%s, messages_count=%d, temp=%.2f",
                     self._cfg.model, len(all_messages), self._cfg.temperature)

        url = f"{self._base_url}/chat/completions"
        return payload, url

    def _send_request(
        self,
        payload: dict[str, Any],
        url: str,
        timeout: float,
    ) -> str:
        """实际发送请求，返回回复文本"""
        tt = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.post(url, json=payload, headers=headers)
        req_ms = (time.perf_counter() - tt) * 1000
        logger.debug("[LLM] HTTP round-trip %.0fms", req_ms)

        match resp.status_code:
            case 200:
                data = resp.json()
                logger.debug("[LLM] Response raw: %s", data)
                choices = data.get("choices", [])
                if not choices:
                    raise APIError(status_code=200, detail="No choices in response")
                msg = choices[0].get("message", {})
                # 如果是 tool_calls 响应，content 为空，直接返回整个 msg JSON
                # 供 chat_with_tools 的 ReAct 循环解析 tool_calls
                if msg.get("tool_calls"):
                    import json
                    return json.dumps(msg)  # 返回完整 message 对象
                return msg.get("content", "")
            case 401 | 403:
                raise APIError(status_code=resp.status_code, detail=resp.text[:200])
            case 429:
                raise RateLimitError(f"LLM rate limited: {resp.text[:200]}")
            case 500 | 502 | 503 | 504:
                raise APIError(status_code=resp.status_code, detail=resp.text[:200])
            case _:
                raise APIError(status_code=resp.status_code, detail=resp.text[:300])

    # ── 上下文管理 ───────────────────────────────────────────

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    # ── ar5iv 全文获取 ───────────────────────────────────────

    def fetch_ar5iv_full_text(self, arxiv_id: str) -> str:
        """获取指定 arXiv 论文的 HTML 全文（通过 ar5iv.org）

        Args:
            arxiv_id: 形如 "2301.00001" 或 "2301.00001v2"
        Returns:
            HTML 正文原始字符串
        """
        try:
            return self._ar5iv.fetch_full_text(arxiv_id)
        except Exception as e:
            logger.warning("ar5iv fetch failed for %s: %s", arxiv_id, e)
            return ""
