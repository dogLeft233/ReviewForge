"""LLM 维护器——统一模型调用，集成 web_search / web_fetch 工具"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


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
        )
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._tool_prompt = _build_tool_use_prompt()

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
        effective_temp = temperature if temperature is not None else self._cfg.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self._cfg.max_tokens

        payload, response_url = self._build_payload(
            external_prompt=external_prompt,
            messages=messages,
            system_prompt=system_prompt,
        )

        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                reply_text = self._send_request(
                    payload=payload,
                    url=response_url,
                    timeout=self._cfg.timeout_seconds,
                )
                logger.debug("LLM chat: attempt %d ok", attempt)
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
    ) -> tuple[str, list[ToolCallResult]]:
        """对话 + 工具调用（future 支持，当前版本等效 chat）

        当前实现：返回 chat 结果和空工具列表。
        工具调用框架待接入。
        """
        reply = self.chat(
            external_prompt=external_prompt,
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return reply, []

    # ── 构建请求 ─────────────────────────────────────────────

    def _build_payload(
        self,
        external_prompt: str,
        messages: list[Message] | None,
        system_prompt: str,
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

        url = f"{self._base_url}/chat/completions"
        return payload, url

    def _send_request(
        self,
        payload: dict[str, Any],
        url: str,
        timeout: float,
    ) -> str:
        """实际发送请求，返回回复文本"""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.post(url, json=payload, headers=headers)

        match resp.status_code:
            case 200:
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise APIError(status_code=200, detail="No choices in response")
                return choices[0]["message"]["content"]
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
