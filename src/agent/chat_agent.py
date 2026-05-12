"""基于 LangChain 的对话 Agent 实现"""

from __future__ import annotations

import sys
from typing import Any, Generator, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import CallbackManager
from langchain_openai import ChatOpenAI

from src.agent.base import AgentBackend
from src.agent.memory import MemoryManager, SessionChatHistory
from src.agent.rag import build_rag_context
from src.logging_config import get_logger
from src.config import settings
from src.llm import LLM, LLMConfig
from src.tools.web_search import web_search, WebSearchResult
from src.tools.web_fetch import web_fetch, WebFetchResult
from src.visualizer.schema import VisualizationData

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# LangChain Tool 定义（基于项目已有工具）
# ──────────────────────────────────────────────────────────────


def _make_web_search_tool() -> Any:
    """创建 web_search LangChain Tool"""
    from langchain_core.tools import StructuredTool

    def web_search_tool(
        query: str,
        count: int = 10,
        freshness: str | None = None,
    ) -> str:
        """Web search using Bocha API. Returns formatted search results."""
        try:
            results: list[WebSearchResult] = web_search(
                query, count=count, freshness=freshness, summary=True
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
        except Exception as e:
            logger.error("web_search failed: %s", e)
            return f"Search failed: {e}"

    return StructuredTool(
        name="web_search",
        description="Search the web for information. Use when you need to find recent facts or data.",
        func=web_search_tool,
        args_schema={
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Number of results (1-50)", "default": 10},
            "freshness": {
                "type": "string",
                "description": "Time filter: oneDay, oneWeek, oneMonth, oneYear",
                "default": None,
            },
        },
    )


def _make_web_fetch_tool() -> Any:
    """创建 web_fetch LangChain Tool"""
    from langchain_core.tools import StructuredTool

    def web_fetch_tool(
        url: str,
        max_chars: int = 5000,
        timeout: float = 30.0,
    ) -> str:
        """Fetch web page content from a URL."""
        try:
            result: WebFetchResult = web_fetch(url, max_chars=max_chars, timeout=timeout)
            if result.status_code != 200:
                return f"Failed to fetch {url}: status {result.status_code}"
            lines = [
                f"Title: {result.title or result.url}",
                f"URL: {result.url}",
                f"Content ({len(result.content)} chars):",
                result.content[:3000],
            ]
            if len(result.content) > 3000:
                lines.append(f"... (truncated, total {len(result.content)} chars)")
            return "\n".join(lines)
        except Exception as e:
            logger.error("web_fetch failed for %s: %s", url, e)
            return f"Fetch failed: {e}"

    return StructuredTool(
        name="web_fetch",
        description="Fetch web page content from a URL. Use after web_search to get details.",
        func=web_fetch_tool,
        args_schema={
            "url": {"type": "string", "description": "Target URL"},
            "max_chars": {"type": "integer", "description": "Max characters", "default": 5000},
            "timeout": {"type": "number", "description": "Timeout in seconds", "default": 30.0},
        },
    )


# ──────────────────────────────────────────────────────────────
# 自定义 LLM 封装（适配 SiliconFlow API）
# ──────────────────────────────────────────────────────────────


class SiliconFlowLLM:
    """基于项目 LLM 的自定义 LangChain LLM 封装。

    将 src.llm.LLM 包装为 LangChain Runnable 接口，
    支持 streaming 和 tool calling。
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
        self._llm = LLM(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        # 流式回调
        self._stream_handlers: list[Any] = []

    def _convert_messages(self, messages: list[HumanMessage | SystemMessage]) -> list[dict[str, str]]:
        """将 LangChain messages 转换为 LLM Message 列表"""
        result = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            if isinstance(msg, SystemMessage):
                role = "system"
            result.append({"role": role, "content": msg.content})
        return result

    def invoke(
        self,
        input: str | list[dict[str, str]] | list[HumanMessage | SystemMessage],
        config: dict[str, Any] | None = None,
    ) -> Any:
        """同步调用，返回 AIMessage"""
        from langchain_core.messages import AIMessage

        # 构建消息列表
        if isinstance(input, str):
            msgs = [{"role": "user", "content": input}]
        elif isinstance(input, list) and input and isinstance(input[0], dict):
            msgs = input
        else:
            msgs = self._convert_messages(input)

        # 调用 LLM
        response = self._llm.chat(messages=[self._llm.Message(role=m["role"], content=m["content"]) for m in msgs])
        return AIMessage(content=response)

    def stream(
        self,
        input: str | list[dict[str, str]] | list[HumanMessage | SystemMessage],
        config: dict[str, Any] | None = None,
    ) -> Generator[str, None, None]:
        """流式调用，逐块产出字符串"""
        from langchain_core.messages import AIMessageChunk

        # 构建消息列表
        if isinstance(input, str):
            msgs = [{"role": "user", "content": input}]
        elif isinstance(input, list) and input and isinstance(input[0], dict):
            msgs = input
        else:
            msgs = self._convert_messages(input)

        # 使用 httpx 流式请求
        import httpx
        headers = {
            "Authorization": f"Bearer {self._llm._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._llm._cfg.model,
            "messages": msgs,
            "temperature": self._llm._cfg.temperature,
            "max_tokens": self._llm._cfg.max_tokens,
            "stream": True,
        }
        url = f"{self._llm._base_url}/chat/completions"

        with httpx.Client(timeout=self._llm._cfg.timeout_seconds, follow_redirects=True) as client:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    import json
                    try:
                        data = json.loads(data_str)
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except Exception:
                        pass

    @property
    def _type(self) -> str:
        return "siliconflow"

    def bind_tools(self, tools: list[Any]) -> "SiliconFlowLLM":
        """绑定工具（预留，当前通过 system prompt 实现）"""
        return self


# ──────────────────────────────────────────────────────────────
# ChatAgent 实现
# ──────────────────────────────────────────────────────────────


class ChatAgent:
    """基于 LangChain 的对话 Agent。

    实现 AgentBackend Protocol，支持：
    - Memory（会话历史）
    - 工具调用（web_search / web_fetch）
    - RAG 预留接口
    - 流式输出

    用法:
        agent = ChatAgent()
        answer = agent.ask("解释 LoRA", data)
        for chunk in agent.ask_iter("解释 LoRA", data):
            print(chunk, end="")
    """

    def __init__(
        self,
        session_id: str = "default",
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        # 加载配置
        self._api_key = getattr(settings, "llm_api_key", "")
        self._model = getattr(settings, "llm_model", "Qwen/Qwen3-8B")
        self._base_url = getattr(settings, "llm_base_url", "https://api.siliconflow.cn/v1")
        self._temperature = temperature or getattr(settings, "llm_temperature", 0.1)
        self._max_tokens = max_tokens or getattr(settings, "llm_max_tokens", 2000)
        self._timeout = getattr(settings, "llm_timeout_seconds", 120.0)
        self._max_retries = getattr(settings, "llm_max_retries", 2)

        self._session_id = session_id
        self._system_prompt = system_prompt or self._default_system_prompt()

        # Memory 管理器
        self._memory = MemoryManager(session_id=session_id)

        # LLM
        self._llm = SiliconFlowLLM(
            api_key=self._api_key,
            model=self._model,
            base_url=self._base_url,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout_seconds=self._timeout,
            max_retries=self._max_retries,
        )

        # LangChain Agent
        self._agent_executor: Any = None

        logger.info(
            "ChatAgent init: model=%s session=%s",
            self._model,
            session_id,
        )

    def _default_system_prompt(self) -> str:
        return """你是一个学术研究领域的 AI 助手，专注于帮助用户探索和理解学术论文、方法和趋势。
你可以使用以下工具：
- web_search: 搜索最新信息，用于实时查询
- web_fetch: 获取网页详情
- explorer_overview: 探索给定学术领域的概况、核心问题和主流方法。当用户询问某个领域的基本介绍、发展历史、核心概念时使用。
回答时结合搜索结果和已有知识，提供准确、有条理的回答。
始终用中文回答，除非用户用英文提问。"""

    def _build_system_prompt(self, data: VisualizationData) -> str:
        """构建完整的 system prompt（含 VisualizationData 上下文）"""
        parts = [self._system_prompt]

        # 添加 RAG 上下文（预留）
        rag_context = build_rag_context("", top_k=5)
        if rag_context:
            parts.append(f"\n\n## 知识库检索结果\n{rag_context}")

        # 添加 VisualizationData 摘要
        if data.overview.definition:
            parts.append(f"\n\n## 领域概述\n{data.overview.definition}")
        if data.overview.core_questions:
            parts.append(f"\n\n## 核心问题\n" + "\n".join(f"- {q}" for q in data.overview.core_questions[:5]))

        return "\n".join(parts)

    def _setup_agent(self, data: VisualizationData) -> Any:
        """设置 LangChain Agent（带工具）"""
        from langchain_coreagents import AgentExecutor
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool

        # 创建工具
        web_search_t = _make_web_search_tool()
        web_fetch_t = _make_web_fetch_tool()
        tools = [web_search_t, web_fetch_t]

        # 使用 ReAct Agent
        from langchain.agents import create_react_agent, AgentType
        from langchain import LangChain

        # 简单实现：直接调用 LLM + tools
        # 由于 Qwen3-8B 可能不支持 function calling，使用 system prompt 注入工具
        return self._llm, tools

    def ask(self, question: str, data: VisualizationData) -> str:
        """同步调用，返回完整回答。"""
        logger.debug("Received question: %s", question[:100])
        return "".join(self.ask_iter(question, data))

    def ask_iter(self, question: str, data: VisualizationData):
        """同步生成器，逐块产出回答字符串。"""
        # 构建 system prompt
        system_content = self._build_system_prompt(data)

        # 获取对话历史
        history = self._memory.get_history_for_chain(self._session_id)

        # 构建消息
        all_messages: list[dict[str, str]] = []
        if system_content:
            all_messages.append({"role": "system", "content": system_content})
        for msg in history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            all_messages.append({"role": role, "content": msg.content})
        all_messages.append({"role": "user", "content": question})

        # 先尝试带工具调用（通过 LLM）
        from src.llm import Message

        msgs_for_llm = [Message(role=m["role"], content=m["content"]) for m in all_messages]
        response, _ = self._llm._llm.chat_with_tools(
            messages=msgs_for_llm,
            tools=self._get_tools_spec(),
            max_turns=3,
        )

        # 更新 Memory
        self._memory.add_user_message(question, self._session_id)
        self._memory.add_ai_message(response, self._session_id)

        # 流式产出
        chunk_size = 15
        for i in range(0, len(response), chunk_size):
            yield response[i:i + chunk_size]

    def _get_tools_spec(self) -> list[dict[str, Any]]:
        """获取工具规格（用于 LLM chat_with_tools）"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "count": {"type": "integer", "description": "Number of results (1-50)", "default": 10},
                            "freshness": {"type": "string", "description": "Time filter", "default": None},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch web page content from a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "Target URL"},
                            "max_chars": {"type": "integer", "description": "Max characters", "default": 5000},
                            "timeout": {"type": "number", "description": "Timeout in seconds", "default": 30.0},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "explorer_overview",
                    "description": "探索给定学术领域的概况、核心问题和主流方法。当用户询问某个领域的基本介绍、发展历史、核心概念时使用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "要探索的学术领域或主题"},
                        },
                        "required": ["topic"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_explorer",
                    "description": "执行领域探索，获取研究领域的基本概况、核心概念、经典论文和 Benchmark。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "要探索的研究领域或主题"},
                        },
                        "required": ["topic"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_explorer_async",
                    "description": "异步执行领域探索，立即返回任务ID，后台执行不阻塞，阶段进度实时更新，完成自动写入 step1_explorer_done.json 供其他 Tab 加载。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "要探索的研究领域或主题"},
                        },
                        "required": ["topic"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "bfs_search",
                    "description": "深度论文发现工具，使用 BFS（广度优先搜索）发现某领域的所有重要论文及其引用关系。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "研究问题或主题"},
                            "expand_layers": {"type": "integer", "description": "BFS 扩展层数，0=只用搜索不扩展引文，1=搜索+一层引文，2=两层（默认2）", "default": 2},
                            "search_papers_count": {"type": "integer", "description": "每个搜索词取多少篇论文", "default": 20},
                        },
                        "required": ["question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "bfs_search_async",
                    "description": "异步执行 BFS 深度论文发现，立即返回任务ID，后台执行不阻塞，完成自动写入 step2_searcher_done.json 供其他 Tab 加载。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "研究问题或主题"},
                            "expand_layers": {"type": "integer", "description": "BFS 扩展层数，0=只用搜索，1=一层引文，2=两层（默认2）", "default": 2},
                            "search_papers_count": {"type": "integer", "description": "每个搜索词取多少篇论文", "default": 20},
                        },
                        "required": ["question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "multi_source_search",
                    "description": "多源检索工具，同时搜索 GitHub（项目）、HuggingFace（模型+数据集）和 arXiv（论文）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "研究问题或技术主题"},
                        },
                        "required": ["question"],
                    },
                },
            },
        ]

    def load_history_from_messages(self, messages: list[dict[str, str]]) -> None:
        """从消息字典列表加载历史到 MemoryManager。

        用于从 Streamlit session_state 恢复历史。
        """
        from langchain_core.messages import HumanMessage, AIMessage

        for msg in messages:
            if msg["role"] == "user":
                self._memory.add_user_message(msg["content"], self._session_id)
            elif msg["role"] == "assistant":
                self._memory.add_ai_message(msg["content"], self._session_id)

    def clear_history(self) -> None:
        """清除当前会话历史"""
        self._memory.clear(self._session_id)
