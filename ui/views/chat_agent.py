"""Chat Agent tab — 对话式问答界面，支持流式输出。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import streamlit as st

from src.agent import ChatAgent
from src.agent.base import AgentBackend
from src.visualizer.schema import VisualizationData


# ─────────────────────────────────────────────────────────────────────────────
# Agent Backend Protocol — 后续可接入真实 LLM / RAG Agent
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class AgentBackend(Protocol):
    """Agent 后端接口协议。

    实现此类以接入真实的 LLM 后端（如 OpenAI、Claude、本地模型等）。
    两种调用方式：
    - ask()      : 同步调用，返回完整回答
    - ask_iter() : 同步生成器，逐块产出回答（用于流式 UI）
    """

    def ask(self, question: str, data: VisualizationData) -> str:
        """同步调用，返回完整回答。"""
        ...

    def ask_iter(self, question: str, data: VisualizationData):
        """同步生成器，逐块产出回答字符串。"""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Chat Agent 后端 — 基于 LangChain 的真实 LLM 对话
# ─────────────────────────────────────────────────────────────────────────────


class ChatAgentBackend:
    """基于 LangChain + Qwen3-8B 的对话 Agent。

    支持：
    - Memory（会话历史）
    - 工具调用（web_search / web_fetch）
    - 流式输出
    """

    def __init__(self, data: VisualizationData) -> None:
        # 从 session_state 获取 session_id
        self._session_id = st.session_state.get("session_id", "default")
        self._data = data

        # 复用 session 中的 ChatAgent 实例，避免每次渲染时重建导致记忆丢失
        if "chat_agent" not in st.session_state:
            st.session_state["chat_agent"] = ChatAgent(session_id=self._session_id)
            # 从 UI 的 chat_messages 加载历史
            messages = st.session_state.get("chat_messages", [])
            st.session_state["chat_agent"].load_history_from_messages(messages)

        self._agent = st.session_state["chat_agent"]

    def ask(self, question: str, data: VisualizationData) -> str:
        """同步调用，返回完整回答。"""
        return "".join(self.ask_iter(question, data))

    def ask_iter(self, question: str, data: VisualizationData):
        """同步生成器，逐块产出回答字符串。"""
        try:
            for chunk in self._agent.ask_iter(question, data):
                yield chunk
        except Exception as e:
            logger = __import__("logging").getLogger(__name__)
            logger.error("ChatAgentBackend error: %s", e)
            yield f"Agent 调用失败：{e}"


# ─────────────────────────────────────────────────────────────────────────────
# 渲染逻辑
# ─────────────────────────────────────────────────────────────────────────────


def render(data: VisualizationData) -> None:
    """渲染对话式问答界面。"""
    st.subheader("💬 Chat Agent")
    st.caption("基于 LangChain + Qwen3-8B 的对话式问答，支持 web_search / web_fetch 工具。")

    # 初始化 session_id
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = __import__("uuid").uuid4().hex[:8]

    # 初始化消息历史
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"]: list[dict[str, str]] = []

    # 渲染历史消息
    for msg in st.session_state["chat_messages"]:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)

    # 用户输入
    if prompt := st.chat_input("请输入您的问题…"):  # noqa: F841
        # 将用户消息追加到历史
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 Agent 后端获取回答（流式输出）
        backend: AgentBackend = ChatAgentBackend(data)
        full_response = ""

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            try:
                # 使用生成器实现流式输出效果
                for chunk in backend.ask_iter(prompt, data):
                    full_response += chunk
                    answer_placeholder.markdown(full_response + "▌")
                # 最终输出（去掉打字光标）
                answer_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Agent 调用失败：{e}")

        # 将助手回答加入历史
        st.session_state["chat_messages"].append({"role": "assistant", "content": full_response})
