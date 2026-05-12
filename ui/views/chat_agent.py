"""Chat Agent tab — 对话式问答界面，支持流式输出。"""

from __future__ import annotations

import json
import os
import time
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

    # 初始化异步任务列表
    if "pending_tasks" not in st.session_state:
        st.session_state["pending_tasks"] = []

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

                # 检测异步任务响应，注册到 pending_tasks
                import re
                if "⏳ 领域探索任务已启动" in full_response:
                    match = re.search(r"topic:\s*([^（）]+)", full_response)
                    topic = match.group(1).strip() if match else prompt
                    st.session_state["pending_tasks"].append({
                        "type": "run_explorer_async",
                        "topic": topic,
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                elif "⏳ BFS 搜索任务已启动" in full_response:
                    match = re.search(r"question:\s*([^（）]+)", full_response)
                    topic = match.group(1).strip() if match else prompt
                    st.session_state["pending_tasks"].append({
                        "type": "bfs_search_async",
                        "topic": topic,
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
            except Exception as e:
                st.error(f"Agent 调用失败：{e}")

        # 将助手回答加入历史
        st.session_state["chat_messages"].append({"role": "assistant", "content": full_response})

        # ── 异步任务状态轮询 ──────────────────────────────────────────────────────
        pending = st.session_state.get("pending_tasks", [])
        if pending:
            st.divider()
            st.subheader("📋 后台任务状态")

            still_pending = []
            for task in pending:
                topic = task["topic"]
                task_dir = f"tmp/{topic}"
                status_file = f"{task_dir}/task_status.json"

                if os.path.exists(status_file):
                    with open(status_file, encoding="utf-8") as f:
                        status = json.load(f)

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"**{status.get('message', '处理中...')}**")
                        st.progress(status.get("progress", 0), text=f"进度 {int(status.get('progress', 0)*100)}%")
                    with col2:
                        st.caption(f"⏱ {status.get('started_at', '')}")

                    if status.get("status") == "completed":
                        st.success(f"✅ 完成！正在加载结果到 UI...")
                        _load_task_result(task)
                    elif status.get("status") == "failed":
                        st.error(f"❌ 失败: {status.get('message', '未知错误')}")
                    else:
                        still_pending.append(task)
                else:
                    st.info(f"⏳ 任务初始化中: {topic}")
                    still_pending.append(task)

            st.session_state["pending_tasks"] = still_pending


def _load_task_result(task: dict) -> None:
    """加载任务结果到 session_state（供其他 Tab 使用）"""
    topic = task["topic"]
    if task["type"] == "run_explorer_async":
        result_file = f"tmp/{topic}/step1_explorer_done.json"
        if os.path.exists(result_file):
            with open(result_file, encoding="utf-8") as f:
                data = json.load(f)
            st.session_state["visualization_data"] = data
            st.rerun()
    elif task["type"] == "bfs_search_async":
        result_file = f"tmp/{topic}/step2_searcher_done.json"
        if os.path.exists(result_file):
            with open(result_file, encoding="utf-8") as f:
                data = json.load(f)
            st.session_state["bfs_search_data"] = data
            st.rerun()
