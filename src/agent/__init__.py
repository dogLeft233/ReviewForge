"""Agent 模块——基于 LangChain 的对话 Agent 实现

文件结构:
    __init__.py     — 模块导出
    base.py         — Agent 基类和 AgentBackend Protocol
    chat_agent.py   — 基于 LangChain 的 ChatAgent 实现
    memory.py       — Memory 管理
    tools.py        — 工具注册和管理
    rag.py          — RAG 接口（预留）
"""

from __future__ import annotations

from src.agent.base import AgentBackend
from src.agent.chat_agent import ChatAgent
from src.agent.explorer_orchestrator import ExplorerOrchestrator

__all__ = [
    "AgentBackend",
    "ChatAgent",
    "ExplorerOrchestrator",
]