"""Agent 基类和接口定义"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.visualizer.schema import VisualizationData


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
