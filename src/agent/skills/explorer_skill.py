"""Explorer Stage1 — 领域探索 Skill for ChatAgent

将 ExplorerAgent._run_stage1() 封装为 LangChain StructuredTool，
供 ChatAgent 在对话中调用。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from src.explorer import ExplorerAgent
from src.llm import LLM
from src.logging_config import get_logger

logger = get_logger(__name__)


def make_explorer_stage1_tool(llm: LLM) -> StructuredTool:
    """创建 explorer_overview LangChain Tool"""

    def explorer_overview(topic: str) -> dict[str, Any]:
        """执行领域探索第一阶段，返回领域概况和核心概念。

        用于用户询问某个领域的基本情况时调用。
        后续会自动触发 Stage2（经典论文）和 Stage3（前沿 Benchmark）。

        Args:
            topic: 要探索的学术领域或主题
        Returns:
            包含 overview 和 concepts 的字典
        """
        logger.info("[explorer_overview] 开始探索领域: %s", topic)
        explorer = ExplorerAgent(llm=llm)
        result = explorer._run_stage1(topic)
        logger.info("[explorer_overview] 完成，探索到 %d 个概念", len(result.get("concepts", [])))
        return {
            "overview": result["overview"],
            "concepts": result["concepts"],
            "topic": topic,
        }

    return StructuredTool(
        name="explorer_overview",
        description="探索给定学术领域的概况、核心问题和主流方法。当用户询问某个领域的基本介绍、发展历史、核心概念时使用。调用后会返回结构化的领域概述。",
        func=explorer_overview,
        args_schema={
            "topic": {"type": "string", "description": "要探索的学术领域或主题"},
        },
    )