"""ReviewForge agent → 前端 visualization_data 适配层。

两阶段管线：
  1. script_adapter.writer_json_to_visualization() — 规则化转换，必跑
  2. llm_refiner.refine_with_llm()                — 可选 LLM 校验/补全

凡是 LLM 与脚本都不能确定填上的字段，留空（""/[]/0）并写入
VisualizationData.needs_research，等后续 research 阶段补齐。
"""

from src.adapter.pipeline import convert, convert_with_llm
from src.adapter.schema import (
    Benchmark,
    Frontier,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    Method,
    Overview,
    Paper,
    ResearchTask,
    TimelineEvent,
    VisualizationData,
)

__all__ = [
    "convert",
    "convert_with_llm",
    "VisualizationData",
    "Overview",
    "TimelineEvent",
    "Paper",
    "Method",
    "Benchmark",
    "Frontier",
    "GraphNode",
    "GraphEdge",
    "KnowledgeGraph",
    "ResearchTask",
]
