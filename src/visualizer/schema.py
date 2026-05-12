"""向后兼容：UI 各 view 仍 `from src.visualizer.schema import VisualizationData`。

实际定义在 src.adapter.schema，本文件只做透传。
"""

from src.adapter.schema import (  # noqa: F401
    Benchmark,
    EdgeRelation,
    Frontier,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    Method,
    NodeType,
    Overview,
    Paper,
    Resource,
    ResearchTask,
    TimelineEvent,
    VisualizationData,
)

__all__ = [
    "Benchmark",
    "EdgeRelation",
    "Frontier",
    "GraphEdge",
    "GraphNode",
    "KnowledgeGraph",
    "Method",
    "NodeType",
    "Overview",
    "Paper",
    "Resource",
    "ResearchTask",
    "TimelineEvent",
    "VisualizationData",
]
