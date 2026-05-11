"""前端 UI 所消费的 VisualizationData Pydantic 模型。

字段顺序与 ui/views/* 的访问点对齐：任何字段都允许为空，UI 各 tab
负责"无数据时显示提示"。新增 needs_research 让 research 阶段能定位待补缺口。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


NodeType = Literal[
    "paper",
    "method",
    "dataset",
    "benchmark",
    "metric",
    "trend",
    "concept",
    "topic",
    "resource",
]

EdgeRelation = Literal[
    "proposes",
    "improves_on",
    "evaluated_on",
    "belongs_to",
    "uses",
    "compares_with",
    "related_to",
]


class Overview(BaseModel):
    definition: str = ""
    core_questions: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)


class TimelineEvent(BaseModel):
    year: int = 0
    title: str = ""
    category: str = ""
    description: str = ""
    related_papers: list[str] = Field(default_factory=list)


class Paper(BaseModel):
    id: str
    title: str = ""
    year: int = 0
    authors: str = ""
    venue: str = ""
    summary: str = ""
    url: str = ""


class Method(BaseModel):
    id: str
    name: str = ""
    category: str = ""
    description: str = ""
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    papers: list[str] = Field(default_factory=list)


class Benchmark(BaseModel):
    model: str = ""
    dataset: str = ""
    metric: str = ""
    score: float = 0.0
    year: int = 0
    url: str = ""


class Frontier(BaseModel):
    name: str = ""
    description: str = ""
    importance: str = ""
    related_methods: list[str] = Field(default_factory=list)
    related_papers: list[str] = Field(default_factory=list)


class Resource(BaseModel):
    id: str
    name: str = ""
    type: str = "resource"
    url: str = ""
    description: str = ""
    related_methods: list[str] = Field(default_factory=list)
    related_papers: list[str] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    label: str = ""
    type: NodeType = "concept"


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: EdgeRelation = "related_to"


class KnowledgeGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class ResearchTask(BaseModel):
    """指向某个具体待补字段的占位记录。

    target  形如 "paper:p_xxxx.authors" / "method:m_xxxx.pros"
            / "benchmark:0.score" / "overview.definition"
    reason  为何留空（脚本/LLM 无法确定 / 来源不可信 ...）
    hint    给后续 research 阶段的检索建议
    """

    target: str
    reason: str = ""
    hint: str = ""


class VisualizationData(BaseModel):
    topic: str = "Unknown Topic"
    overview: Overview = Field(default_factory=Overview)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    methods: list[Method] = Field(default_factory=list)
    papers: list[Paper] = Field(default_factory=list)
    benchmarks: list[Benchmark] = Field(default_factory=list)
    frontiers: list[Frontier] = Field(default_factory=list)
    resources: list[Resource] = Field(default_factory=list)
    graph: KnowledgeGraph = Field(default_factory=KnowledgeGraph)
    needs_research: list[ResearchTask] = Field(default_factory=list)
    quality_report: dict[str, Any] = Field(default_factory=dict)
