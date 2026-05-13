"""Knowledge Graph tab — PyVis interactive graph embedded in Streamlit."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.visualizer.schema import VisualizationData


_NODE_COLORS = {
    "topic": "#1f77b4",
    "paper": "#ff7f0e",
    "method": "#2ca02c",
    "dataset": "#d62728",
    "benchmark": "#9467bd",
    "metric": "#8c564b",
    "trend": "#e377c2",
    "concept": "#7f7f7f",
    "resource": "#17becf",
}

_NODE_SIZE = 15
_NODE_FONT_SIZE = 15
_EDGE_FONT_SIZE = 14
_LABEL_DRAW_THRESHOLD = 0
_LABEL_MAX_VISIBLE = 1000
_PHYSICS_GRAVITY = -3500
_PHYSICS_CENTRAL_GRAVITY = 0.18
_PHYSICS_SPRING_LENGTH = 170
_PHYSICS_SPRING_STRENGTH = 0.035
_PHYSICS_DAMPING = 0.35
_PHYSICS_AVOID_OVERLAP = 1.0
_STABILIZATION_ITERATIONS = 500


def _build_pyvis_html(data: VisualizationData) -> str:
    from pyvis.network import Network

    net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#222222", directed=True)
    net.set_options(f"""
    var options = {{
      "layout": {{
        "improvedLayout": true
      }},
      "nodes": {{
        "font": {{
          "size": {_NODE_FONT_SIZE}
        }},
        "scaling": {{
          "label": {{
            "enabled": true,
            "min": {_NODE_FONT_SIZE},
            "max": {_NODE_FONT_SIZE},
            "maxVisible": {_LABEL_MAX_VISIBLE},
            "drawThreshold": {_LABEL_DRAW_THRESHOLD}
          }}
        }}
      }},
      "edges": {{
        "font": {{
          "size": {_EDGE_FONT_SIZE}
        }},
        "scaling": {{
          "label": {{
            "enabled": true,
            "min": {_EDGE_FONT_SIZE},
            "max": {_EDGE_FONT_SIZE},
            "maxVisible": {_LABEL_MAX_VISIBLE},
            "drawThreshold": {_LABEL_DRAW_THRESHOLD}
          }}
        }}
      }},
      "interaction": {{
        "hover": true,
        "navigationButtons": true,
        "keyboard": true
      }},
      "physics": {{
        "enabled": true,
        "solver": "barnesHut",
        "barnesHut": {{
          "gravitationalConstant": {_PHYSICS_GRAVITY},
          "centralGravity": {_PHYSICS_CENTRAL_GRAVITY},
          "springLength": {_PHYSICS_SPRING_LENGTH},
          "springConstant": {_PHYSICS_SPRING_STRENGTH},
          "damping": {_PHYSICS_DAMPING},
          "avoidOverlap": {_PHYSICS_AVOID_OVERLAP}
        }},
        "stabilization": {{
          "enabled": true,
          "iterations": {_STABILIZATION_ITERATIONS},
          "updateInterval": 25,
          "fit": true
        }}
      }}
    }}
    """)

    seen: set[str] = set()
    for node in data.graph.nodes:
        color = _NODE_COLORS.get(node.type, "#7f7f7f")
        net.add_node(
            node.id,
            label=node.label or node.id,
            color=color,
            title=f"type: {node.type}",
            shape="dot",
            size=_NODE_SIZE,
            font={"size": _NODE_FONT_SIZE},
        )
        seen.add(node.id)

    for edge in data.graph.edges:
        if edge.source not in seen:
            net.add_node(
                edge.source,
                label=edge.source,
                color="#7f7f7f",
                title="type: unknown",
                size=_NODE_SIZE,
                font={"size": _NODE_FONT_SIZE},
            )
            seen.add(edge.source)
        if edge.target not in seen:
            net.add_node(
                edge.target,
                label=edge.target,
                color="#7f7f7f",
                title="type: unknown",
                size=_NODE_SIZE,
                font={"size": _NODE_FONT_SIZE},
            )
            seen.add(edge.target)
        net.add_edge(
            edge.source,
            edge.target,
            label=edge.relation,
            title=edge.relation,
            arrows="to",
            font={"size": _EDGE_FONT_SIZE},
        )

    tmp = Path(tempfile.mkdtemp()) / "graph.html"
    net.write_html(str(tmp), open_browser=False, notebook=False)
    return tmp.read_text(encoding="utf-8")


def render(data: VisualizationData) -> None:
    st.subheader("🕸️ 文献知识图谱")

    if not data.graph.nodes and not data.graph.edges:
        st.info("当前数据中没有知识图谱节点/边。")
        return

    legend = " ".join(
        f"<span style='display:inline-block;padding:2px 8px;margin:2px;border-radius:8px;"
        f"background:{color};color:white;font-size:12px'>{ntype}</span>"
        for ntype, color in _NODE_COLORS.items()
    )
    st.markdown(f"**图例**：{legend}", unsafe_allow_html=True)

    try:
        html = _build_pyvis_html(data)
    except Exception as exc:  # PyVis can fail on weird ids; show a friendly message
        st.error(f"知识图谱渲染失败：{exc}")
        return

    components.html(html, height=680, scrolling=True)
