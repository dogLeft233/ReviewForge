"""LLM 校验/补全：把 script_adapter 的产物送给 LLM 清洗。

设计原则：
- 每段独立调用一次 LLM（papers / timeline / methods / benchmarks / frontiers / overview），
  失败的段直接保留 script 输出，不阻塞其它段。
- 严格 JSON-in-JSON-out：LLM 出错就 fallback。
- LLM 不允许编造事实型字段；它只做"删碎片、清作者、判断 importance/pros/cons"等
  确定性补全。剩下的缺口由 needs_research 承接（这一步不删 needs_research，
  refiner 只在确实补上后才剔除对应的 task）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.adapter import prompts
from src.adapter.schema import (
    Benchmark,
    Frontier,
    Method,
    Overview,
    Paper,
    ResearchTask,
    TimelineEvent,
    VisualizationData,
)
from src.llm import LLM, Message

logger = logging.getLogger(__name__)


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict[str, Any] | None:
    """从 LLM 回复里挖出第一个 { ... } JSON 对象。"""
    if not text:
        return None
    text = text.strip()
    # 去 markdown code fence
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("LLM JSON parse failed: %s; raw=%r", exc, text[:300])
        return None


def _llm_call(
    llm: LLM,
    user_prompt: str,
    *,
    max_tokens: int = 4000,
) -> dict[str, Any] | None:
    try:
        reply = llm.chat(
            external_prompt=prompts.REFINER_SYSTEM,
            messages=[Message(role="user", content=user_prompt)],
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return None
    return _extract_json(reply)


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "\n…(truncated)"


# ─────────────────────────────────────────────────────────────────────────────
# 各段 refine 函数
# ─────────────────────────────────────────────────────────────────────────────


def _refine_papers(
    papers: list[Paper],
    raw: dict[str, Any],
    llm: LLM,
) -> list[Paper]:
    if not papers:
        return papers
    payload = prompts.REFINER_PAPERS_USER.format(
        papers_json=json.dumps(
            [p.model_dump() for p in papers], ensure_ascii=False, indent=2
        ),
        stage2_timeline=_truncate(
            str((raw.get("explorer_report") or {}).get("stage2_timeline") or ""), 6000
        ),
    )
    res = _llm_call(llm, payload)
    if not res or "papers" not in res or not isinstance(res["papers"], list):
        logger.info("[refiner] papers: LLM 返回不可用，保留脚本结果")
        return papers
    out: list[Paper] = []
    for item in res["papers"]:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        try:
            out.append(Paper(**item))
        except Exception as exc:
            logger.debug("paper validate fail: %s", exc)
    return out or papers


def _refine_timeline(
    timeline: list[TimelineEvent],
    papers: list[Paper],
    raw: dict[str, Any],
    llm: LLM,
) -> list[TimelineEvent]:
    if not timeline:
        return timeline
    payload = prompts.REFINER_TIMELINE_USER.format(
        timeline_json=json.dumps(
            [e.model_dump() for e in timeline], ensure_ascii=False, indent=2
        ),
        stage2_timeline=_truncate(
            str((raw.get("explorer_report") or {}).get("stage2_timeline") or ""), 6000
        ),
        papers_brief=json.dumps(
            [{"id": p.id, "title": p.title, "year": p.year} for p in papers],
            ensure_ascii=False,
        ),
    )
    res = _llm_call(llm, payload)
    if not res or "timeline" not in res or not isinstance(res["timeline"], list):
        return timeline
    out: list[TimelineEvent] = []
    for item in res["timeline"]:
        if not isinstance(item, dict):
            continue
        try:
            out.append(TimelineEvent(**item))
        except Exception:
            pass
    return out or timeline


def _refine_methods(
    methods: list[Method],
    raw: dict[str, Any],
    llm: LLM,
) -> list[Method]:
    if not methods:
        return methods
    report = raw.get("explorer_report") or {}
    ctx = "\n\n---\n\n".join(
        _truncate(str(report.get(k) or ""), 2500)
        for k in ("stage1_overview", "downstream_report")
    )
    if report.get("stage1_concepts"):
        ctx += "\n\n---\n\n[stage1_concepts]\n" + "\n".join(
            f"- {c}" for c in report["stage1_concepts"][:30]
        )

    payload = prompts.REFINER_METHODS_USER.format(
        methods_json=json.dumps(
            [m.model_dump() for m in methods], ensure_ascii=False, indent=2
        ),
        methods_context=ctx,
    )
    res = _llm_call(llm, payload)
    if not res or "methods" not in res or not isinstance(res["methods"], list):
        return methods
    out: list[Method] = []
    for item in res["methods"]:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        try:
            out.append(Method(**item))
        except Exception:
            pass
    return out or methods


def _refine_benchmarks(
    benchmarks: list[Benchmark],
    raw: dict[str, Any],
    llm: LLM,
) -> list[Benchmark]:
    if not benchmarks:
        return benchmarks
    report = raw.get("explorer_report") or {}
    ctx = json.dumps(report.get("stage3_benchmarks") or [], ensure_ascii=False, indent=2)[
        :5000
    ]
    ctx += "\n\n[stage3_search_results]\n" + _truncate(
        str(report.get("stage3_search_results") or ""), 3000
    )
    payload = prompts.REFINER_BENCHMARKS_USER.format(
        benchmarks_json=json.dumps(
            [b.model_dump() for b in benchmarks], ensure_ascii=False, indent=2
        ),
        benchmarks_context=ctx,
    )
    res = _llm_call(llm, payload)
    if not res or "benchmarks" not in res or not isinstance(res["benchmarks"], list):
        return benchmarks
    out: list[Benchmark] = []
    for item in res["benchmarks"]:
        if not isinstance(item, dict):
            continue
        try:
            out.append(Benchmark(**item))
        except Exception:
            pass
    return out or benchmarks


def _refine_frontiers(
    frontiers: list[Frontier],
    raw: dict[str, Any],
    llm: LLM,
) -> list[Frontier]:
    if not frontiers:
        return frontiers
    report = raw.get("explorer_report") or {}
    ctx = "\n\n---\n\n".join(
        _truncate(str(report.get(k) or ""), 2500)
        for k in ("stage3_state_of_art", "downstream_report")
    )
    payload = prompts.REFINER_FRONTIERS_USER.format(
        frontiers_json=json.dumps(
            [f.model_dump() for f in frontiers], ensure_ascii=False, indent=2
        ),
        frontiers_context=ctx,
    )
    res = _llm_call(llm, payload)
    if not res or "frontiers" not in res or not isinstance(res["frontiers"], list):
        return frontiers
    out: list[Frontier] = []
    for item in res["frontiers"]:
        if not isinstance(item, dict):
            continue
        try:
            out.append(Frontier(**item))
        except Exception:
            pass
    return out or frontiers


def _refine_overview(overview: Overview, raw: dict[str, Any], llm: LLM) -> Overview:
    report = raw.get("explorer_report") or {}
    ctx = _truncate(str(report.get("stage1_overview") or ""), 4000)
    if report.get("stage1_concepts"):
        ctx += "\n\n[stage1_concepts]\n" + "\n".join(
            f"- {c}" for c in report["stage1_concepts"][:30]
        )
    payload = prompts.REFINER_OVERVIEW_USER.format(
        overview_json=json.dumps(overview.model_dump(), ensure_ascii=False, indent=2),
        overview_context=ctx,
    )
    res = _llm_call(llm, payload)
    if not res or "overview" not in res or not isinstance(res["overview"], dict):
        return overview
    try:
        return Overview(**res["overview"])
    except Exception:
        return overview


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────


def _drop_satisfied_tasks(viz: VisualizationData) -> list[ResearchTask]:
    """根据 refine 后的实际值，把已经被 LLM 填上的 needs_research 项剔除。"""

    def _resolved(target: str) -> bool:
        # target 形如 "paper:p_xxx.authors" / "method:m_xxx.pros" / "benchmark:0.score"
        # / "frontier:2.importance" / "overview.definition"
        try:
            head, field = target.split(".", 1)
        except ValueError:
            return False
        if head == "overview":
            v = getattr(viz.overview, field, None)
            return bool(v)
        if ":" not in head:
            return False
        kind, ident = head.split(":", 1)
        match kind:
            case "paper":
                obj = next((p for p in viz.papers if p.id == ident), None)
            case "method":
                obj = next((m for m in viz.methods if m.id == ident), None)
            case "benchmark":
                idx = int(ident) if ident.isdigit() else -1
                obj = viz.benchmarks[idx] if 0 <= idx < len(viz.benchmarks) else None
            case "frontier":
                idx = int(ident) if ident.isdigit() else -1
                obj = viz.frontiers[idx] if 0 <= idx < len(viz.frontiers) else None
            case _:
                return False
        if obj is None:
            return False
        v = getattr(obj, field, None)
        return bool(v)

    return [t for t in viz.needs_research if not _resolved(t.target)]


def refine_with_llm(viz: VisualizationData, raw: dict[str, Any], llm: LLM) -> VisualizationData:
    """逐段 LLM 校验。任何一段失败都退回脚本结果，不抛异常。"""

    logger.info("[refiner] start: papers=%d timeline=%d methods=%d benchmarks=%d frontiers=%d",
                len(viz.papers), len(viz.timeline), len(viz.methods),
                len(viz.benchmarks), len(viz.frontiers))

    new_overview = _refine_overview(viz.overview, raw, llm)
    new_papers = _refine_papers(viz.papers, raw, llm)
    new_timeline = _refine_timeline(viz.timeline, new_papers, raw, llm)
    new_methods = _refine_methods(viz.methods, raw, llm)
    new_benchmarks = _refine_benchmarks(viz.benchmarks, raw, llm)
    new_frontiers = _refine_frontiers(viz.frontiers, raw, llm)

    refined = viz.model_copy(update={
        "overview": new_overview,
        "papers": new_papers,
        "timeline": new_timeline,
        "methods": new_methods,
        "benchmarks": new_benchmarks,
        "frontiers": new_frontiers,
    })
    refined.needs_research = _drop_satisfied_tasks(refined)
    logger.info("[refiner] done: needs_research left=%d", len(refined.needs_research))
    return refined
