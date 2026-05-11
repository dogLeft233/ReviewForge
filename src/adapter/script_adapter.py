"""规则化转换：step3_writer_done.json → VisualizationData。

只做"明确能从原始数据里读出来"的事；填不了的字段一律留空，
并把缺口写入 needs_research，等 LLM refiner / research 阶段补。
"""

from __future__ import annotations

import re
from typing import Any

from src.adapter import extractor as ex
from src.adapter.schema import (
    Benchmark,
    Frontier,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    Method,
    Overview,
    Paper,
    Resource,
    ResearchTask,
    TimelineEvent,
    VisualizationData,
)


_SLUG_RE = re.compile(r"\W+")


def _slug(s: str, prefix: str = "") -> str:
    base = _SLUG_RE.sub("_", (s or "").lower()).strip("_") or "x"
    return f"{prefix}{base[:38]}"


def _norm_text(s: str) -> str:
    return _SLUG_RE.sub(" ", (s or "").lower()).strip()


def _tokens(s: str) -> set[str]:
    text = (s or "").lower()
    tokens = {t for t in re.split(r"\W+", text) if len(t) >= 3}
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    tokens.update(cjk[i:i + 2] for i in range(max(0, len(cjk) - 1)))
    return tokens


def _has_text_hit(needle: str, haystack: str) -> bool:
    needle = _norm_text(needle)
    haystack = _norm_text(haystack)
    if not needle or not haystack:
        return False
    if needle in haystack:
        return True
    compact_needle = re.sub(r"[\W_]+", "", needle)
    compact_haystack = re.sub(r"[\W_]+", "", haystack)
    if compact_needle and compact_needle in compact_haystack:
        return True
    ns = _tokens(needle)
    hs = _tokens(haystack)
    return bool(ns) and len(ns & hs) >= max(2, min(4, len(ns)))


def _append_pending_once(pending: list[ResearchTask], task: ResearchTask) -> None:
    if not any(t.target == task.target for t in pending):
        pending.append(task)


# ─────────────────────────────────────────────────────────────────────────────
# Overview
# ─────────────────────────────────────────────────────────────────────────────


_CONCEPT_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*\s*[:：]\s*(.+)")
_QUESTION_HINTS = ("挑战", "难题", "问题", "瓶颈", "核心问题", "?", "？")


def _build_overview(raw: dict[str, Any], pending: list[ResearchTask]) -> Overview:
    report = raw.get("explorer_report") or {}

    # definition：取 stage1_overview 第一段实质性内容
    overview_text = str(report.get("stage1_overview") or "")
    definition = ""
    for para in overview_text.split("\n\n"):
        cleaned = "\n".join(
            ln for ln in para.splitlines() if ln.strip() and not ln.strip().startswith("#")
        ).strip()
        if len(cleaned) > 30:
            definition = re.sub(r"\s+", " ", cleaned)[:500]
            break
    if not definition:
        pending.append(
            ResearchTask(
                target="overview.definition",
                reason="stage1_overview 为空或首段过短",
                hint=f"用一句话定义{raw.get('topic', '该领域')}",
            )
        )

    concepts: list[str] = []
    questions: list[str] = []
    for raw_item in report.get("stage1_concepts") or []:
        text = str(raw_item).strip()
        m = _CONCEPT_BOLD_RE.match(text)
        if m:
            name = m.group(1).strip()
            desc = m.group(2).strip()
        else:
            name = text[:30]
            desc = text
        if name and name not in concepts:
            concepts.append(name)
        if any(h in desc for h in _QUESTION_HINTS):
            if name and name not in questions:
                questions.append(name)

    # 如果一个核心问题都没识别到，从 downstream_report 顶部数字列表兜底
    if not questions:
        ds = str(report.get("downstream_report") or "")
        for m in re.finditer(r"\n\s*\d+\.\s*\*\*([^*\n]+?)\*\*", ds):
            questions.append(m.group(1).strip())
            if len(questions) >= 5:
                break

    if not concepts:
        pending.append(
            ResearchTask(
                target="overview.key_concepts",
                reason="stage1_concepts 为空",
                hint=f"列出 {raw.get('topic', '该领域')} 的 6-10 个关键概念",
            )
        )
    if not questions:
        pending.append(
            ResearchTask(
                target="overview.core_questions",
                reason="未在 stage1_concepts/downstream_report 中检索到核心问题",
                hint=f"列出 {raw.get('topic', '该领域')} 的 3-5 个核心研究问题",
            )
        )

    return Overview(
        definition=definition,
        core_questions=questions[:6],
        key_concepts=concepts[:12],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Methods (从 stage1_concepts 派生：核心问题 vs 主流方法)
# ─────────────────────────────────────────────────────────────────────────────


def _build_methods(raw: dict[str, Any], pending: list[ResearchTask]) -> list[Method]:
    report = raw.get("explorer_report") or {}
    items = report.get("stage1_concepts") or []
    methods: list[Method] = []
    seen: set[str] = set()

    for raw_item in items:
        text = str(raw_item).strip()
        m = _CONCEPT_BOLD_RE.match(text)
        if m:
            name = m.group(1).strip()
            desc = m.group(2).strip()
        else:
            # 没有粗体名 → 当作整体描述，跳过（避免把整段塞进 name）
            continue

        category = (
            "核心问题"
            if any(h in desc for h in _QUESTION_HINTS)
            else "主流方法"
        )
        mid = _slug(name, prefix="m_")
        if mid in seen:
            continue
        seen.add(mid)
        methods.append(
            Method(
                id=mid,
                name=name,
                category=category,
                description=desc,
                pros=[],
                cons=[],
                papers=[],
            )
        )
        if category == "主流方法":
            pending.append(
                ResearchTask(
                    target=f"method:{mid}.pros",
                    reason="脚本无法从描述中可靠抽取优点",
                    hint=f"列出方法「{name}」的 2-3 个主要优点",
                )
            )
            pending.append(
                ResearchTask(
                    target=f"method:{mid}.cons",
                    reason="脚本无法从描述中可靠抽取缺点",
                    hint=f"列出方法「{name}」的 2-3 个主要局限",
                )
            )

    if not methods:
        pending.append(
            ResearchTask(
                target="methods",
                reason="stage1_concepts 为空或全部无粗体名",
                hint="按「核心问题/主流方法」两类列出 6-10 个方法/问题",
            )
        )

    return methods


# ─────────────────────────────────────────────────────────────────────────────
# Papers (主源：stage2_timeline 论文表；副源：searcher_papers)
# ─────────────────────────────────────────────────────────────────────────────


def _build_papers(
    raw: dict[str, Any], pending: list[ResearchTask]
) -> tuple[list[Paper], dict[str, str]]:
    """返回 (papers, title_lower → id 映射)，方便后面 timeline/graph 引用。"""

    report = raw.get("explorer_report") or {}
    raw_papers = ex.extract_papers_from_table(str(report.get("stage2_timeline") or ""))
    raw_papers += ex.extract_papers_from_searcher(raw.get("searcher_papers") or [])

    papers: list[Paper] = []
    title_to_id: dict[str, str] = {}
    seen_ids: set[str] = set()

    for rp in raw_papers:
        if not rp.title:
            continue
        pid = _slug(rp.title, prefix="p_")
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        title_to_id[rp.title.lower()] = pid

        paper = Paper(
            id=pid,
            title=rp.title,
            year=rp.year,
            authors=rp.authors,
            venue=rp.venue,
            summary=rp.summary,
            url="",
        )
        papers.append(paper)

        if not rp.year:
            pending.append(
                ResearchTask(
                    target=f"paper:{pid}.year",
                    reason="stage2 表格里没解析到年份",
                    hint=f"查证论文「{rp.title}」的发表年份",
                )
            )
        if not rp.authors:
            pending.append(
                ResearchTask(
                    target=f"paper:{pid}.authors",
                    reason="stage2 表格里没解析到作者",
                    hint=f"查证论文「{rp.title}」的作者列表",
                )
            )
        # url 一律留空 → research 阶段去找
        pending.append(
            ResearchTask(
                target=f"paper:{pid}.url",
                reason="原始数据未提供链接",
                hint=f"为论文「{rp.title}」找到 arXiv / DOI / 官方页面 URL",
            )
        )

    if not papers:
        pending.append(
            ResearchTask(
                target="papers",
                reason="既未在 stage2_timeline 表格里解析出论文，searcher_papers 也为空",
                hint=f"为「{raw.get('topic', '该领域')}」检索 5-10 篇代表论文",
            )
        )

    return papers, title_to_id


# ─────────────────────────────────────────────────────────────────────────────
# Timeline
# ─────────────────────────────────────────────────────────────────────────────


def _build_timeline(
    raw: dict[str, Any],
    papers: list[Paper],
    title_to_id: dict[str, str],
    pending: list[ResearchTask],
) -> list[TimelineEvent]:
    report = raw.get("explorer_report") or {}
    raw_papers = [
        ex.RawPaper(
            title=p.title,
            year=p.year,
            authors=p.authors,
            summary=p.summary,
            contribution=p.summary,
        )
        for p in papers
    ]
    raw_events = ex.extract_timeline(str(report.get("stage2_timeline") or ""), raw_papers)

    events: list[TimelineEvent] = []
    for ev in raw_events:
        related = [
            title_to_id[t.lower()]
            for t in ev.paper_titles
            if t.lower() in title_to_id
        ]
        events.append(
            TimelineEvent(
                year=ev.year,
                title=ev.title,
                category=ev.category,
                description=ev.description,
                related_papers=related,
            )
        )

    if not events:
        pending.append(
            ResearchTask(
                target="timeline",
                reason="papers 列表为空或全无年份",
                hint=f"为「{raw.get('topic', '该领域')}」按年份列出 6-10 个里程碑事件",
            )
        )

    return events


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────


def _build_benchmarks(raw: dict[str, Any], pending: list[ResearchTask]) -> list[Benchmark]:
    report = raw.get("explorer_report") or {}
    raw_rows = ex.extract_benchmarks_from_field(report.get("stage3_benchmarks") or [])

    out: list[Benchmark] = []
    for r in raw_rows:
        out.append(
            Benchmark(
                model=r.model,
                dataset=r.dataset,
                metric=r.metric,
                score=r.score,
                year=r.year,
                url=r.url,
            )
        )

    # 列出所有缺口
    for i, b in enumerate(out):
        if not b.model:
            pending.append(
                ResearchTask(
                    target=f"benchmark:{i}.model",
                    reason="leaderboard 解析未拿到 model 名",
                    hint=f"查 {b.dataset} 数据集上的 SOTA 模型名",
                )
            )
        if not b.score:
            pending.append(
                ResearchTask(
                    target=f"benchmark:{i}.score",
                    reason="leaderboard 解析未拿到分数",
                    hint=f"查 {b.model or '?'} 在 {b.dataset} 上的 {b.metric or '主指标'} 分数",
                )
            )
        if not b.year:
            pending.append(
                ResearchTask(
                    target=f"benchmark:{i}.year",
                    reason="leaderboard 未给出年份",
                    hint=f"查 {b.model or '?'} 在 {b.dataset} 上 SOTA 结果发布年份",
                )
            )

    if not out:
        pending.append(
            ResearchTask(
                target="benchmarks",
                reason="stage3_benchmarks 为空或无可解析 leaderboard",
                hint=f"列出「{raw.get('topic', '该领域')}」3-5 个常用 benchmark + 当前 SOTA",
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Frontiers
# ─────────────────────────────────────────────────────────────────────────────


def _build_frontiers(raw: dict[str, Any], pending: list[ResearchTask]) -> list[Frontier]:
    report = raw.get("explorer_report") or {}
    raws = ex.extract_frontiers(
        str(report.get("stage3_state_of_art") or ""),
        str(report.get("downstream_report") or ""),
        str(report.get("stage3_search_results") or ""),
    )
    out = [
        Frontier(name=f.name, description=f.description, importance="")
        for f in raws
    ]
    if not out:
        pending.append(
            ResearchTask(
                target="frontiers",
                reason="未在 stage3 报告中匹配到 - **xxx**: yyy 形式的趋势",
                hint=f"列出「{raw.get('topic', '该领域')}」5-8 个当前前沿方向",
            )
        )
    else:
        for i, f in enumerate(out):
            pending.append(
                ResearchTask(
                    target=f"frontier:{i}.importance",
                    reason="脚本无法判断重要性等级",
                    hint=f"评估前沿方向「{f.name}」的重要性（high/medium/low + 一句理由）",
                )
            )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Graph (启发式：能确定的关系才连)
# ─────────────────────────────────────────────────────────────────────────────


def _build_graph(
    topic: str,
    methods: list[Method],
    papers: list[Paper],
    benchmarks: list[Benchmark],
    frontiers: list[Frontier],
) -> KnowledgeGraph:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen: set[str] = set()

    def add(nid: str, label: str, ntype: str) -> None:
        if nid in seen:
            return
        seen.add(nid)
        nodes.append(GraphNode(id=nid, label=label, type=ntype))  # type: ignore[arg-type]

    topic_id = _slug(topic, prefix="t_")
    add(topic_id, topic, "topic")

    cat_ids: dict[str, str] = {}
    for m in methods:
        if m.category:
            cid = cat_ids.setdefault(m.category, _slug(m.category, prefix="c_"))
            add(cid, m.category, "concept")
            edges.append(GraphEdge(source=cid, target=topic_id, relation="belongs_to"))
        add(m.id, m.name, "method")
        target = cat_ids.get(m.category, topic_id)
        edges.append(GraphEdge(source=m.id, target=target, relation="belongs_to"))

    for p in papers:
        add(p.id, p.title[:30], "paper")
        text = f"{p.title} {p.summary}".lower()
        proposed = next(
            (m for m in methods if m.name and m.name.lower() in text and m.category != "核心问题"),
            None,
        )
        if proposed:
            edges.append(GraphEdge(source=p.id, target=proposed.id, relation="proposes"))
        else:
            edges.append(GraphEdge(source=p.id, target=topic_id, relation="related_to"))

    for i, b in enumerate(benchmarks):
        bid = _slug(b.model or f"bm_{i}", prefix="bm_")
        label = f"{b.model or '(unknown)'} ({b.score}{b.metric})" if b.score else (b.model or f"bm_{i}")
        add(bid, label, "benchmark")
        if b.dataset:
            did = _slug(b.dataset, prefix="d_")
            add(did, b.dataset, "dataset")
            edges.append(GraphEdge(source=bid, target=did, relation="evaluated_on"))
        for m in methods:
            if m.name and m.name.lower() in (b.model or "").lower():
                edges.append(GraphEdge(source=bid, target=m.id, relation="uses"))
                break

    for fr in frontiers:
        fid = _slug(fr.name, prefix="tr_")
        add(fid, fr.name, "trend")
        edges.append(GraphEdge(source=fid, target=topic_id, relation="related_to"))

    return KnowledgeGraph(nodes=nodes, edges=edges)


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────


def writer_json_to_visualization(raw: dict[str, Any]) -> VisualizationData:
    """脚本路径——拿到原始 step3_writer_done.json dict，输出 VisualizationData。"""
    topic = str(raw.get("topic") or "Unknown Topic")
    pending: list[ResearchTask] = []

    overview = _build_overview(raw, pending)
    methods = _build_methods(raw, pending)
    papers, title_to_id = _build_papers(raw, pending)
    timeline = _build_timeline(raw, papers, title_to_id, pending)
    benchmarks = _build_benchmarks(raw, pending)
    frontiers = _build_frontiers(raw, pending)
    graph = _build_graph(topic, methods, papers, benchmarks, frontiers)

    return VisualizationData(
        topic=topic,
        overview=overview,
        timeline=timeline,
        methods=methods,
        papers=papers,
        benchmarks=benchmarks,
        frontiers=frontiers,
        graph=graph,
        needs_research=pending,
    )


# ---------------------------------------------------------------------------
# Enhanced adapter pass.
#
# These definitions intentionally shadow the earlier conservative builders.
# Keeping the original code above makes the old behavior easy to compare while
# allowing the exported module functions to consume richer source fields.
# ---------------------------------------------------------------------------


def _build_papers(
    raw: dict[str, Any], pending: list[ResearchTask]
) -> tuple[list[Paper], dict[str, str]]:
    """Return merged papers plus title-to-id mapping for later references."""

    report = raw.get("explorer_report") or {}
    raw_papers = ex.extract_papers_from_classics(report.get("stage2_classics") or [])
    raw_papers += ex.extract_papers_from_table(str(report.get("stage2_timeline") or ""))
    raw_papers += ex.extract_papers_from_searcher(raw.get("searcher_papers") or [])

    by_id: dict[str, Paper] = {}
    title_to_id: dict[str, str] = {}
    for rp in raw_papers:
        if not rp.title:
            continue
        pid = _slug(rp.title, prefix="p_")
        title_to_id[rp.title.lower()] = pid
        existing = by_id.get(pid)
        if existing is None:
            by_id[pid] = Paper(
                id=pid,
                title=rp.title,
                year=rp.year,
                authors=rp.authors,
                venue=rp.venue,
                summary=rp.summary,
                url=rp.url,
            )
        else:
            by_id[pid] = existing.model_copy(update={
                "year": existing.year or rp.year,
                "authors": existing.authors or rp.authors,
                "venue": existing.venue or rp.venue,
                "summary": existing.summary or rp.summary,
                "url": existing.url or rp.url,
            })

    papers = list(by_id.values())
    for paper in papers:
        if not paper.year:
            _append_pending_once(
                pending,
                ResearchTask(
                    target=f"paper:{paper.id}.year",
                    reason="source data did not provide a reliable publication year",
                    hint=f"Find the publication year for paper: {paper.title}",
                ),
            )
        if not paper.authors:
            _append_pending_once(
                pending,
                ResearchTask(
                    target=f"paper:{paper.id}.authors",
                    reason="source data did not provide reliable authors",
                    hint=f"Find the author list for paper: {paper.title}",
                ),
            )
        if not paper.url:
            _append_pending_once(
                pending,
                ResearchTask(
                    target=f"paper:{paper.id}.url",
                    reason="source data did not provide a paper URL",
                    hint=f"Find the arXiv / DOI / official URL for paper: {paper.title}",
                ),
            )

    if not papers:
        _append_pending_once(
            pending,
            ResearchTask(
                target="papers",
                reason="no papers parsed from stage2_classics, stage2_timeline, or searcher_papers",
                hint=f"Find 5-10 representative papers for {raw.get('topic', 'this topic')}",
            ),
        )

    return papers, title_to_id


def _build_benchmarks(raw: dict[str, Any], pending: list[ResearchTask]) -> list[Benchmark]:
    report = raw.get("explorer_report") or {}
    raw_rows = ex.extract_benchmarks_from_field(report.get("stage3_benchmarks") or [])
    raw_rows += ex.extract_benchmarks_from_text(str(report.get("stage3_search_results") or ""))
    raw_rows += ex.extract_benchmarks_from_text(str(report.get("stage3_state_of_art") or ""))

    seen: set[tuple[str, str, str, float, int]] = set()
    out: list[Benchmark] = []
    for r in raw_rows:
        key = (r.model.lower(), r.dataset.lower(), r.metric.lower(), r.score, r.year)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Benchmark(
                model=r.model,
                dataset=r.dataset,
                metric=r.metric,
                score=r.score,
                year=r.year,
                url=r.url,
            )
        )

    for i, b in enumerate(out):
        if not b.model:
            _append_pending_once(
                pending,
                ResearchTask(
                    target=f"benchmark:{i}.model",
                    reason="leaderboard parsing did not find a model name",
                    hint=f"Find the SOTA model for dataset {b.dataset}",
                ),
            )
        if not b.score:
            _append_pending_once(
                pending,
                ResearchTask(
                    target=f"benchmark:{i}.score",
                    reason="leaderboard parsing did not find a numeric score",
                    hint=f"Find {b.metric or 'the main metric'} for {b.model or '?'} on {b.dataset}",
                ),
            )
        if not b.year:
            _append_pending_once(
                pending,
                ResearchTask(
                    target=f"benchmark:{i}.year",
                    reason="leaderboard parsing did not find a result year",
                    hint=f"Find the release year for {b.model or '?'} on {b.dataset}",
                ),
            )

    if not out:
        _append_pending_once(
            pending,
            ResearchTask(
                target="benchmarks",
                reason="no benchmark rows parsed from stage3_benchmarks or stage3 reports",
                hint=f"List 3-5 common benchmarks and current SOTA rows for {raw.get('topic', 'this topic')}",
            ),
        )
    return out


def _build_frontiers(raw: dict[str, Any], pending: list[ResearchTask]) -> list[Frontier]:
    report = raw.get("explorer_report") or {}
    raws = ex.extract_frontiers(
        str(report.get("stage3_state_of_art") or ""),
        str(report.get("downstream_report") or ""),
        str(report.get("stage3_search_results") or ""),
    )
    trend_raws = ex.extract_frontiers_from_trends(report.get("stage3_trends") or [])

    seen: set[str] = set()
    out: list[Frontier] = []
    for f in [*raws, *trend_raws]:
        key = f.name.lower()
        if not f.name or key in seen:
            continue
        seen.add(key)
        out.append(Frontier(name=f.name, description=f.description, importance=""))

    if not out:
        _append_pending_once(
            pending,
            ResearchTask(
                target="frontiers",
                reason="no frontier trends parsed from stage3 reports",
                hint=f"List 5-8 current frontier directions for {raw.get('topic', 'this topic')}",
            ),
        )
    else:
        for i, f in enumerate(out):
            _append_pending_once(
                pending,
                ResearchTask(
                    target=f"frontier:{i}.importance",
                    reason="script cannot rank trend importance reliably",
                    hint=f"Assess importance of frontier direction: {f.name}",
                ),
            )
    return out


def _link_methods_to_papers(methods: list[Method], papers: list[Paper]) -> list[Method]:
    linked: list[Method] = []
    for method in methods:
        refs: list[str] = []
        for paper in papers:
            haystack = f"{paper.title} {paper.summary}"
            if _has_text_hit(method.name, haystack):
                refs.append(paper.id)
        linked.append(method.model_copy(update={"papers": refs or method.papers}))
    return linked


def _link_frontiers(
    frontiers: list[Frontier],
    methods: list[Method],
    papers: list[Paper],
) -> list[Frontier]:
    linked: list[Frontier] = []
    for frontier in frontiers:
        text = f"{frontier.name} {frontier.description}"
        related_methods = [
            method.id for method in methods
            if _has_text_hit(method.name, text) or _has_text_hit(text, method.description)
        ]
        related_papers = [
            paper.id for paper in papers
            if _has_text_hit(frontier.name, f"{paper.title} {paper.summary}")
        ]
        linked.append(frontier.model_copy(update={
            "related_methods": related_methods or frontier.related_methods,
            "related_papers": related_papers or frontier.related_papers,
        }))
    return linked


def _build_resources(
    raw: dict[str, Any],
    methods: list[Method],
    papers: list[Paper],
) -> list[Resource]:
    report = raw.get("explorer_report") or {}
    resources = ex.extract_resources_from_text(
        str(report.get("stage2_timeline") or ""),
        str(report.get("stage2_search_results") or ""),
        str(report.get("stage3_state_of_art") or ""),
        str(report.get("stage3_search_results") or ""),
        str(report.get("downstream_report") or ""),
        str(raw.get("searcher_result") or ""),
    )

    out: list[Resource] = []
    seen: set[str] = set()
    for r in resources:
        if not r.url or r.url in seen:
            continue
        seen.add(r.url)
        text = f"{r.name} {r.description} {r.url}"
        related_methods = [
            method.id for method in methods
            if _has_text_hit(method.name, text)
        ]
        related_papers = [
            paper.id for paper in papers
            if _has_text_hit(paper.title, text) or _has_text_hit(text, paper.title)
        ]
        out.append(
            Resource(
                id=_slug(r.url, prefix="r_"),
                name=r.name,
                type=r.type,
                url=r.url,
                description=r.description,
                related_methods=related_methods,
                related_papers=related_papers,
            )
        )
    return out


def _backfill_paper_urls_from_resources(
    papers: list[Paper],
    resources: list[Resource],
) -> list[Paper]:
    updated: list[Paper] = []
    for paper in papers:
        if paper.url:
            updated.append(paper)
            continue
        match = next(
            (
                r for r in resources
                if r.type == "paper"
                and (_has_text_hit(paper.title, r.name) or _has_text_hit(paper.title, r.description))
            ),
            None,
        )
        updated.append(paper.model_copy(update={"url": match.url}) if match else paper)
    return updated


def _build_graph(
    topic: str,
    methods: list[Method],
    papers: list[Paper],
    benchmarks: list[Benchmark],
    frontiers: list[Frontier],
    resources: list[Resource] | None = None,
) -> KnowledgeGraph:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen: set[str] = set()
    edge_seen: set[tuple[str, str, str]] = set()

    def add(nid: str, label: str, ntype: str) -> None:
        if nid in seen:
            return
        seen.add(nid)
        nodes.append(GraphNode(id=nid, label=label, type=ntype))  # type: ignore[arg-type]

    def connect(source: str, target: str, relation: str) -> None:
        key = (source, target, relation)
        if source == target or key in edge_seen:
            return
        edge_seen.add(key)
        edges.append(GraphEdge(source=source, target=target, relation=relation))  # type: ignore[arg-type]

    topic_id = _slug(topic, prefix="t_")
    add(topic_id, topic, "topic")

    cat_ids: dict[str, str] = {}
    for method in methods:
        if method.category:
            cid = cat_ids.setdefault(method.category, _slug(method.category, prefix="c_"))
            add(cid, method.category, "concept")
            connect(cid, topic_id, "belongs_to")
        add(method.id, method.name, "method")
        connect(method.id, cat_ids.get(method.category, topic_id), "belongs_to")

    paper_lookup = {p.id: p for p in papers}
    for paper in papers:
        add(paper.id, paper.title[:45] or paper.id, "paper")
        connected = False
        for method in methods:
            if paper.id in method.papers or _has_text_hit(method.name, f"{paper.title} {paper.summary}"):
                connect(paper.id, method.id, "proposes")
                connected = True
        if not connected:
            connect(paper.id, topic_id, "related_to")

    for i, bench in enumerate(benchmarks):
        bid = _slug(f"{bench.model}_{bench.dataset}_{i}", prefix="bm_")
        label = f"{bench.model or '(unknown)'} ({bench.score}{bench.metric})" if bench.score else (bench.model or f"benchmark {i + 1}")
        add(bid, label, "benchmark")
        connect(bid, topic_id, "related_to")
        if bench.dataset:
            did = _slug(bench.dataset, prefix="d_")
            add(did, bench.dataset, "dataset")
            connect(bid, did, "evaluated_on")
        if bench.metric:
            mid = _slug(bench.metric, prefix="metric_")
            add(mid, bench.metric, "metric")
            connect(bid, mid, "uses")
        for method in methods:
            if _has_text_hit(method.name, bench.model):
                connect(bid, method.id, "uses")
                break

    for i, frontier in enumerate(frontiers):
        fid = _slug(frontier.name or f"frontier_{i}", prefix="tr_")
        add(fid, frontier.name, "trend")
        connect(fid, topic_id, "related_to")
        for method_id in frontier.related_methods:
            connect(fid, method_id, "related_to")
        for paper_id in frontier.related_papers:
            if paper_id in paper_lookup:
                connect(fid, paper_id, "related_to")

    for resource in resources or []:
        add(resource.id, resource.name, "resource")
        connect(resource.id, topic_id, "related_to")
        for method_id in resource.related_methods:
            connect(resource.id, method_id, "related_to")
        for paper_id in resource.related_papers:
            if paper_id in paper_lookup:
                connect(resource.id, paper_id, "related_to")

    return KnowledgeGraph(nodes=nodes, edges=edges)


def rebuild_graph(viz: VisualizationData) -> VisualizationData:
    """Rebuild graph from the current refined entities."""

    return viz.model_copy(update={
        "graph": _build_graph(viz.topic, viz.methods, viz.papers, viz.benchmarks, viz.frontiers, viz.resources)
    })


def writer_json_to_visualization(raw: dict[str, Any]) -> VisualizationData:
    """Enhanced conversion from step3_writer_done.json to VisualizationData."""

    topic = str(raw.get("topic") or "Unknown Topic")
    pending: list[ResearchTask] = []

    overview = _build_overview(raw, pending)
    methods = _build_methods(raw, pending)
    papers, title_to_id = _build_papers(raw, pending)
    methods = _link_methods_to_papers(methods, papers)
    resources = _build_resources(raw, methods, papers)
    papers = _backfill_paper_urls_from_resources(papers, resources)
    timeline = _build_timeline(raw, papers, title_to_id, pending)
    benchmarks = _build_benchmarks(raw, pending)
    frontiers = _link_frontiers(_build_frontiers(raw, pending), methods, papers)
    graph = _build_graph(topic, methods, papers, benchmarks, frontiers, resources)

    return VisualizationData(
        topic=topic,
        overview=overview,
        timeline=timeline,
        methods=methods,
        papers=papers,
        benchmarks=benchmarks,
        frontiers=frontiers,
        resources=resources,
        graph=graph,
        needs_research=pending,
    )
