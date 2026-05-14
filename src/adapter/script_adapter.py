"""规则化转换：step3_writer_done.json → VisualizationData。

只做"明确能从原始数据里读出来"的事；填不了的字段一律留空，
并把缺口写入 needs_research，等 LLM refiner / research 阶段补。
"""

from __future__ import annotations

import json
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
_METHOD_NAME_HINTS = (
    "模型", "方法", "机制", "学习", "架构", "算法", "范式", "建模",
    "model", "method", "learning", "architecture", "attention",
)
_NON_METHOD_NAME_HINTS = (
    "鲁棒性", "口音", "语速", "语境", "多语言支持", "实时性", "效率",
    "challenge", "problem",
)


def _first_body_paragraph(markdown: str) -> str:
    for para in (markdown or "").split("\n\n"):
        cleaned = "\n".join(
            ln.strip()
            for ln in para.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ).strip()
        if len(cleaned) > 30:
            return re.sub(r"\s+", " ", cleaned)[:500]
    return ""


def _downstream_overview_definition(text: str) -> str:
    m = re.search(r"##\s*[一1][、.]\s*领域全景(?P<body>.*?)(?:\n##\s*[二2][、.]|\Z)", text or "", re.S)
    return _first_body_paragraph(m.group("body")) if m else ""


def _build_overview(raw: dict[str, Any], pending: list[ResearchTask]) -> Overview:
    report = raw.get("explorer_report") or {}

    # definition：严格优先 stage1_overview；仅当 stage1 缺失时才用 downstream_report 兜底。
    overview_text = str(report.get("stage1_overview") or "")
    definition = _first_body_paragraph(overview_text)
    if not definition:
        definition = _downstream_overview_definition(str(report.get("downstream_report") or ""))
    if not definition:
        pending.append(
            ResearchTask(
                target="overview.definition",
                reason="stage1_overview 为空或首段过短，downstream_report 也没有可用领域全景段落",
                hint=f"用一句话定义{raw.get('topic', '该领域')}",
            )
        )

    concepts: list[str] = []
    explanations: dict[str, str] = {}
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
        if name and desc:
            explanations[name] = re.sub(r"\s+", " ", desc).strip()
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
        key_concept_explanations={
            k: v for k, v in explanations.items() if k in concepts[:12]
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Methods
# ─────────────────────────────────────────────────────────────────────────────


def _is_method_name(name: str, desc: str = "") -> bool:
    text = f"{name} {desc}".lower()
    if any(h in desc for h in _QUESTION_HINTS):
        return False
    if any(h.lower() in text for h in _NON_METHOD_NAME_HINTS):
        return False
    return any(h.lower() in text for h in _METHOD_NAME_HINTS)


def _method_sections(raw: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract method-like sections from writer_report.body."""

    body = str((raw.get("writer_report") or {}).get("body") or "")
    sections: list[tuple[str, str]] = []
    if not body:
        return sections

    header_re = re.compile(r"^##\s+3\.\d+\s+(.+?)\s*$", re.MULTILINE)
    matches = list(header_re.finditer(body))
    for i, match in enumerate(matches):
        name = re.sub(r"[*_`#]+", "", match.group(1)).strip()
        name = re.sub(r"\s+", " ", name)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section = body[start:end].strip()
        if name and _is_method_name(name, section):
            sections.append((name, section))
    return sections


def _sentences_with_any(text: str, hints: tuple[str, ...], limit: int = 2) -> list[str]:
    out: list[str] = []
    for sent in re.split(r"[。.!?？]\s*", text or ""):
        cleaned = re.sub(r"\s+", " ", sent).strip(" -*\t")
        if not cleaned:
            continue
        if any(h in cleaned for h in hints):
            out.append(cleaned[:180])
        if len(out) >= limit:
            break
    return out


def _method_notes(section: str, fallback_desc: str) -> tuple[str, list[str], list[str]]:
    description = ""
    for sent in re.split(r"[。.!?？]\s*", section or fallback_desc):
        cleaned = re.sub(r"\s+", " ", sent).strip(" -*\t")
        if len(cleaned) >= 20:
            description = cleaned[:360]
            break
    description = description or fallback_desc[:360]

    pros = _sentences_with_any(section, ("优势", "提升", "降低", "减少", "增强", "适应", "简化"))
    cons = _sentences_with_any(section, ("但", "局限", "依赖", "成本", "复杂", "不足", "受限"))
    if not pros and fallback_desc:
        pros = [fallback_desc[:160]]
    if not cons:
        cons = ["需要结合具体数据规模、实时性约束和应用场景评估适用边界"]
    return description, pros[:3], cons[:3]


def _build_methods(raw: dict[str, Any], pending: list[ResearchTask]) -> list[Method]:
    report = raw.get("explorer_report") or {}
    concept_desc: dict[str, str] = {}
    for raw_item in report.get("stage1_concepts") or []:
        text = str(raw_item).strip()
        m = _CONCEPT_BOLD_RE.match(text)
        if not m:
            continue
        name = m.group(1).strip()
        desc = m.group(2).strip()
        if _is_method_name(name, desc):
            concept_desc[name] = desc

    candidates: list[tuple[str, str, str]] = []
    for name, section in _method_sections(raw):
        candidates.append((name, "主流方法", section))
    for name, desc in concept_desc.items():
        if not any(_has_text_hit(name, candidate[0]) for candidate in candidates):
            candidates.append((name, "主流方法", desc))

    methods: list[Method] = []
    seen: set[str] = set()
    for name, category, source_text in candidates:
        mid = _slug(name, prefix="m_")
        if mid in seen:
            continue
        seen.add(mid)
        description, pros, cons = _method_notes(source_text, concept_desc.get(name, source_text))
        methods.append(
            Method(
                id=mid,
                name=name,
                category=category,
                description=description,
                pros=pros,
                cons=cons,
                papers=[],
            )
        )

    if not methods:
        pending.append(
            ResearchTask(
                target="methods",
                reason="未从 stage1_concepts 或 writer_report.body 中识别出方法类条目",
                hint="按主流技术路线列出 4-8 个方法，并为每个方法关联代表论文",
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


_BROAD_TIMELINE_TITLES = {
    "attention is all you need",
}
_ASR_PAPER_HINTS = (
    "speech", "recognition", "asr", "acoustic", "ctc", "wav2vec",
    "deepspeech", "conformer", "语音", "识别", "声学",
)
_STAGE2_PHASE_LINE_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:\d+[.、]\s*)?\*\*([^*\n]*(?:(?:19|20)\d{2}|至今)[^*\n]*)\*\*"
)
_STAGE2_TITLE_RE = re.compile(r"[《〈]([^》〉]+)[》〉]")


def _is_timeline_paper_relevant(topic: str, title: str, description: str) -> bool:
    title_norm = _norm_text(title)
    if title_norm in _BROAD_TIMELINE_TITLES:
        return False
    if "asr" in topic.lower() or "语音" in topic or "speech" in topic.lower():
        text = f"{title} {description}".lower()
        return any(h in text for h in _ASR_PAPER_HINTS)
    return True


def _clean_stage2_category(header: str) -> str:
    text = re.sub(r"^\s*(?:#+\s*)?(?:\d+[.、]\s*)?", "", header or "").strip()
    text = re.sub(r"[*_`]+", "", text).strip()
    return re.sub(r"\s+", " ", text).strip("：: ")


def _parse_stage2_bullet_events(
    stage2_timeline_md: str,
    papers: list[Paper],
) -> list[ex.RawTimelineEvent]:
    """Preserve stage2_timeline's own phase headings and paper order."""

    paper_by_title = {p.title.lower(): p for p in papers if p.title}
    events: list[ex.RawTimelineEvent] = []
    seen: set[str] = set()
    current_category = ""

    for raw_line in (stage2_timeline_md or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        phase_match = _STAGE2_PHASE_LINE_RE.match(line)
        if phase_match:
            current_category = _clean_stage2_category(phase_match.group(1))
            continue
        if line.startswith("|"):
            continue
        title_match = _STAGE2_TITLE_RE.search(line)
        if not title_match:
            continue
        title = title_match.group(1).strip()
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        paper = paper_by_title.get(key)
        year_match = re.search(r"(?:19|20)\d{2}", line)
        year = paper.year if paper and paper.year else int(year_match.group(0)) if year_match else 0
        description = paper.summary if paper and paper.summary else re.sub(r"\s+", " ", line).strip(" -*")
        events.append(
            ex.RawTimelineEvent(
                year=year,
                title=title,
                category=current_category,
                description=description,
                paper_titles=[title],
            )
        )
    return events


def _parse_downstream_timeline_events(
    downstream_report: str,
    title_to_id: dict[str, str],
) -> list[TimelineEvent]:
    """Fallback only: parse downstream timeline table when stage2 has no usable events."""

    events: list[TimelineEvent] = []
    for raw_line in (downstream_report or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [re.sub(r"[*_`]+", "", c.strip()) for c in line.strip("|").split("|")]
        if len(cells) < 3 or set(cells[0]) <= set("-: "):
            continue
        if "时间" in cells[0] or "阶段" in cells[0]:
            continue
        category = cells[0]
        title = cells[1]
        year_match = re.search(r"(?:19|20)\d{2}", title)
        if not year_match:
            year_match = re.search(r"(?:19|20)\d{2}", category)
        if not title or not year_match:
            continue
        related = [
            pid for source_title, pid in title_to_id.items()
            if _has_text_hit(title, source_title) or _has_text_hit(source_title, title)
        ]
        events.append(
            TimelineEvent(
                year=int(year_match.group(0)),
                title=title,
                category=category,
                description=cells[2],
                related_papers=related[:1],
            )
        )
    events.sort(key=lambda e: (e.year, e.title))
    return events


def _build_timeline(
    raw: dict[str, Any],
    papers: list[Paper],
    title_to_id: dict[str, str],
    pending: list[ResearchTask],
) -> list[TimelineEvent]:
    report = raw.get("explorer_report") or {}
    stage2_timeline = str(report.get("stage2_timeline") or "")
    raw_events = _parse_stage2_bullet_events(stage2_timeline, papers)
    if not raw_events:
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
        raw_events = ex.extract_timeline(stage2_timeline, raw_papers)

    events: list[TimelineEvent] = []
    for ev in raw_events:
        if not _is_timeline_paper_relevant(str(raw.get("topic") or ""), ev.title, ev.description):
            _append_pending_once(
                pending,
                ResearchTask(
                    target=f"timeline:{_slug(ev.title)}",
                    reason="脚本判定该事件标题过于泛化，未作为领域时间线里程碑输出",
                    hint=f"为「{ev.title}」查找更具体的领域内代表论文或替代里程碑",
                ),
            )
            continue
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
        events = _parse_downstream_timeline_events(str(report.get("downstream_report") or ""), title_to_id)

    if not events:
        _append_pending_once(
            pending,
            ResearchTask(
                target="timeline",
                reason="stage2_timeline 中没有可用论文/年份，downstream_report 兜底也未解析出事件",
                hint=f"为「{raw.get('topic', '该领域')}」按年份列出 6-10 个里程碑事件",
            ),
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


_GENERIC_FRONTIER_NAMES = {
    "数据准备", "模型选择", "评估指标", "当前最佳成绩", "url", "链接",
    "leaderboard", "benchmark", "排行榜", "说明", "description",
}


def _is_recent_frontier_text(text: str) -> bool:
    lower = text.lower()
    return (
        any(str(year) in lower for year in range(2024, 2027))
        or bool(re.search(r"arxiv\.org/(?:abs|pdf)/2[4-6]\d{2}", lower))
    )


def _is_open_source_text(text: str) -> bool:
    lower = text.lower()
    return any(token in lower for token in ("github", "huggingface.co", "open-source", "open source", "开源"))


def _resource_supports_frontier(resource: Resource) -> bool:
    text = f"{resource.name} {resource.description} {resource.url}"
    if resource.type == "github_repo":
        return True
    if resource.type == "model_or_space" and "leaderboard" not in text.lower():
        return True
    return resource.type == "paper" and _is_recent_frontier_text(text)


def _build_frontiers(
    raw: dict[str, Any],
    pending: list[ResearchTask],
    resources: list[Resource] | None = None,
) -> list[Frontier]:
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


def _friendly_link_name(url: str, fallback: str = "") -> str:
    lower = url.lower()
    if "paperswithcode.com" in lower:
        return "Papers With Code - ASR SOTA"
    if "huggingface.co" in lower:
        return "Hugging Face ASR Leaderboard"
    if "kaggle.com" in lower:
        return "Kaggle Speech Recognition Competitions"
    if "librispeech" in lower:
        return "LibriSpeech"
    if fallback and not fallback.startswith(("http://", "https://")):
        return fallback.strip()
    return url.rstrip("/").split("/")[-1] or fallback or "Benchmark"


def _is_benchmark_catalog_url(url: str) -> bool:
    lower = url.lower()
    if "arxiv.org" in lower or "doi.org" in lower:
        return False
    return any(
        token in lower
        for token in (
            "paperswithcode.com",
            "huggingface.co/spaces",
            "kaggle.com/competitions",
            "librispeech",
            "commonvoice",
            "catalog.ldc.upenn.edu",
            "benchmark",
            "leaderboard",
            "dataset",
        )
    )


def _extract_value_after_label(block: str, labels: tuple[str, ...]) -> str:
    for label in labels:
        m = re.search(rf"\*\*{re.escape(label)}\*\*\s*[:：]\s*([^\n]+)", block)
        if m:
            return re.sub(r"\s+", " ", re.sub(r"\[[^\]]+\]\([^)]+\)", "", m.group(1))).strip(" -")
    return ""


def _extract_benchmark_catalog_from_sections(text: str) -> list[Benchmark]:
    section_re = re.compile(r"^###\s+\d+\.\s*(?:\*\*)?(.+?)(?:\*\*)?\s*$", re.MULTILINE)
    matches = list(section_re.finditer(text or ""))
    out: list[Benchmark] = []
    for i, match in enumerate(matches):
        name = re.sub(r"[*_`#]+", "", match.group(1)).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        url = ex._clean_url(block)  # noqa: SLF001 - adapter-level reuse of extractor URL normalization
        if not url or not _is_benchmark_catalog_url(url):
            continue
        metric = _extract_value_after_label(block, ("评估指标", "Metric", "Metrics"))

        dataset_lines: list[str] = []
        in_dataset = False
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if "测试数据集" in line or "Dataset" in line:
                in_dataset = True
                after = re.split(r"[:：]", line, maxsplit=1)
                if len(after) > 1 and after[1].strip():
                    dataset_lines.append(after[1].strip(" -*"))
                continue
            if in_dataset and line.startswith("- **"):
                break
            if in_dataset and line.startswith("-"):
                dataset_lines.append(line.strip(" -*"))
        dataset = ", ".join(dict.fromkeys(d for d in dataset_lines if d))

        desc = _extract_value_after_label(block, ("说明", "描述", "Description"))
        if not desc:
            desc = "常用评测入口，汇总该方向的公开数据集、指标和模型对比结果"
        out.append(
            Benchmark(
                model=_friendly_link_name(url, name),
                dataset=dataset,
                metric=metric,
                description=desc,
                url=url,
            )
        )
    return out


def _extract_benchmark_catalog_from_table(text: str) -> list[Benchmark]:
    out: list[Benchmark] = []
    for raw_line in (text or "").splitlines():
        if not raw_line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in raw_line.strip().strip("|").split("|")]
        if len(cells) < 3 or set(cells[0]) <= set("-: "):
            continue
        if cells[0] in {"榜单名称", "Benchmark", "名称"}:
            continue
        url = ex._clean_url(raw_line)  # noqa: SLF001
        if not url or not _is_benchmark_catalog_url(url):
            continue
        out.append(
            Benchmark(
                model=re.sub(r"[*_`]+", "", cells[0]).strip(),
                dataset="",
                metric="",
                description=re.sub(r"\[[^\]]+\]\([^)]+\)", "", cells[2]).strip() if len(cells) >= 3 else "",
                url=url,
            )
        )
    return out


def _build_benchmarks(raw: dict[str, Any], pending: list[ResearchTask]) -> list[Benchmark]:
    report = raw.get("explorer_report") or {}
    candidates: list[Benchmark] = []

    candidates += _extract_benchmark_catalog_from_sections(str(report.get("stage3_search_results") or ""))
    candidates += _extract_benchmark_catalog_from_sections(str(report.get("downstream_report") or ""))
    candidates += _extract_benchmark_catalog_from_table(str(report.get("downstream_report") or ""))

    for item in report.get("stage3_benchmarks") or []:
        if not isinstance(item, dict):
            continue
        url = ex._clean_url(item.get("url") or item.get("name") or item.get("description"))  # noqa: SLF001
        if not url or not _is_benchmark_catalog_url(url):
            continue
        candidates.append(
            Benchmark(
                model=_friendly_link_name(url, str(item.get("name") or "")),
                dataset=str(item.get("dataset") or "").strip(),
                metric=str(item.get("metric") or "").strip(),
                description=str(item.get("description") or "").strip(),
                url=url,
            )
        )

    out: list[Benchmark] = []
    seen: set[str] = set()
    for b in candidates:
        key = b.url or b.model
        if not key or key in seen:
            continue
        seen.add(key)
        description = b.description
        if description.startswith("- **URL**"):
            description = "常用评测入口，汇总该方向的公开数据集、指标和模型对比结果"
        out.append(b.model_copy(update={"description": description}))

    for i, b in enumerate(out):
        if not b.url:
            _append_pending_once(
                pending,
                ResearchTask(
                    target=f"benchmark:{i}.url",
                    reason="benchmark catalog row has no source URL",
                    hint=f"Find official benchmark URL for {b.model or b.dataset}",
                ),
            )

    if not out:
        _append_pending_once(
            pending,
            ResearchTask(
                target="benchmarks",
                reason="no benchmark catalog links parsed from stage3_benchmarks or reports",
                hint=f"List common benchmark pages with descriptions for {raw.get('topic', 'this topic')}",
            ),
        )
    return out


def _build_frontiers(
    raw: dict[str, Any],
    pending: list[ResearchTask],
    resources: list[Resource] | None = None,
) -> list[Frontier]:
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
        text = f"{f.name} {f.description}"
        if not f.name or key in seen or key in _GENERIC_FRONTIER_NAMES:
            continue
        if not (_is_recent_frontier_text(text) or _is_open_source_text(text)):
            continue
        seen.add(key)
        out.append(Frontier(name=f.name, description=f.description, importance="source-backed trend"))

    for resource in resources or []:
        if not _resource_supports_frontier(resource):
            continue
        key = (resource.url or resource.name).lower()
        if key in seen:
            continue
        seen.add(key)
        importance = "open-source project" if resource.type in {"github_repo", "model_or_space"} else "recent frontier paper"
        out.append(
            Frontier(
                name=resource.name,
                description=resource.description,
                importance=importance,
                related_methods=resource.related_methods,
                related_papers=resource.related_papers,
            )
        )

    if not out:
        _append_pending_once(
            pending,
            ResearchTask(
                target="frontiers",
                reason="no frontier trends parsed from stage3 reports",
                hint=f"List 5-8 current frontier directions for {raw.get('topic', 'this topic')}",
            ),
        )
    return out


def _method_aliases(name: str) -> set[str]:
    aliases = {name}
    for part in re.split(r"[与和及/、+]+", name):
        part = part.strip()
        if len(part) >= 2:
            aliases.add(part)

    lower = name.lower()
    if "传统" in name or "统计" in name or "声学" in name:
        aliases.update({"hmm", "gmm", "acoustic modeling", "deep neural networks for acoustic modeling"})
    if "端到端" in name or "end-to-end" in lower:
        aliases.update({"end-to-end", "deep speech", "ctc", "connectionist temporal classification"})
    if "自监督" in name or "self-supervised" in lower:
        aliases.update({"self-supervised", "wav2vec", "wav2vec 2.0", "ssl"})
    if "注意力" in name or "attention" in lower or "transformer" in lower:
        aliases.update({"attention-based", "transformer", "conformer"})
    if "多任务" in name or "multi-task" in lower:
        aliases.update({"multi-task", "multitask", "speechx"})
    if "迁移" in name or "多语言" in name:
        aliases.update({"transfer learning", "xlsr", "multilingual"})
    if "混合" in name or "conformer" in lower:
        aliases.update({"hybrid", "conformer"})
    return {a for a in aliases if a}


def _is_broad_paper_for_method(paper: Paper) -> bool:
    return _norm_text(paper.title) in _BROAD_TIMELINE_TITLES


def _link_methods_to_papers(methods: list[Method], papers: list[Paper]) -> list[Method]:
    linked: list[Method] = []
    for method in methods:
        refs: list[str] = []
        aliases = _method_aliases(method.name)
        for paper in papers:
            if _is_broad_paper_for_method(paper):
                continue
            haystack = f"{paper.title} {paper.summary}"
            if any(_has_text_hit(alias, haystack) for alias in aliases):
                refs.append(paper.id)
        linked.append(method.model_copy(update={"papers": refs or method.papers}))
    return linked


def _drop_incomplete_methods(methods: list[Method], pending: list[ResearchTask]) -> list[Method]:
    complete: list[Method] = []
    for method in methods:
        if method.description and method.pros and method.cons and method.papers:
            complete.append(method)
            continue
        _append_pending_once(
            pending,
            ResearchTask(
                target=f"method:{method.id}.papers",
                reason="method details were incomplete or had no representative paper, so it was omitted from visualization_data.methods",
                hint=f"Find at least one representative paper plus pros/cons for method: {method.name}",
            ),
        )
    return complete


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


def _parse_graph_relations(graph_relations: str) -> dict[str, Any]:
    """解析 graph_relations JSON 字符串"""
    if not graph_relations:
        return {
            "paper_relations": [],
            "method_comparisons": [],
            "evolution_chains": [],
            "concept_hierarchy": [],
        }
    try:
        # 尝试直接解析
        data = json.loads(graph_relations)
        return {
            "paper_relations": data.get("paper_relations", []),
            "method_comparisons": data.get("method_comparisons", []),
            "evolution_chains": data.get("evolution_chains", []),
            "concept_hierarchy": data.get("concept_hierarchy", []),
        }
    except json.JSONDecodeError:
        # 尝试提取 JSON 块
        try:
            for line in graph_relations.split("\n"):
                if line.strip().startswith("{"):
                    start = graph_relations.find(line.strip())
                    for i in range(start, len(graph_relations)):
                        if graph_relations[i] == "{":
                            bracket_count = 0
                            for j in range(i, len(graph_relations)):
                                if graph_relations[j] == "{":
                                    bracket_count += 1
                                elif graph_relations[j] == "}":
                                    bracket_count -= 1
                                    if bracket_count == 0:
                                        json_str = graph_relations[i:j+1]
                                        data = json.loads(json_str)
                                        return {
                                            "paper_relations": data.get("paper_relations", []),
                                            "method_comparisons": data.get("method_comparisons", []),
                                            "evolution_chains": data.get("evolution_chains", []),
                                            "concept_hierarchy": data.get("concept_hierarchy", []),
                                        }
                                        break
                    break
        except Exception:
            pass
    return {
        "paper_relations": [],
        "method_comparisons": [],
        "evolution_chains": [],
        "concept_hierarchy": [],
    }


def _build_enhanced_graph(
    topic: str,
    methods: list[Method],
    papers: list[Paper],
    benchmarks: list[Benchmark],
    frontiers: list[Frontier],
    resources: list[Resource] | None,
    relations: dict[str, Any],
) -> KnowledgeGraph:
    """构建增强版知识图谱（包含 LLM 抽取的关系）"""
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

    # ── 1. 基础节点：方法分类 ───────────────────────────────────
    cat_ids: dict[str, str] = {}
    for method in methods:
        if method.category:
            cid = cat_ids.setdefault(method.category, _slug(method.category, prefix="c_"))
            add(cid, method.category, "concept")
            connect(cid, topic_id, "belongs_to")
        add(method.id, method.name, "method")
        connect(method.id, cat_ids.get(method.category, topic_id), "belongs_to")

    # ── 2. 论文节点 ────────────────────────────────────────────
    paper_lookup = {p.id: p for p in papers}
    paper_title_to_id: dict[str, str] = {}
    for paper in papers:
        add(paper.id, paper.title[:45] or paper.id, "paper")
        paper_title_to_id[paper.title.lower()] = paper.id

    # ── 3. Benchmark 节点 ──────────────────────────────────────
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

    # ── 4. Frontier 节点 ───────────────────────────────────────
    for i, frontier in enumerate(frontiers):
        fid = _slug(frontier.name or f"frontier_{i}", prefix="tr_")
        add(fid, frontier.name, "trend")
        connect(fid, topic_id, "related_to")
        for method_id in frontier.related_methods:
            connect(fid, method_id, "related_to")
        for paper_id in frontier.related_papers:
            if paper_id in paper_lookup:
                connect(fid, paper_id, "related_to")

    # ── 5. Resource 节点 ───────────────────────────────────────
    for resource in resources or []:
        add(resource.id, resource.name, "resource")
        connect(resource.id, topic_id, "related_to")
        for method_id in resource.related_methods:
            connect(resource.id, method_id, "related_to")
        for paper_id in resource.related_papers:
            if paper_id in paper_lookup:
                connect(resource.id, paper_id, "related_to")

    # ── 6. LLM 抽取的论文间关系 ────────────────────────────────
    for rel in relations.get("paper_relations", []):
        paper_a_title = rel.get("paper_a", "")
        paper_b_title = rel.get("paper_b", "")
        relation_type = rel.get("relation", "related_to")

        # 尝试通过标题匹配论文节点
        aid = paper_title_to_id.get(paper_a_title.lower())
        bid = paper_title_to_id.get(paper_b_title.lower())

        if aid and bid:
            # 映射关系类型到 EdgeRelation
            edge_rel = _map_relation_to_edge_type(relation_type)
            connect(aid, bid, edge_rel)

    # ── 7. LLM 抽取的方法对比关系 ──────────────────────────────
    for comp in relations.get("method_comparisons", []):
        winner = comp.get("winner", "")
        loser = comp.get("loser", "")

        # 尝试匹配方法节点
        winner_id = _find_method_id(methods, winner)
        loser_id = _find_method_id(methods, loser)

        if winner_id and loser_id:
            connect(winner_id, loser_id, "compares_with")

    # ── 8. LLM 抽取的技术演进链 ────────────────────────────────
    for chain_data in relations.get("evolution_chains", []):
        chain = chain_data.get("chain", [])
        if len(chain) >= 2:
            # 构建演进链边
            for i in range(len(chain) - 1):
                prev_id = _find_method_id(methods, chain[i])
                next_id = _find_method_id(methods, chain[i + 1])
                if prev_id and next_id:
                    connect(prev_id, next_id, "succeeds")

    # ── 9. LLM 抽取的概念层次结构 ─────────────────────────────
    for hier in relations.get("concept_hierarchy", []):
        concept = hier.get("concept", "")
        sub_concepts = hier.get("sub_concepts", [])
        parent = hier.get("parent", "")

        # 创建概念节点
        concept_id = _slug(concept, prefix="c_")
        add(concept_id, concept, "concept")

        if parent:
            parent_id = _slug(parent, prefix="c_")
            add(parent_id, parent, "concept")
            connect(concept_id, parent_id, "belongs_to")

        # 连接子概念
        for sub in sub_concepts:
            sub_id = _slug(sub, prefix="c_")
            add(sub_id, sub, "concept")
            connect(sub_id, concept_id, "belongs_to")

    # ── 10. 论文与方法的基础关联（原有逻辑）───────────────────
    for paper in papers:
        connected = False
        for method in methods:
            if paper.id in method.papers or _has_text_hit(method.name, f"{paper.title} {paper.summary}"):
                connect(paper.id, method.id, "proposes")
                connected = True
        if not connected:
            connect(paper.id, topic_id, "related_to")

    return KnowledgeGraph(nodes=nodes, edges=edges)


def _map_relation_to_edge_type(relation: str) -> str:
    """将 LLM 抽取的关系类型映射到 EdgeRelation"""
    mapping = {
        "cites": "cites",
        "improves_on": "improves_on",
        "competes_with": "competes_with",
        "complementary": "complementary",
        "succeeds": "succeeds",
        "related_to": "related_to",
    }
    return mapping.get(relation.lower(), "related_to")


def _find_method_id(methods: list[Method], name: str) -> str | None:
    """通过名称模糊匹配方法节点 ID"""
    if not name:
        return None
    name_lower = name.lower()
    for method in methods:
        if name_lower in method.name.lower() or method.name.lower() in name_lower:
            return method.id
    return None


def rebuild_graph(viz: VisualizationData) -> VisualizationData:
    """Rebuild graph from the current refined entities."""

    return viz.model_copy(update={
        "graph": _build_graph(viz.topic, viz.methods, viz.papers, viz.benchmarks, viz.frontiers, viz.resources)
    })


def writer_json_to_visualization(raw: dict[str, Any], graph_relations: str = "") -> VisualizationData:
    """Enhanced conversion from step3_writer_done.json to VisualizationData.

    Args:
        raw: step3_writer_done.json 原始数据
        graph_relations: WriterReport.graph_relations (LLM抽取的JSON格式图谱关系)
    """

    topic = str(raw.get("topic") or "Unknown Topic")
    pending: list[ResearchTask] = []

    overview = _build_overview(raw, pending)
    methods = _build_methods(raw, pending)
    papers, title_to_id = _build_papers(raw, pending)
    methods = _link_methods_to_papers(methods, papers)
    methods = _drop_incomplete_methods(methods, pending)
    resources = _build_resources(raw, methods, papers)
    papers = _backfill_paper_urls_from_resources(papers, resources)
    timeline = _build_timeline(raw, papers, title_to_id, pending)
    benchmarks = _build_benchmarks(raw, pending)
    frontiers = _link_frontiers(_build_frontiers(raw, pending, resources), methods, papers)

    # 解析 graph_relations
    parsed_relations = _parse_graph_relations(graph_relations)

    # 构建图谱（支持增强关系）
    graph = _build_enhanced_graph(
        topic, methods, papers, benchmarks, frontiers, resources, parsed_relations
    )

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
