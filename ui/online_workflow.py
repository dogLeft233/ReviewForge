"""Online Streamlit workflow for ReviewForge.

This module keeps the UI orchestration separate from ``ui/app.py`` so the
offline visualization dashboard can stay small and predictable.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from src.adapter import convert, convert_with_llm
from src.adapter.schema import VisualizationData
from src.core import CoreConfig, PipelineResult, ReviewForge, _now, _slugify
from src.visualizer.quality import assess_visualization_data

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TMP_ROOT = _PROJECT_ROOT / "tmp"
_DEFAULT_ENHANCE_CONFIG = "adapter_enhance.yaml"


@dataclass(frozen=True)
class OnlineOptions:
    resume_existing: bool
    use_llm_refine: bool
    use_openai_enhance: bool
    enhance_config: str
    use_snapshot: bool
    snapshot_mode: str
    snapshot_fields_to_run: list[str]
    bfs_enabled: bool
    bfs_expand_layers: int
    bfs_expand_papers_count: int
    bfs_search_queries_count: int
    bfs_search_papers_count: int
    writer_max_tokens: int


def render(*, compact: bool = False) -> tuple[VisualizationData | None, str]:
    """Render the online workflow panel and return generated data if available."""

    if compact:
        st.header("在线检索")
        st.caption("重新生成或调整增强策略。")
    else:
        st.subheader("在线领域检索")
        st.caption("输入领域后自动执行 Explorer、文献检索、综述解析、可视化数据导出与可选 OpenAI Responses 增强。")

    topic = st.text_input(
        "研究领域",
        key="online_topic",
        placeholder="例如：ASR 自动语音识别、多模态大模型评测、LoRA 微调",
    ).strip()

    options = _render_options(topic, compact=compact)

    if compact:
        run_clicked = st.button("开始在线生成", type="primary", use_container_width=True)
        clear_clicked = st.button("清除当前在线结果", use_container_width=True)
    else:
        col_run, col_clear = st.columns([1, 1])
        with col_run:
            run_clicked = st.button("开始在线生成", type="primary", use_container_width=True)
        with col_clear:
            clear_clicked = st.button("清除当前在线结果", use_container_width=True)

    if clear_clicked:
        st.session_state.pop("online_visualization_data", None)
        st.session_state.pop("online_artifacts", None)
        st.rerun()

    if run_clicked:
        if not topic:
            st.warning("请先输入一个研究领域。")
        else:
            _run_and_store(topic, options)

    artifacts = st.session_state.get("online_artifacts")
    data = st.session_state.get("online_visualization_data")
    if isinstance(artifacts, dict):
        _render_artifacts(artifacts, compact=compact)
    if isinstance(data, VisualizationData):
        source_label = st.session_state.get("online_source_label", "在线生成 visualization_data.json")
        return data, source_label
    return None, ""


def _render_options(topic: str, *, compact: bool = False) -> OnlineOptions:
    with st.expander("运行选项", expanded=not compact):
        if compact:
            resume_existing = st.checkbox("复用已有中间结果", value=True)
            use_llm_refine = st.checkbox("导出时运行本地 LLM refine", value=True)
            use_openai_enhance = st.checkbox("启用 OpenAI Responses 增强", value=True)
            enhance_config = st.text_input("增强配置文件", value=_DEFAULT_ENHANCE_CONFIG)
            bfs_enabled = st.checkbox("启用 BFS 文献检索", value=True)
            writer_max_tokens = st.number_input("Writer max tokens", min_value=1024, max_value=16000, value=4096, step=512)
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                resume_existing = st.checkbox("复用已有中间结果", value=True)
                use_llm_refine = st.checkbox("导出时运行本地 LLM refine", value=True)
            with c2:
                use_openai_enhance = st.checkbox("启用 OpenAI Responses 增强", value=True)
                enhance_config = st.text_input("增强配置文件", value=_DEFAULT_ENHANCE_CONFIG)
            with c3:
                bfs_enabled = st.checkbox("启用 BFS 文献检索", value=True)
                writer_max_tokens = st.number_input("Writer max tokens", min_value=1024, max_value=16000, value=4096, step=512)

        st.divider()
        st.markdown("**检索参数**")
        if compact:
            bfs_expand_layers = st.number_input("BFS 层数", min_value=0, max_value=4, value=2, step=1)
            bfs_expand_papers_count = st.number_input("每层扩展论文数", min_value=1, max_value=50, value=15, step=1)
            bfs_search_queries_count = st.number_input("搜索词数量", min_value=1, max_value=20, value=5, step=1)
            bfs_search_papers_count = st.number_input("每词取论文数", min_value=1, max_value=50, value=20, step=1)
        else:
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                bfs_expand_layers = st.number_input("BFS 层数", min_value=0, max_value=4, value=2, step=1)
            with s2:
                bfs_expand_papers_count = st.number_input("每层扩展论文数", min_value=1, max_value=50, value=15, step=1)
            with s3:
                bfs_search_queries_count = st.number_input("搜索词数量", min_value=1, max_value=20, value=5, step=1)
            with s4:
                bfs_search_papers_count = st.number_input("每词取论文数", min_value=1, max_value=50, value=20, step=1)

    use_snapshot = True
    snapshot_mode = "existing"
    selected_fields: list[str] = []
    if use_openai_enhance:
        with st.expander("OpenAI 增强 snapshot", expanded=not compact):
            use_snapshot = st.checkbox("使用 snapshot 控制增强模块", value=True)
            if use_snapshot:
                snapshot_path = _topic_dir(topic) / "snapshot.json" if topic else None
                if snapshot_path:
                    st.caption(f"snapshot: `{snapshot_path}`")
                snapshot_mode = st.radio(
                    "本次增强策略",
                    options=["existing", "rerun_all", "skip_all", "custom"],
                    format_func={
                        "existing": "沿用现有 snapshot；没有则全部重跑",
                        "rerun_all": "重跑全部增强字段",
                        "skip_all": "跳过全部增强字段，只导出基础数据",
                        "custom": "自定义本次需要重跑的字段",
                    }.__getitem__,
                    horizontal=False,
                )
                all_fields = _snapshot_field_names()
                default_selected = _existing_snapshot_true_fields(snapshot_path, all_fields) if snapshot_path else all_fields
                if snapshot_mode == "custom":
                    selected_fields = st.multiselect(
                        "需要 OpenAI 重跑的字段",
                        options=all_fields,
                        default=default_selected,
                    )
                elif snapshot_mode == "rerun_all":
                    selected_fields = all_fields
                elif snapshot_mode == "skip_all":
                    selected_fields = []
                else:
                    selected_fields = default_selected
            else:
                st.info("不使用 snapshot 时，OpenAI 增强不会写入字段开关；每次运行都会按增强器默认策略尝试执行。")

    return OnlineOptions(
        resume_existing=resume_existing,
        use_llm_refine=use_llm_refine,
        use_openai_enhance=use_openai_enhance,
        enhance_config=enhance_config,
        use_snapshot=use_snapshot,
        snapshot_mode=snapshot_mode,
        snapshot_fields_to_run=selected_fields,
        bfs_enabled=bfs_enabled,
        bfs_expand_layers=int(bfs_expand_layers),
        bfs_expand_papers_count=int(bfs_expand_papers_count),
        bfs_search_queries_count=int(bfs_search_queries_count),
        bfs_search_papers_count=int(bfs_search_papers_count),
        writer_max_tokens=int(writer_max_tokens),
    )


def _run_and_store(topic: str, options: OnlineOptions) -> None:
    with st.status("在线流程运行中", expanded=True) as status:
        events: list[dict[str, Any]] = []

        def emit(stage: str, message: str, **extra: Any) -> None:
            events.append({"stage": stage, "message": message, **extra})
            status.update(label=message, state="running")
            st.write(message)

        try:
            data, artifacts = _execute_online_pipeline(topic, options, emit)
        except Exception as exc:
            status.update(label="在线流程失败", state="error")
            logger.exception("online workflow failed")
            st.error(f"在线流程失败：{exc}")
            st.session_state["online_artifacts"] = {
                "topic": topic,
                "events": events,
                "error": str(exc),
            }
            return

        artifacts["events"] = events
        st.session_state["online_visualization_data"] = data
        st.session_state["online_artifacts"] = artifacts
        st.session_state["online_source_label"] = f"在线生成：{artifacts['viz_path']}"
        status.update(label="在线流程完成", state="complete")
    st.rerun()


def _execute_online_pipeline(
    topic: str,
    options: OnlineOptions,
    emit: Callable[..., None],
) -> tuple[VisualizationData, dict[str, Any]]:
    slug = _slugify(topic)
    tmp_dir = _TMP_ROOT / slug
    tmp_dir.mkdir(parents=True, exist_ok=True)
    emit("prepare", f"准备工作目录：`{tmp_dir}`")

    rf = _build_reviewforge(options)
    result = _load_latest_result(tmp_dir) if options.resume_existing else None
    if result and result.topic != topic:
        result.topic = topic
    if result:
        emit("resume", f"复用已有阶段文件，当前 step=`{result.step}`。")
    else:
        result = PipelineResult(topic=topic, tmp_dir=str(tmp_dir), created_at=_now())

    searcher_keywords = ""

    if result.step not in {"explorer_done", "searcher_done", "complete"} or not result.explorer_report:
        emit("explorer", "Step 1/4 Explorer：开始理解领域、收集经典论文和研究脉络。")
        t0 = time.time()
        result.explorer_report = rf.explorer.run(topic)
        result.explorer_elapsed_seconds = time.time() - t0
        result.step = "explorer_done"
        result.save(tmp_dir / "step1_explorer_done.json")
        total_queries = getattr(result.explorer_report, "total_queries", 0)
        emit("explorer", f"Explorer 完成，用时 {result.explorer_elapsed_seconds:.1f}s，查询 {total_queries} 次。")
    else:
        emit("explorer", "Step 1/4 Explorer：已从中间文件恢复。")

    if result.step == "explorer_done":
        emit("searcher", "Step 2/4 Searcher：生成检索词并执行文献检索。")
        t0 = time.time()
        messages: list[dict[str, str]] = []
        search_text = ""
        papers: list[Any] = []
        searcher_keywords = rf.searcher.run_with_explorer_report(topic, result.explorer_report)
        emit("searcher", "SearcherAgent 已生成检索策略。")
        if options.bfs_enabled:
            try:
                emit("searcher", "BFSSearcher 主检索开始。")
                papers = rf.bfs_searcher.search(topic)
                search_text = rf._format_papers_for_writer(papers)
                emit("searcher", f"BFSSearcher 完成，获得 {len(papers)} 篇候选论文。")
            except Exception as exc:
                emit("searcher", f"BFSSearcher 失败，切换 MultiSourceSearcher：{exc}")
                papers = []
                search_text = ""
        if not papers:
            search_text, messages = rf.multi_searcher.run(topic, messages=messages)
            emit("searcher", "MultiSourceSearcher 完成备用检索。")
        result.searcher_messages = messages
        result.searcher_result = search_text
        result.searcher_papers = papers
        result.searcher_elapsed_seconds = time.time() - t0
        result.step = "searcher_done"
        result.save(tmp_dir / "step2_searcher_done.json")
        emit("searcher", f"Searcher 完成，用时 {result.searcher_elapsed_seconds:.1f}s。")
    else:
        emit("searcher", "Step 2/4 Searcher：已从中间文件恢复。")

    if result.step == "searcher_done":
        emit("writer", "Step 3/4 Writer：基于探索与检索结果生成综述结构。")
        t0 = time.time()
        result.writer_report = rf.writer.write(topic=topic, explorer_report=result.explorer_report)
        result.writer_elapsed_seconds = time.time() - t0
        result.step = "complete"
        result.save(tmp_dir / "step3_writer_done.json")
        emit("writer", f"Writer 完成，用时 {result.writer_elapsed_seconds:.1f}s。")
    else:
        emit("writer", "Step 3/4 Writer：已从中间文件恢复。")

    emit("export", "Step 4/4 Export：解析综述结果并生成 visualization_data.json。")
    step3_path = tmp_dir / "step3_writer_done.json"
    viz_path = tmp_dir / "visualization_data.json"
    snapshot_path = tmp_dir / "snapshot.json" if options.use_snapshot else None
    if options.use_openai_enhance and options.use_snapshot:
        _prepare_snapshot(snapshot_path, options)
    data = _export_visualization_data(
        step3_path,
        viz_path,
        llm=rf.llm if options.use_llm_refine else None,
        use_llm_refine=options.use_llm_refine,
        use_openai_enhance=options.use_openai_enhance,
        enhance_config=options.enhance_config,
        snapshot_path=snapshot_path,
        reuse_existing_base=options.resume_existing,
    )
    emit("export", f"导出完成：`{viz_path}`。")

    artifacts = {
        "topic": topic,
        "tmp_dir": str(tmp_dir),
        "viz_path": str(viz_path),
        "snapshot_path": str(snapshot_path) if snapshot_path else "",
        "step_files": {
            "Explorer": str(tmp_dir / "step1_explorer_done.json"),
            "Searcher": str(tmp_dir / "step2_searcher_done.json"),
            "Writer": str(step3_path),
        },
        "searcher_keywords": searcher_keywords,
        "searcher_papers": _serialize_papers(result.searcher_papers),
        "searcher_result": result.searcher_result,
        "searcher_messages": result.searcher_messages,
        "quality": assess_visualization_data(data),
        "quality_report": data.quality_report,
    }
    return data, artifacts


def _build_reviewforge(options: OnlineOptions) -> ReviewForge:
    from src.config import settings
    from src.llm import LLM

    llm = LLM(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        max_concurrency=settings.llm_max_concurrency,
        timeout_seconds=300.0,
        max_tokens=4096,
    )
    config = CoreConfig(
        writer_max_tokens=options.writer_max_tokens,
        verbose=False,
        bfs_enabled=options.bfs_enabled,
        bfs_expand_layers=options.bfs_expand_layers,
        bfs_expand_papers_count=options.bfs_expand_papers_count,
        bfs_search_queries_count=options.bfs_search_queries_count,
        bfs_search_papers_count=options.bfs_search_papers_count,
    )
    return ReviewForge(llm=llm, config=config, tmp_root=_TMP_ROOT)


def _export_visualization_data(
    step3_path: Path,
    output_path: Path,
    *,
    llm: Any | None,
    use_llm_refine: bool,
    use_openai_enhance: bool,
    enhance_config: str,
    snapshot_path: Path | None,
    reuse_existing_base: bool,
) -> VisualizationData:
    raw = json.loads(step3_path.read_text(encoding="utf-8"))
    if use_openai_enhance:
        from src.adapter.openai_responses_enhancer import enhance_with_openai_responses

        if reuse_existing_base and snapshot_path and snapshot_path.exists() and output_path.exists():
            base = VisualizationData.model_validate_json(output_path.read_text(encoding="utf-8"))
        else:
            base = convert_with_llm(raw, llm=llm) if use_llm_refine else convert(raw)
        data = enhance_with_openai_responses(
            base,
            raw,
            config_path=enhance_config,
            snapshot_path=snapshot_path,
            force=True,
        )
    else:
        data = convert_with_llm(raw, llm=llm) if use_llm_refine else convert(raw)

    output_path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
    return data


def _render_artifacts(artifacts: dict[str, Any], *, compact: bool = False) -> None:
    st.divider()
    if compact:
        st.markdown("**中间过程**")
    else:
        st.subheader("在线流程中间过程")
    if artifacts.get("error"):
        st.error(artifacts["error"])

    events = artifacts.get("events") or []
    if events:
        with st.expander("运行日志", expanded=False):
            for event in events:
                st.write(f"**{event.get('stage', '')}** · {event.get('message', '')}")

    quality = artifacts.get("quality") or {}
    score = (artifacts.get("quality_report") or {}).get("overall_score", "N/A")
    if compact:
        metric_columns = st.columns(2)
        with metric_columns[0]:
            st.metric("候选论文", len(artifacts.get("searcher_papers") or []))
        with metric_columns[1]:
            st.metric("图节点", quality.get("graph", {}).get("nodes", 0))
        metric_columns = st.columns(2)
        with metric_columns[0]:
            st.metric("资源链接", quality.get("resources", {}).get("count", 0))
        with metric_columns[1]:
            st.metric("增强评分", score)
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("候选论文", len(artifacts.get("searcher_papers") or []))
        with m2:
            st.metric("图节点", quality.get("graph", {}).get("nodes", 0))
        with m3:
            st.metric("资源链接", quality.get("resources", {}).get("count", 0))
        with m4:
            st.metric("增强评分", score)

    if compact:
        with st.expander("文献检索", expanded=False):
            _render_search_process(artifacts, compact=True)
        with st.expander("阶段文件", expanded=False):
            _render_step_files(artifacts, show_content=False)
        with st.expander("Snapshot", expanded=False):
            _render_snapshot(artifacts, compact=True)
        with st.expander("质量报告", expanded=False):
            st.json({
                "quality": artifacts.get("quality") or {},
                "quality_report": artifacts.get("quality_report") or {},
            }, expanded=False)
        return

    tabs = st.tabs(["文献检索", "阶段文件", "Snapshot", "质量报告"])
    with tabs[0]:
        _render_search_process(artifacts)
    with tabs[1]:
        _render_step_files(artifacts, show_content=True)
    with tabs[2]:
        _render_snapshot(artifacts)
    with tabs[3]:
        st.json({
            "quality": artifacts.get("quality") or {},
            "quality_report": artifacts.get("quality_report") or {},
        }, expanded=False)


def _render_search_process(artifacts: dict[str, Any], *, compact: bool = False) -> None:
    keywords = artifacts.get("searcher_keywords") or ""
    if keywords:
        with st.expander("SearcherAgent 检索策略", expanded=True):
            st.markdown(keywords)

    papers = artifacts.get("searcher_papers") or []
    if papers:
        if compact:
            for paper in papers[:5]:
                title = paper.get("title") or "(untitled)"
                year = paper.get("year") or "n.d."
                st.markdown(f"- **{title}** ({year})")
            if len(papers) > 5:
                st.caption(f"还有 {len(papers) - 5} 篇候选论文。")
            return
        rows = [
            {
                "title": p.get("title", ""),
                "year": p.get("year", ""),
                "venue": p.get("venue", ""),
                "cited_by": p.get("cited_by", ""),
                "score": p.get("select_score", ""),
            }
            for p in papers
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        with st.expander("论文摘要与检索文本", expanded=False):
            st.markdown(artifacts.get("searcher_result") or "")
    else:
        st.info("当前没有结构化候选论文，可能使用了备用检索文本。")
        if artifacts.get("searcher_result"):
            st.markdown(artifacts["searcher_result"])

    messages = artifacts.get("searcher_messages") or []
    if messages:
        with st.expander("MultiSourceSearcher messages", expanded=False):
            st.json(messages, expanded=False)


def _render_step_files(artifacts: dict[str, Any], *, show_content: bool) -> None:
    for label, file_path in (artifacts.get("step_files") or {}).items():
        path = Path(file_path)
        with st.expander(f"{label}: {path.name}", expanded=False):
            st.caption(str(path))
            if not path.exists():
                st.warning("文件不存在。")
                continue
            st.caption(f"{path.stat().st_size / 1024:.1f} KB")
            if not show_content:
                continue
            text = path.read_text(encoding="utf-8")
            if len(text) > 30000:
                st.code(text[:30000] + "\n\n... truncated ...", language="json")
            else:
                st.code(text, language="json")


def _render_snapshot(artifacts: dict[str, Any], *, compact: bool = False) -> None:
    snapshot_path = artifacts.get("snapshot_path")
    if not snapshot_path:
        st.info("本次未启用 snapshot。")
        return
    path = Path(snapshot_path)
    st.caption(str(path))
    if not path.exists():
        st.warning("snapshot.json 尚未生成。")
        return
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    fields = snapshot.get("fields") or {}
    rows = [{"field": key, "rerun_next_time": bool(value)} for key, value in sorted(fields.items())]
    if compact:
        for row in rows:
            mark = "rerun" if row["rerun_next_time"] else "skip"
            st.caption(f"{mark}: {row['field']}")
    else:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander("snapshot.json", expanded=False):
        st.json(snapshot, expanded=False)


def _load_latest_result(tmp_dir: Path) -> PipelineResult | None:
    for name in ("step3_writer_done.json", "step2_searcher_done.json", "step1_explorer_done.json"):
        path = tmp_dir / name
        if path.exists():
            return PipelineResult.load(path)
    return None


def _topic_dir(topic: str) -> Path:
    return _TMP_ROOT / _slugify(topic)


def _snapshot_field_names() -> list[str]:
    try:
        from src.adapter.openai_responses_enhancer import _default_snapshot, _enhancement_modules

        snapshot = _default_snapshot(_enhancement_modules(VisualizationData()), needs_rerun=True)
        return sorted(snapshot.get("fields", {}).keys())
    except Exception:
        return [
            "benchmarks.description",
            "frontiers.description",
            "frontiers.items",
            "frontiers.resources",
            "methods.cons",
            "methods.papers",
            "methods.pros",
            "papers.url",
            "resources.method_links",
            "resources.paper_links",
            "timeline.related_papers",
        ]


def _existing_snapshot_true_fields(snapshot_path: Path | None, all_fields: list[str]) -> list[str]:
    if not snapshot_path or not snapshot_path.exists():
        return all_fields
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        fields = snapshot.get("fields") or {}
        return [field for field in all_fields if bool(fields.get(field, True))]
    except Exception:
        return all_fields


def _prepare_snapshot(snapshot_path: Path | None, options: OnlineOptions) -> None:
    if not snapshot_path or options.snapshot_mode == "existing":
        return
    all_fields = _snapshot_field_names()
    selected = set(options.snapshot_fields_to_run)
    snapshot = {
        "version": 1,
        "description": "True means the field should be regenerated by OpenAI Responses API on the next export.",
        "updated_at": _now(),
        "fields": {field: field in selected for field in all_fields},
        "modules": _snapshot_modules_for_fields(all_fields),
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def _snapshot_modules_for_fields(all_fields: list[str]) -> dict[str, list[str]]:
    try:
        from src.adapter.openai_responses_enhancer import _default_snapshot, _enhancement_modules

        snapshot = _default_snapshot(_enhancement_modules(VisualizationData()), needs_rerun=True)
        return snapshot.get("modules", {})
    except Exception:
        return {"custom": all_fields}


def _serialize_papers(papers: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for paper in papers or []:
        rows.append({
            "paper_id": getattr(paper, "paper_id", ""),
            "title": getattr(paper, "title", ""),
            "authors": getattr(paper, "authors", []),
            "year": getattr(paper, "year", ""),
            "venue": getattr(paper, "venue", ""),
            "abstract": getattr(paper, "abstract", ""),
            "cited_by": getattr(paper, "cited_by", ""),
            "select_score": getattr(paper, "select_score", ""),
        })
    return rows
