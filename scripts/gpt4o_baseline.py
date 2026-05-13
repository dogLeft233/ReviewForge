#!/usr/bin/env python3
"""Generate a ReviewForge-compatible baseline with OpenAI Responses API.

This script intentionally does not use ReviewForge Explorer/Searcher/BFS/Writer.
The model is asked to research with the Responses API web search tool and return
the same JSON surface that ``scripts/run_pipeline.py`` eventually writes.

Usage:
    python scripts/gpt4o_baseline.py -t "ASR自动语音识别"
    python scripts/gpt4o_baseline.py -t "ASR自动语音识别" --config adapter_enhance.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI gpt-4o baseline that writes ReviewForge-compatible artifacts."
    )
    parser.add_argument("--topic", "-t", type=str, required=True, help="研究领域/综述主题")
    parser.add_argument(
        "--config",
        type=str,
        default="adapter_enhance.yaml",
        help="OpenAI Responses API 配置，默认复用 adapter_enhance.yaml",
    )
    parser.add_argument("--model", type=str, default=None, help="覆盖配置中的 openai.model")
    parser.add_argument(
        "--tmp-root",
        type=str,
        default=None,
        help="输出根目录，默认 baseline/",
    )
    parser.add_argument("--api-key", type=str, default=None, help="覆盖配置中的 openai.api_key")
    parser.add_argument("--base-url", type=str, default=None, help="覆盖配置中的 openai.base_url")
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--web-tool",
        choices=["web_search_preview", "web_search"],
        default=None,
        help="覆盖配置中的 web_search.tool_type",
    )
    parser.add_argument(
        "--search-context-size",
        choices=["low", "medium", "high"],
        default="high",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="只写 step*.json，不生成 visualization_data.json",
    )
    args = parser.parse_args()

    from src.adapter.openai_responses_enhancer import load_openai_enhance_config

    cfg = load_openai_enhance_config(args.config)
    api_key = args.api_key or cfg.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"错误: {args.config} 中缺少 openai.api_key，也未设置 OPENAI_API_KEY。")
        sys.exit(2)
    base_url = (args.base_url or cfg.base_url or "https://api.openai.com/v1").rstrip("/")
    model = args.model or cfg.model
    timeout_seconds = args.timeout_seconds or cfg.timeout_seconds
    max_output_tokens = args.max_output_tokens or cfg.max_output_tokens
    temperature = args.temperature if args.temperature is not None else cfg.temperature
    web_tool = args.web_tool or cfg.web_search_tool_type

    tmp_root = Path(args.tmp_root) if args.tmp_root else _ROOT / "baseline"
    tmp_dir = tmp_root / _slugify(args.topic)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Topic: {args.topic}")
    print(f"Config: {args.config}")
    print(f"API: {base_url}/responses")
    print(f"Model: {model}")
    print(f"Output: {tmp_dir}")
    print("Running OpenAI Responses API research baseline...")

    started = time.time()
    response = _create_response(
        api_key=api_key,
        base_url=base_url,
        topic=args.topic,
        model=model,
        web_tool=web_tool,
        allowed_domains=cfg.allowed_domains,
        search_context_size=args.search_context_size,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
    )
    elapsed = time.time() - started

    output_text = _extract_output_text(response)
    generated = _extract_json_object(output_text)
    data = _build_pipeline_result(
        topic=args.topic,
        tmp_dir=tmp_dir,
        generated=generated,
        response=response,
        elapsed_seconds=elapsed,
    )

    _write_step_files(data, tmp_dir)
    print(f"Wrote: {tmp_dir / 'step1_explorer_done.json'}")
    print(f"Wrote: {tmp_dir / 'step2_searcher_done.json'}")
    print(f"Wrote: {tmp_dir / 'step3_writer_done.json'}")

    if not args.no_viz:
        try:
            viz_path = _write_visualization(data, tmp_dir)
            print(f"Wrote: {viz_path}")
        except Exception as exc:
            print(f"Warning: visualization_data.json 生成失败（step3 已写入）: {exc}")

    wr = data.get("writer_report") or {}
    print("\nDone.")
    print(f"Title: {wr.get('title') or '(empty)'}")
    print(f"Elapsed: {elapsed:.1f}s")


def _create_response(
    *,
    api_key: str,
    base_url: str,
    topic: str,
    model: str,
    web_tool: str,
    allowed_domains: list[str],
    search_context_size: str,
    max_output_tokens: int,
    timeout_seconds: float,
    temperature: float,
) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "type": web_tool,
        "search_context_size": search_context_size,
    }
    if allowed_domains and web_tool == "web_search":
        tool["filters"] = {"allowed_domains": allowed_domains}

    payload = {
        "model": model,
        "instructions": _instructions(),
        "input": _user_prompt(topic),
        "tools": [tool],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "store": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout_seconds) as client:
        resp = client.post(f"{base_url}/responses", headers=headers, json=payload)
        if resp.status_code == 400 and "include" in payload:
            # Some model/tool combinations reject optional include fields. Keep
            # the baseline usable while still using Responses API web search.
            payload.pop("include", None)
            resp = client.post(f"{base_url}/responses", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


def _instructions() -> str:
    return """你是一个严谨的科研综述 baseline agent。

你必须使用 OpenAI Responses API 的 web search 工具自行调研，不要假装已经有本地 pipeline 的 Explorer、Searcher、BFS 或 Writer 结果。
目标是输出与 ReviewForge run_pipeline.py 的 step3_writer_done.json 兼容的内容，而不是复刻其内部过程。

要求：
1. 输出语言以中文为主，论文标题、数据集、模型名可保留英文。
2. 优先核查真实论文、官方 benchmark、Papers with Code、arXiv、ACL/ICASSP/Interspeech/NeurIPS/IEEE/ACM 等可信来源。
3. 不要编造论文、作者、年份、benchmark 数值；不确定的内容写成趋势性描述，不给假数值。
4. 最终只输出一个合法 JSON 对象，不要 Markdown 代码块，不要解释性前后缀。
5. JSON 必须包含 explorer_report、searcher_result、searcher_papers、writer_report 四个顶层字段。"""


def _user_prompt(topic: str) -> str:
    return f"""请围绕主题「{topic}」完成一次独立调研，并返回 ReviewForge 兼容 JSON。

JSON schema 形状如下，字段名必须保持一致：
{{
  "explorer_report": {{
    "topic": "{topic}",
    "stage1_overview": "领域定义、核心问题、主流方法的中文 Markdown",
    "stage1_concepts": ["核心概念1", "核心概念2"],
    "stage1_search_results": "阶段1调研记录/总结 Markdown",
    "stage2_classics": [
      {{"title": "论文或经典工作标题", "year": 2020, "authors": "作者", "venue": "会议/期刊", "key_idea": "核心贡献", "impact": "影响"}}
    ],
    "stage2_timeline": "历史演进时间线 Markdown",
    "stage2_search_results": "经典工作与历史阶段 Markdown",
    "stage3_benchmarks": [
      {{"name": "Benchmark/Leaderboard", "url": "https://...", "description": "说明", "metric": "WER/CER/...", "dataset": "数据集", "leaderboard": []}}
    ],
    "stage3_state_of_art": "当前 SOTA 与代表方法 Markdown",
    "stage3_trends": ["趋势1", "趋势2"],
    "stage3_search_results": "前沿、benchmark、leaderboard 调研 Markdown",
    "downstream_report": "给写作阶段使用的综合 Markdown 报告",
    "total_queries": 0
  }},
  "searcher_result": "可供写作引用的论文/资源列表 Markdown，包含标题、作者、年份、来源 URL、摘要式说明",
  "searcher_papers": [
    {{"paper_id": "url-or-doi", "title": "标题", "authors": ["作者1"], "year": 2020, "venue": "会议/期刊", "abstract": "摘要", "cited_by": 0, "select_score": 0.0}}
  ],
  "writer_report": {{
    "plan": "综述写作计划 Markdown",
    "title_candidates": ["候选标题1", "候选标题2", "候选标题3"],
    "title": "最终综述标题",
    "introduction": "引言 Markdown",
    "body": "正文 Markdown，需有清晰章节、方法分类、经典工作、前沿进展、挑战分析",
    "conclusion": "结论 Markdown",
    "abstract": "摘要",
    "keywords": ["关键词1", "关键词2"],
    "references": "参考文献 Markdown 或 GB/T 风格列表，尽量含 URL/DOI/arXiv",
    "benchmarks": "Benchmark 对比 Markdown",
    "trends": "未来趋势 Markdown"
  }}
}}

请让内容足够完整，以便后续 visualization_data.json 适配器能抽取 overview、timeline、papers、methods、benchmarks、frontiers 和 knowledge graph。"""


def _build_pipeline_result(
    *,
    topic: str,
    tmp_dir: Path,
    generated: dict[str, Any],
    response: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    explorer_report = _normalize_explorer_report(topic, generated.get("explorer_report") or {})
    writer_report = _normalize_writer_report(generated.get("writer_report") or {})
    searcher_result = str(generated.get("searcher_result") or "")
    searcher_papers = _normalize_papers(generated.get("searcher_papers") or [])
    sources = _extract_sources(response)

    if sources:
        source_lines = "\n".join(f"- {s}" for s in sources)
        searcher_result = (searcher_result.rstrip() + "\n\n## OpenAI web search sources\n" + source_lines).strip()

    return {
        "topic": topic,
        "step": "complete",
        "tmp_dir": str(tmp_dir),
        "explorer_elapsed_seconds": 0.0,
        "searcher_elapsed_seconds": 0.0,
        "writer_elapsed_seconds": elapsed_seconds,
        "created_at": datetime.now().isoformat(),
        "error": "",
        "explorer_report": explorer_report,
        "searcher_messages": [
            {
                "role": "assistant",
                "content": "Generated by OpenAI Responses API baseline.",
                "response_id": str(response.get("id") or ""),
                "model": str(response.get("model") or ""),
                "sources": sources,
            }
        ],
        "searcher_result": searcher_result,
        "searcher_papers": searcher_papers,
        "writer_report": writer_report,
    }


def _normalize_explorer_report(topic: str, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "topic": str(raw.get("topic") or topic),
        "stage1_overview": str(raw.get("stage1_overview") or ""),
        "stage1_concepts": _str_list(raw.get("stage1_concepts")),
        "stage1_search_results": str(raw.get("stage1_search_results") or raw.get("stage1_overview") or ""),
        "stage2_classics": [
            {
                "title": str(item.get("title") or ""),
                "year": _int(item.get("year")),
                "authors": str(item.get("authors") or ""),
                "venue": str(item.get("venue") or ""),
                "key_idea": str(item.get("key_idea") or ""),
                "impact": str(item.get("impact") or ""),
            }
            for item in _dict_list(raw.get("stage2_classics"))
            if item.get("title")
        ],
        "stage2_timeline": str(raw.get("stage2_timeline") or ""),
        "stage2_search_results": str(raw.get("stage2_search_results") or raw.get("stage2_timeline") or ""),
        "stage3_benchmarks": [
            {
                "name": str(item.get("name") or ""),
                "url": str(item.get("url") or ""),
                "description": str(item.get("description") or ""),
                "metric": str(item.get("metric") or ""),
                "dataset": str(item.get("dataset") or ""),
                "leaderboard": item.get("leaderboard") if isinstance(item.get("leaderboard"), list) else [],
            }
            for item in _dict_list(raw.get("stage3_benchmarks"))
            if item.get("name")
        ],
        "stage3_state_of_art": str(raw.get("stage3_state_of_art") or ""),
        "stage3_trends": _str_list(raw.get("stage3_trends")),
        "stage3_search_results": str(raw.get("stage3_search_results") or raw.get("stage3_state_of_art") or ""),
        "downstream_report": str(raw.get("downstream_report") or ""),
        "created_at": datetime.now().isoformat(),
        "total_queries": _int(raw.get("total_queries")),
    }


def _normalize_writer_report(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": str(raw.get("plan") or ""),
        "title_candidates": _str_list(raw.get("title_candidates"))[:3],
        "title": str(raw.get("title") or ""),
        "introduction": str(raw.get("introduction") or ""),
        "body": str(raw.get("body") or ""),
        "conclusion": str(raw.get("conclusion") or ""),
        "abstract": str(raw.get("abstract") or ""),
        "keywords": _str_list(raw.get("keywords")),
        "references": str(raw.get("references") or ""),
        "benchmarks": str(raw.get("benchmarks") or ""),
        "trends": str(raw.get("trends") or ""),
    }


def _normalize_papers(raw: Any) -> list[dict[str, Any]]:
    papers = []
    for item in _dict_list(raw):
        title = str(item.get("title") or "")
        if not title:
            continue
        papers.append(
            {
                "paper_id": str(item.get("paper_id") or item.get("url") or item.get("doi") or title),
                "title": title,
                "authors": _str_list(item.get("authors")),
                "year": _int(item.get("year")),
                "venue": str(item.get("venue") or ""),
                "abstract": str(item.get("abstract") or ""),
                "cited_by": _int(item.get("cited_by")),
                "select_score": _float(item.get("select_score")),
            }
        )
    return papers


def _write_step_files(data: dict[str, Any], tmp_dir: Path) -> None:
    step1 = dict(data)
    step1["step"] = "explorer_done"
    step1["searcher_messages"] = []
    step1["searcher_result"] = ""
    step1["searcher_papers"] = []
    step1["writer_report"] = None

    step2 = dict(data)
    step2["step"] = "searcher_done"
    step2["writer_report"] = None

    for name, obj in [
        ("step1_explorer_done.json", step1),
        ("step2_searcher_done.json", step2),
        ("step3_writer_done.json", data),
    ]:
        (tmp_dir / name).write_text(
            json.dumps(obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _write_visualization(data: dict[str, Any], tmp_dir: Path) -> Path:
    from src.adapter import convert

    viz = convert(data)
    out_path = tmp_dir / "visualization_data.json"
    out_path.write_text(viz.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def _extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]

    chunks: list[str] = []
    for item in response.get("output") or []:
        for content in item.get("content") or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("output_text")
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunks)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Responses API output did not contain a JSON object.")
        return json.loads(match.group(0))


def _extract_sources(response: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    for item in response.get("output") or []:
        action = item.get("action") if isinstance(item, dict) else None
        for source in (action or {}).get("sources") or []:
            url = source.get("url") if isinstance(source, dict) else None
            if url and url not in sources:
                sources.append(url)
    return sources


def _slugify(topic: str) -> str:
    slug = re.sub(r"[\s\-]+", "_", topic.strip())
    slug = re.sub(r"[^\w\u4e00-\u9fa5_]", "", slug)
    return slug[:40] or "untitled"


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
