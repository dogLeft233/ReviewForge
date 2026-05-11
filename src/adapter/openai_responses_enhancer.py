"""Optional OpenAI Responses API quality critic/enhancer for adapter output.

This module is intentionally conservative: it asks a stronger model to critique
and improve presentation quality, but merges only safe, structured patches.
Fact-heavy fields such as authors, years, URLs, and scores are accepted only
when they arrive as source-backed resources or when explicitly enabled.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from src.adapter.schema import (
    Frontier,
    Method,
    Overview,
    Paper,
    Resource,
    ResearchTask,
    VisualizationData,
)
from src.adapter.script_adapter import rebuild_graph
from src.visualizer.quality import assess_visualization_data

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = Path("adapter_enhance.yaml")
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


@dataclass(slots=True)
class OpenAIEnhanceConfig:
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    timeout_seconds: float = 120.0
    max_output_tokens: int = 5000
    temperature: float = 0.1
    web_search_enabled: bool = True
    web_search_tool_type: str = "web_search_preview"
    allowed_domains: list[str] = field(default_factory=list)
    max_raw_chars: int = 12000
    max_viz_chars: int = 20000
    allow_fact_field_updates: bool = False
    debug_enabled: bool = False
    debug_max_chars: int = 12000


def load_openai_enhance_config(path: str | Path | None = None) -> OpenAIEnhanceConfig:
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        return OpenAIEnhanceConfig()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    openai_cfg = raw.get("openai") or {}
    search_cfg = raw.get("web_search") or {}
    enhance_cfg = raw.get("enhancement") or {}
    debug_cfg = raw.get("debug") or {}

    api_key = (
        str(openai_cfg.get("api_key") or "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )

    return OpenAIEnhanceConfig(
        enabled=bool(raw.get("enabled", False)),
        api_key=api_key,
        base_url=str(openai_cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/"),
        model=str(openai_cfg.get("model") or "gpt-4o"),
        timeout_seconds=float(openai_cfg.get("timeout_seconds") or 120),
        max_output_tokens=int(openai_cfg.get("max_output_tokens") or 5000),
        temperature=float(openai_cfg.get("temperature") or 0.1),
        web_search_enabled=bool(search_cfg.get("enabled", True)),
        web_search_tool_type=str(search_cfg.get("tool_type") or "web_search_preview"),
        allowed_domains=list(search_cfg.get("allowed_domains") or []),
        max_raw_chars=int(enhance_cfg.get("max_raw_chars") or 12000),
        max_viz_chars=int(enhance_cfg.get("max_viz_chars") or 20000),
        allow_fact_field_updates=bool(enhance_cfg.get("allow_fact_field_updates", False)),
        debug_enabled=bool(debug_cfg.get("enabled", False)),
        debug_max_chars=int(debug_cfg.get("max_chars") or 12000),
    )


def enhance_with_openai_responses(
    viz: VisualizationData,
    raw: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    force: bool = False,
) -> VisualizationData:
    cfg = load_openai_enhance_config(config_path)
    if not force and not cfg.enabled:
        logger.info("[openai-enhance] disabled")
        return viz
    if not cfg.api_key:
        logger.warning("[openai-enhance] missing OpenAI API key; skipping")
        return viz

    payload = _build_request_payload(viz, raw, cfg)
    try:
        response = _call_responses_api(payload, cfg)
        patch = _extract_patch(response)
    except Exception as exc:
        logger.warning("[openai-enhance] failed: %s", exc)
        return viz

    enhanced = _merge_patch(viz, patch, cfg)
    enhanced = rebuild_graph(enhanced)
    logger.info("[openai-enhance] quality=%s", assess_visualization_data(enhanced))
    return enhanced


def _build_request_payload(
    viz: VisualizationData,
    raw: dict[str, Any],
    cfg: OpenAIEnhanceConfig,
) -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    if cfg.web_search_enabled:
        tool: dict[str, Any] = {"type": cfg.web_search_tool_type}
        if cfg.allowed_domains and cfg.web_search_tool_type == "web_search":
            tool["filters"] = {"allowed_domains": cfg.allowed_domains}
        tools.append(tool)

    input_text = _build_user_prompt(viz, raw, cfg)
    payload: dict[str, Any] = {
        "model": cfg.model,
        "instructions": _system_prompt(),
        "input": input_text,
        "temperature": cfg.temperature,
        "max_output_tokens": cfg.max_output_tokens,
        "store": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        payload["include"] = ["web_search_call.action.sources"]
    return payload


def _system_prompt() -> str:
    return (
        "You are ReviewForge's research-report quality critic and enhancer. "
        "You may use web search to verify and add source-backed paper/project/resource links. "
        "Do not invent factual fields. If a fact cannot be supported by the provided material or search sources, "
        "leave it unchanged and add a remaining gap. Output only valid JSON."
    )


def _build_user_prompt(
    viz: VisualizationData,
    raw: dict[str, Any],
    cfg: OpenAIEnhanceConfig,
) -> str:
    report = raw.get("explorer_report") or {}
    raw_context = {
        "topic": raw.get("topic"),
        "stage1_overview": report.get("stage1_overview"),
        "stage1_concepts": report.get("stage1_concepts"),
        "stage2_timeline": report.get("stage2_timeline"),
        "stage3_state_of_art": report.get("stage3_state_of_art"),
        "stage3_trends": report.get("stage3_trends"),
        "stage3_benchmarks": report.get("stage3_benchmarks"),
        "stage3_search_results": report.get("stage3_search_results"),
        "downstream_report": report.get("downstream_report"),
    }
    viz_json = _truncate(viz.model_dump_json(indent=2), cfg.max_viz_chars)
    raw_json = _truncate(json.dumps(raw_context, ensure_ascii=False, indent=2), cfg.max_raw_chars)

    return f"""Assess and enhance this visualization report for academic/professional quality.

Return JSON with this exact top-level shape:
{{
  "quality_report": {{
    "overall_score": 0,
    "section_scores": {{"overview": 0, "papers": 0, "methods": 0, "benchmarks": 0, "frontiers": 0, "resources": 0, "graph": 0}},
    "major_issues": [],
    "enhancement_summary": ""
  }},
  "patch": {{
    "overview": {{"definition": "", "core_questions": [], "key_concepts": []}},
    "methods": [{{"id": "", "name": "", "category": "", "description": "", "pros": [], "cons": []}}],
    "frontiers": [{{"name": "", "description": "", "importance": "", "related_methods": [], "related_papers": []}}],
    "resources": [{{"name": "", "type": "paper|github_repo|model_or_space|leaderboard|dataset|resource", "url": "", "description": "", "related_methods": [], "related_papers": []}}],
    "papers": [{{"id": "", "summary": "", "url": ""}}]
  }},
  "remaining_gaps": [{{"target": "", "reason": "", "hint": ""}}]
}}

Rules:
- Improve professionalism, taxonomy, clarity, and evidence linkage.
- Search for authoritative paper/project/resource links if useful.
- Do not change authors, years, venues, benchmark scores, or benchmark years.
- Only provide paper.url/resource.url when supported by source material or web search.
- Preserve existing IDs. For methods, use existing method ids only.
- If no change is needed for a section, return an empty array/object for it.

Current quality metrics:
{json.dumps(assess_visualization_data(viz), ensure_ascii=False, indent=2)}

Current VisualizationData:
```json
{viz_json}
```

Raw report context:
```json
{raw_json}
```
"""


def _call_responses_api(payload: dict[str, Any], cfg: OpenAIEnhanceConfig) -> dict[str, Any]:
    url = f"{cfg.base_url}/responses"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    _debug_print(cfg, "request url", url)
    _debug_print(cfg, "request payload", _json_dumps(payload))
    with httpx.Client(timeout=cfg.timeout_seconds) as client:
        resp = client.post(url, headers=headers, json=payload)
        _debug_print(cfg, "response status", f"{resp.status_code} {resp.reason_phrase}")
        _debug_print(cfg, "response content-type", resp.headers.get("content-type", ""))
        _debug_print(cfg, "response body", resp.text)
        resp.raise_for_status()
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            body = _truncate_debug(resp.text, cfg.debug_max_chars)
            raise ValueError(f"Responses API returned non-JSON body: {body!r}") from exc


def _extract_patch(response: dict[str, Any]) -> dict[str, Any]:
    logger.info("[openai-enhance] response keys=%s", sorted(response.keys()))
    texts: list[str] = []
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"}:
                texts.append(str(content.get("text") or ""))
    if not texts and response.get("output_text"):
        texts.append(str(response["output_text"]))
    joined = "\n".join(texts).strip()
    logger.info("[openai-enhance] extracted text chars=%d", len(joined))
    match = _JSON_BLOCK_RE.search(joined)
    if not match:
        raise ValueError("Responses API did not return a JSON object")
    return json.loads(match.group(0))


def _merge_patch(
    viz: VisualizationData,
    patch: dict[str, Any],
    cfg: OpenAIEnhanceConfig,
) -> VisualizationData:
    patch_obj = patch.get("patch") or {}
    updates: dict[str, Any] = {}

    overview = _merge_overview(viz.overview, patch_obj.get("overview") or {})
    updates["overview"] = overview

    methods = _merge_methods(viz.methods, patch_obj.get("methods") or [])
    updates["methods"] = methods

    frontiers = _merge_frontiers(viz.frontiers, patch_obj.get("frontiers") or [])
    updates["frontiers"] = frontiers

    resources = _merge_resources(viz.resources, patch_obj.get("resources") or [])
    updates["resources"] = resources

    papers = _merge_papers(viz.papers, patch_obj.get("papers") or [], cfg)
    updates["papers"] = papers

    needs = list(viz.needs_research)
    for item in patch.get("remaining_gaps") or []:
        if not isinstance(item, dict) or not item.get("target"):
            continue
        task = ResearchTask(**item)
        if not any(t.target == task.target for t in needs):
            needs.append(task)
    updates["needs_research"] = needs
    updates["quality_report"] = patch.get("quality_report") or {}

    return viz.model_copy(update=updates)


def _merge_overview(current: Overview, patch: dict[str, Any]) -> Overview:
    if not isinstance(patch, dict):
        return current
    return current.model_copy(update={
        "definition": patch.get("definition") or current.definition,
        "core_questions": patch.get("core_questions") or current.core_questions,
        "key_concepts": patch.get("key_concepts") or current.key_concepts,
    })


def _merge_methods(current: list[Method], patches: list[Any]) -> list[Method]:
    by_id = {m.id: m for m in current}
    for item in patches:
        if not isinstance(item, dict) or not item.get("id") or item["id"] not in by_id:
            continue
        old = by_id[item["id"]]
        by_id[item["id"]] = old.model_copy(update={
            "name": item.get("name") or old.name,
            "category": item.get("category") or old.category,
            "description": item.get("description") or old.description,
            "pros": item.get("pros") or old.pros,
            "cons": item.get("cons") or old.cons,
        })
    return [by_id[m.id] for m in current]


def _merge_frontiers(current: list[Frontier], patches: list[Any]) -> list[Frontier]:
    by_name = {f.name.lower(): f for f in current if f.name}
    out = list(current)
    for item in patches:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in by_name:
            old = by_name[key]
            new = old.model_copy(update={
                "description": item.get("description") or old.description,
                "importance": item.get("importance") or old.importance,
                "related_methods": item.get("related_methods") or old.related_methods,
                "related_papers": item.get("related_papers") or old.related_papers,
            })
            out = [new if f.name.lower() == key else f for f in out]
        else:
            out.append(Frontier(**item))
    return out


def _merge_resources(current: list[Resource], patches: list[Any]) -> list[Resource]:
    out = list(current)
    seen = {r.url for r in out if r.url}
    for item in patches:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        data = dict(item)
        data["id"] = data.get("id") or _resource_id(data["url"])
        out.append(Resource(**data))
    return out


def _merge_papers(
    current: list[Paper],
    patches: list[Any],
    cfg: OpenAIEnhanceConfig,
) -> list[Paper]:
    by_id = {p.id: p for p in current}
    for item in patches:
        if not isinstance(item, dict) or not item.get("id") or item["id"] not in by_id:
            continue
        old = by_id[item["id"]]
        update: dict[str, Any] = {"summary": item.get("summary") or old.summary}
        if cfg.allow_fact_field_updates and item.get("url"):
            update["url"] = item["url"]
        elif item.get("url") and not old.url:
            update["url"] = item["url"]
        by_id[item["id"]] = old.model_copy(update=update)
    return [by_id[p.id] for p in current]


def _resource_id(url: str) -> str:
    return "r_" + re.sub(r"\W+", "_", url.lower()).strip("_")[:60]


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...truncated"


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _truncate_debug(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[debug truncated]"


def _debug_print(cfg: OpenAIEnhanceConfig, label: str, value: str) -> None:
    if not cfg.debug_enabled:
        return
    print(f"\n[openai-enhance debug] {label}:")
    print(_truncate_debug(value, cfg.debug_max_chars))
