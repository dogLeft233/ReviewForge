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
    Benchmark,
    Frontier,
    Method,
    Overview,
    Paper,
    Resource,
    ResearchTask,
    TimelineEvent,
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

    enhanced = viz
    module_reports: dict[str, Any] = {}
    for module in _enhancement_modules(enhanced):
        payload = _build_module_request_payload(module, enhanced, raw, cfg)
        try:
            logger.info("[openai-enhance] module=%s request", module["name"])
            response = _call_responses_api(payload, cfg)
            patch = _extract_patch(response)
        except Exception as exc:
            logger.warning("[openai-enhance] module=%s failed: %s", module["name"], exc)
            continue
        patch = _filter_patch_for_module(patch, module["name"])
        enhanced = _merge_patch(enhanced, patch, cfg)
        module_reports[module["name"]] = patch.get("quality_report") or {}

    enhanced = rebuild_graph(enhanced)
    scored_reports = [
        report.get("overall_score")
        for report in module_reports.values()
        if isinstance(report, dict) and report.get("overall_score")
    ]
    overall_score = round(sum(scored_reports) / len(scored_reports), 2) if scored_reports else 0
    enhanced = enhanced.model_copy(update={
        "quality_report": {
            "overall_score": overall_score,
            "modules": module_reports,
            "final_metrics": assess_visualization_data(enhanced),
        }
    })
    logger.info("[openai-enhance] quality=%s", assess_visualization_data(enhanced))
    return enhanced


def _enhancement_modules(viz: VisualizationData) -> list[dict[str, Any]]:
    """Run focused Responses API calls instead of one monolithic request."""

    return [
        {
            "name": "paper_timeline_links",
            "title": "Paper and timeline URL enrichment",
            "web_search": True,
            "goal": (
                "Find authoritative URLs for existing papers and make timeline events point to "
                "the best matching existing paper ids. Do not add new timeline events unless the "
                "current event cannot be linked and a stage-2 source-backed replacement is needed."
            ),
        },
        {
            "name": "method_details",
            "title": "Concise method pros and cons",
            "web_search": False,
            "goal": (
                "Rewrite each existing method's pros and cons as concise LLM-summarized bullets "
                "for the Method Details table. Use only the current method description, linked "
                "papers, and raw report evidence."
            ),
        },
        {
            "name": "method_links",
            "title": "Method representative-paper linkage",
            "web_search": True,
            "goal": (
                "For each existing method, verify or repair representative paper ids using only "
                "existing paper ids. Add resources only when they directly support a method."
            ),
        },
        {
            "name": "frontier_sources",
            "title": "Frontier Chinese descriptions and recent-paper supplementation",
            "web_search": True,
            "goal": (
                "Supplement Frontier Trends with more source-backed 2024-2026 frontier papers "
                "and open-source projects. Add URL-bearing resources for every returned frontier, "
                "and write all frontier descriptions in Chinese."
            ),
        },
        {
            "name": "frontier_descriptions",
            "title": "Concise frontier card descriptions",
            "web_search": False,
            "goal": (
                "Rewrite every existing frontier card description as a short, readable Chinese LLM summary. "
                "Use current frontier names, resources, linked papers, and raw stage-3 evidence."
            ),
        },
    ]


def _filter_patch_for_module(patch: dict[str, Any], module_name: str) -> dict[str, Any]:
    allowed: dict[str, set[str]] = {
        "paper_timeline_links": {"papers", "timeline", "resources"},
        "method_details": {"methods"},
        "method_links": {"papers", "methods", "resources"},
        "frontier_sources": {"frontiers", "resources"},
        "frontier_descriptions": {"frontiers"},
    }
    allowed_keys = allowed.get(module_name, set())
    patch_obj = dict(patch.get("patch") or {})
    filtered_patch = {
        key: value
        for key, value in patch_obj.items()
        if key in allowed_keys
    }
    return {
        "quality_report": patch.get("quality_report") or {},
        "patch": filtered_patch,
        "remaining_gaps": patch.get("remaining_gaps") or [],
    }


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


def _build_module_request_payload(
    module: dict[str, Any],
    viz: VisualizationData,
    raw: dict[str, Any],
    cfg: OpenAIEnhanceConfig,
) -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    if cfg.web_search_enabled and module.get("web_search", True):
        tool: dict[str, Any] = {"type": cfg.web_search_tool_type}
        if cfg.allowed_domains and cfg.web_search_tool_type == "web_search":
            tool["filters"] = {"allowed_domains": cfg.allowed_domains}
        tools.append(tool)

    payload: dict[str, Any] = {
        "model": cfg.model,
        "instructions": _module_system_prompt(module),
        "input": _build_module_user_prompt(module, viz, raw, cfg),
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
        "You are ReviewForge's visualization-data reviewer and evidence-backed information enhancer. "
        "Use web search to verify papers, benchmarks, open-source projects, and authoritative URLs whenever "
        "the current data lacks evidence. Remove weakly related, over-general, or non-domain-core items. "
        "Prefer entries backed by representative papers, official benchmark pages, GitHub repositories, "
        "Hugging Face project pages, arXiv/DOI pages, or conference/project pages. "
        "Do not invent factual fields. If a fact cannot be supported by provided material or search sources, "
        "leave it unchanged, remove the weak item, or add a remaining gap. Output only one valid JSON object, "
        "with no Markdown and no explanatory prose. Do not return an empty patch when the input has obvious "
        "missing explanations, over-general timeline items, methods without papers, or benchmarks without descriptions."
    )


def _module_system_prompt(module: dict[str, Any]) -> str:
    search_sentence = (
        "Use web search when URLs or current frontier sources are missing. "
        if module.get("web_search", True)
        else "Do not use web search; summarize only from the supplied structured data and raw evidence. "
    )
    return (
        "You are ReviewForge's evidence-backed visualization-data enhancer. "
        f"Current module: {module['title']}. "
        f"{search_sentence}"
        "Return only one valid JSON object, with no Markdown and no prose. "
        "Do not invent facts. Every URL must be authoritative: arXiv/DOI/conference/project page, "
        "official benchmark page, GitHub repository, Hugging Face page, or official dataset page. "
        "Preserve existing ids. Do not change authors, years, venues, benchmark scores, or benchmark years."
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

Current date: 2026-05-11. Treat "recent two years" as 2024-2026.

Return JSON with this exact top-level shape:
{{
  "quality_report": {{
    "overall_score": 0,
    "section_scores": {{"overview": 0, "papers": 0, "methods": 0, "benchmarks": 0, "frontiers": 0, "resources": 0, "graph": 0}},
    "major_issues": [],
    "enhancement_summary": ""
  }},
  "patch": {{
    "overview": {{"key_concept_explanations": {{}}}},
    "timeline": [{{"year": 0, "title": "", "category": "", "description": "", "related_papers": []}}],
    "methods": [{{"id": "", "name": "", "category": "", "description": "", "pros": [], "cons": [], "papers": []}}],
    "remove_method_ids": [],
    "frontiers": [{{"name": "", "description": "", "importance": "", "related_methods": [], "related_papers": []}}],
    "benchmarks": [{{"model": "", "dataset": "", "metric": "", "score": 0, "year": 0, "url": "", "description": ""}}],
    "resources": [{{"name": "", "type": "paper|github_repo|model_or_space|leaderboard|dataset|resource", "url": "", "description": "", "related_methods": [], "related_papers": []}}],
    "papers": [{{"id": "", "summary": "", "url": ""}}]
  }},
  "remaining_gaps": [{{"target": "", "reason": "", "hint": ""}}]
}}

Page-specific requirements:
- Overview: Fill key_concept_explanations for every existing key concept. Each explanation must be 1-2 concise academic sentences.
- Timeline: Return representative domain milestones, prioritizing quality over count. Each item must be a domain-specific paper, method, dataset, benchmark, or system. Do not include broad foundation items such as "Attention Is All You Need" unless the description states a direct contribution to this exact domain. If the current timeline contains broad NLP papers, return a cleaned replacement timeline patch.
- Method Map: Every returned method must have description, pros, cons, and at least one representative paper id. If an existing method cannot be supported by representative evidence, put its id in remove_method_ids. You may use existing paper ids from Current VisualizationData.
- Frontier: Return only open-source projects and 2024-2026 frontier papers. Exclude generic trends, product news, and unverified claims. Link them through related_papers or resources.
- Benchmark: Return only common benchmark/dataset entries with description, metric, and URL. Do not return SOTA score-trend rows; keep score/year at 0 unless already source-backed in current data.

Global rules:
- Improve professionalism, taxonomy, clarity, and evidence linkage.
- Search for authoritative paper/project/resource links when filling gaps.
- Do not change authors, years, venues, benchmark scores, or benchmark years unless the existing value is empty and the source is authoritative.
- Only provide paper.url/resource.url/benchmark.url when supported by source material or web search.
- Preserve existing paper and method IDs. For methods, use existing method ids only.
- If no change is needed for a section, return an empty array/object for it. However, the current input is not compliant when concept explanations are empty, timeline has fewer than 8 milestones, methods have no papers, or benchmark descriptions/URLs are missing.
- Each new URL-bearing item must include a meaningful description that explains why the source supports the item.
- Do not spend the entire response on one missing URL. Use the supplied raw report as evidence for conceptual explanations and method-paper linkage, and use web search mainly to add authoritative URLs.
- The quality_report scores must be 1-10, not all zeros. Use 0 only if a section is completely unusable.
- At minimum, when key_concept_explanations is empty, patch.overview.key_concept_explanations must contain every current key concept.

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


def _build_module_user_prompt(
    module: dict[str, str],
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
    compact_viz = _compact_viz_for_module(viz, module["name"])
    module_rules = _module_rules(module["name"])
    return f"""Enhance one ReviewForge visualization module.

Current date: 2026-05-12. Treat "recent two years" as 2024-2026.
Topic: {viz.topic}
Module goal: {module["goal"]}

Return JSON with this exact top-level shape:
{{
  "quality_report": {{
    "overall_score": 0,
    "major_issues": [],
    "enhancement_summary": ""
  }},
  "patch": {{
    "papers": [{{"id": "", "summary": "", "url": ""}}],
    "timeline": [{{"year": 0, "title": "", "category": "", "description": "", "related_papers": []}}],
    "methods": [{{"id": "", "name": "", "category": "", "description": "", "pros": [], "cons": [], "papers": []}}],
    "frontiers": [{{"name": "", "description": "", "importance": "", "related_methods": [], "related_papers": []}}],
    "resources": [{{"name": "", "type": "paper|github_repo|model_or_space|leaderboard|dataset|resource", "url": "", "description": "", "related_methods": [], "related_papers": []}}]
  }},
  "remaining_gaps": [{{"target": "", "reason": "", "hint": ""}}]
}}

Module-specific rules:
{module_rules}

Global constraints:
- Keep the patch focused on this module only. Empty arrays are allowed for unrelated sections.
- Use existing paper ids and method ids only. Do not create new paper ids.
- For related_papers, return ids as strings, not paper titles.
- If you add a paper URL, add it through patch.papers using the existing paper id.
- If you add a URL-bearing source that should appear on the Links tab, add it to patch.resources.
- Every resource must have a meaningful description and, when applicable, related_papers or related_methods.
- For Frontier modules, all frontier and paper-resource descriptions must be Chinese.
- quality_report scores must be 1-10 unless the module is completely unusable.

Current module data:
```json
{json.dumps(compact_viz, ensure_ascii=False, indent=2)}
```

Raw stage evidence:
```json
{_truncate(json.dumps(raw_context, ensure_ascii=False, indent=2), cfg.max_raw_chars)}
```
"""


def _module_rules(module_name: str) -> str:
    if module_name == "paper_timeline_links":
        return (
            "- Fill missing URLs for existing papers with authoritative paper pages.\n"
            "- Make every timeline event include related_papers when an existing paper clearly matches.\n"
            "- Preserve stage2_timeline wording and phase categories; do not rewrite the whole timeline.\n"
            "- Add paper resources for found URLs so the Links tab can list literature links again."
        )
    if module_name == "method_details":
        return (
            "- Return one patch.methods entry for every existing method id.\n"
            "- Keep method id/name/category/papers unchanged unless already present in current data.\n"
            "- Rewrite pros as 1-2 short bullet strings, each no more than 18 Chinese characters or 12 English words.\n"
            "- Rewrite cons as 1-2 short bullet strings, each no more than 18 Chinese characters or 12 English words.\n"
            "- Pros and cons must be specific, not generic placeholders.\n"
            "- Do not add resources, papers, timeline, or frontiers in this module."
        )
    if module_name == "method_links":
        return (
            "- Each method should keep or receive at least one representative existing paper id.\n"
            "- Prefer direct conceptual matches: CTC, DeepSpeech, acoustic modeling, Wav2Vec, Conformer, self-supervised learning.\n"
            "- Add resources only for method-supporting papers/projects that have a clear URL.\n"
            "- Do not remove methods in this module; repair linkage instead."
        )
    if module_name == "frontier_sources":
        return (
            "- Return only open-source projects and 2024-2026 frontier papers.\n"
            "- Search broadly and return at least 6 recent frontier paper resources when available.\n"
            "- Prefer a diverse set: efficient/streaming ASR, multilingual ASR, self-supervised speech models, robust/noisy ASR, long-form ASR, and open ASR evaluation.\n"
            "- Each frontier must be backed by at least one returned resource URL.\n"
            "- Each returned paper resource must use type=\"paper\", include an arXiv/DOI/conference URL, and explain the contribution in Chinese.\n"
            "- For open-source projects, prefer GitHub or Hugging Face pages.\n"
            "- For recent papers, prefer arXiv/DOI/conference pages and mention the year in description or importance.\n"
            "- All patch.frontiers[].description values must be Chinese, one concise sentence, and must not contain Markdown links.\n"
            "- Exclude generic advice, benchmarks, product news, and source-free trend labels."
        )
    if module_name == "frontier_descriptions":
        return (
            "- Return one patch.frontiers entry for every existing frontier name.\n"
            "- Keep frontier name, related_methods, and related_papers unchanged unless already present in current data.\n"
            "- Rewrite description in Chinese as one concise sentence, no more than 42 Chinese characters.\n"
            "- Description must explain the technical point or why it matters; do not output English-only text, URLs, or Markdown links.\n"
            "- Do not add resources, papers, timeline, or methods in this module."
        )
    return "- Improve only the stated module."


def _compact_viz_for_module(viz: VisualizationData, module_name: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "topic": viz.topic,
        "papers": [p.model_dump() for p in viz.papers],
        "methods": [m.model_dump() for m in viz.methods],
        "resources": [r.model_dump() for r in viz.resources],
    }
    if module_name == "paper_timeline_links":
        base["timeline"] = [e.model_dump() for e in viz.timeline]
    elif module_name in {"method_details", "method_links"}:
        base["timeline"] = [e.model_dump() for e in viz.timeline]
    elif module_name in {"frontier_sources", "frontier_descriptions"}:
        base["frontiers"] = [f.model_dump() for f in viz.frontiers]
    return base


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

    papers = _merge_papers(viz.papers, patch_obj.get("papers") or [], cfg)
    updates["papers"] = papers

    methods = _merge_methods(viz.methods, patch_obj.get("methods") or [])
    methods = _remove_methods(methods, patch_obj.get("remove_method_ids") or [])
    updates["methods"] = methods

    paper_ids = {paper.id for paper in papers}
    method_ids = {method.id for method in methods}
    resources = _merge_resources(
        viz.resources,
        patch_obj.get("resources") or [],
        valid_method_ids=method_ids,
        valid_paper_ids=paper_ids,
    )
    updates["resources"] = resources

    benchmarks = _merge_benchmarks(viz.benchmarks, patch_obj.get("benchmarks") or [])
    updates["benchmarks"] = benchmarks

    frontiers = _merge_frontiers(viz.frontiers, patch_obj.get("frontiers") or [], papers, methods, resources)
    updates["frontiers"] = frontiers

    timeline = _merge_timeline(viz.timeline, patch_obj.get("timeline") or [], papers)
    updates["timeline"] = timeline

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
    explanations = dict(current.key_concept_explanations)
    for key, value in (patch.get("key_concept_explanations") or {}).items():
        concept = str(key or "").strip()
        explanation = str(value or "").strip()
        if concept and explanation:
            explanations[concept] = explanation
    return current.model_copy(update={
        "definition": patch.get("definition") or current.definition,
        "core_questions": patch.get("core_questions") or current.core_questions,
        "key_concepts": patch.get("key_concepts") or current.key_concepts,
        "key_concept_explanations": explanations,
    })


def _merge_methods(current: list[Method], patches: list[Any]) -> list[Method]:
    by_id = {m.id: m for m in current}
    for item in patches:
        if not isinstance(item, dict) or not item.get("id") or item["id"] not in by_id:
            continue
        old = by_id[item["id"]]
        paper_ids, _ = _normalize_related_papers(item.get("papers") or [], set())
        by_id[item["id"]] = old.model_copy(update={
            "name": item.get("name") or old.name,
            "category": item.get("category") or old.category,
            "description": item.get("description") or old.description,
            "pros": _concise_items(item.get("pros") or old.pros, max_items=2, max_chars=36),
            "cons": _concise_items(item.get("cons") or old.cons, max_items=2, max_chars=36),
            "papers": paper_ids or old.papers,
        })
    return [by_id[m.id] for m in current]


def _concise_items(values: Any, *, max_items: int, max_chars: int) -> list[str]:
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for value in values or []:
        text = re.sub(r"\s+", " ", str(value or "")).strip(" -*\t")
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip(" ，,。.;；") + "..."
        if text not in out:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _concise_text(value: Any, *, max_chars: int) -> str:
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", "", str(value or ""))
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -:：\t")
    text = re.sub(r"^论文\s*", "", text).strip(" -:：\t")
    if len(text) > max_chars:
        text = text[:max_chars].rstrip(" ，,。.;；") + "..."
    return text


def _remove_methods(current: list[Method], remove_ids: list[Any]) -> list[Method]:
    doomed = {str(mid).strip() for mid in remove_ids if str(mid or "").strip()}
    if not doomed:
        return current
    return [method for method in current if method.id not in doomed]


def _merge_frontiers(
    current: list[Frontier],
    patches: list[Any],
    papers: list[Paper],
    methods: list[Method],
    resources: list[Resource],
) -> list[Frontier]:
    by_name = {f.name.lower(): f for f in current if f.name}
    out: list[Frontier] = []
    paper_ids = {p.id for p in papers}
    method_ids = {m.id for m in methods}
    resource_text = " ".join(f"{r.name} {r.url} {r.description}" for r in resources).lower()
    resource_method_ids = {mid for r in resources for mid in r.related_methods}
    resource_paper_ids = {pid for r in resources for pid in r.related_papers}
    for item in patches:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        related_papers, _ = _normalize_related_papers(item.get("related_papers") or [], paper_ids)
        related_methods = [
            str(mid).strip()
            for mid in item.get("related_methods") or []
            if str(mid or "").strip() in method_ids
        ]
        has_resource_evidence = (
            name.lower() in resource_text
            or bool(set(related_papers) & resource_paper_ids)
            or bool(set(related_methods) & resource_method_ids)
            or bool(related_papers)
        )
        if not has_resource_evidence:
            continue
        key = name.lower()
        if key in by_name:
            old = by_name[key]
            new = old.model_copy(update={
                "description": _concise_text(item.get("description") or old.description, max_chars=80),
                "importance": item.get("importance") or old.importance,
                "related_methods": related_methods or old.related_methods,
                "related_papers": related_papers or old.related_papers,
            })
            out.append(new)
        else:
            data = dict(item)
            data["description"] = _concise_text(data.get("description"), max_chars=80)
            data["related_papers"] = related_papers
            data["related_methods"] = related_methods
            out.append(Frontier(**data))
    return out or current


def _merge_benchmarks(current: list[Benchmark], patches: list[Any]) -> list[Benchmark]:
    out: list[Benchmark] = []
    seen: set[tuple[str, str, str]] = set()
    for item in patches:
        if not isinstance(item, dict):
            continue
        data = dict(item)
        if data.get("name") and not data.get("model"):
            data["model"] = data["name"]
        if not (data.get("dataset") or data.get("model")) or not data.get("url"):
            continue
        # Benchmark tab is a catalog, not a SOTA leaderboard. Keep the
        # catalog fields and discard model-score trend fields from LLM patches.
        data["score"] = 0.0
        data["year"] = 0
        try:
            benchmark = Benchmark(**data)
        except Exception:
            continue
        key = (benchmark.model.lower(), benchmark.dataset.lower(), benchmark.metric.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(benchmark)
    return out or current


def _merge_timeline(
    current: list[TimelineEvent],
    patches: list[Any],
    papers: list[Paper],
) -> list[TimelineEvent]:
    paper_ids = {p.id for p in papers}
    by_key = {(event.year, event.title.lower()): event for event in current}
    for item in patches:
        if not isinstance(item, dict):
            continue
        data = dict(item)
        title = str(data.get("title") or "").strip()
        year = int(data.get("year") or 0)
        if not title or not year:
            continue
        if _is_overgeneral_timeline_title(title, str(data.get("description") or "")):
            continue
        key = (year, title.lower())
        if key not in by_key:
            continue
        old = by_key[key]
        related_papers, related_urls = _normalize_related_papers(data.get("related_papers") or [], paper_ids)
        _ = related_urls
        by_key[key] = old.model_copy(update={
            "category": data.get("category") or old.category,
            "description": data.get("description") or old.description,
            "related_papers": related_papers or old.related_papers,
        })
    return [by_key[(event.year, event.title.lower())] for event in current]


def _normalize_related_papers(values: list[Any], valid_ids: set[str]) -> tuple[list[str], dict[str, str]]:
    """Accept paper refs as ids or {"id": ..., "url": ...} objects."""

    ids: list[str] = []
    urls: dict[str, str] = {}
    for value in values:
        if isinstance(value, str):
            pid = value
            url = ""
        elif isinstance(value, dict):
            pid = str(value.get("id") or "").strip()
            url = str(value.get("url") or "").strip()
        else:
            continue
        if valid_ids and pid not in valid_ids:
            continue
        if pid and pid not in ids:
            ids.append(pid)
        if pid and url:
            urls[pid] = url
    return ids, urls


def _is_overgeneral_timeline_title(title: str, description: str) -> bool:
    broad_titles = {
        "attention is all you need",
    }
    title_key = title.strip().lower()
    if title_key not in broad_titles:
        return False
    domain_terms = ("asr", "automatic speech recognition", "speech recognition")
    return not any(term in description.lower() for term in domain_terms)


def _merge_resources(
    current: list[Resource],
    patches: list[Any],
    *,
    valid_method_ids: set[str] | None = None,
    valid_paper_ids: set[str] | None = None,
) -> list[Resource]:
    out = list(current)
    seen = {r.url for r in out if r.url}
    valid_method_ids = valid_method_ids or set()
    valid_paper_ids = valid_paper_ids or set()
    for item in patches:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        data = dict(item)
        data["id"] = data.get("id") or _resource_id(data["url"])
        data["related_papers"], _ = _normalize_related_papers(data.get("related_papers") or [], valid_paper_ids)
        data["related_methods"] = [
            str(mid).strip()
            for mid in data.get("related_methods") or []
            if str(mid or "").strip() and (not valid_method_ids or str(mid).strip() in valid_method_ids)
        ]
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
