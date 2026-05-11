"""Convert tmp/<topic>/step3_writer_done.json to visualization_data.json.

Default path:
    script adapter -> local LLM refiner

Optional path:
    script adapter -> local LLM refiner -> OpenAI Responses API enhancement

Examples:
    python scripts/export_viz.py --no-llm
    python scripts/export_viz.py tmp/ASR自动语音识别 --no-llm
    python scripts/export_viz.py tmp/ASR自动语音识别 --openai-enhance
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.adapter import convert, convert_with_llm, convert_with_openai_enhancement  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def _convert(
    input_path: Path,
    output_path: Path,
    *,
    use_llm: bool,
    use_openai_enhance: bool,
    enhance_config: str | None,
) -> None:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if use_openai_enhance:
        if use_llm:
            viz = convert_with_openai_enhancement(
                raw,
                config_path=enhance_config,
                force=True,
            )
        else:
            from src.adapter.openai_responses_enhancer import enhance_with_openai_responses

            viz = enhance_with_openai_responses(
                convert(raw),
                raw,
                config_path=enhance_config,
                force=True,
            )
    else:
        viz = convert_with_llm(raw) if use_llm else convert(raw)

    if use_openai_enhance and not _openai_enhancement_effective(viz):
        raise RuntimeError(
            "OpenAI enhancement returned no useful patch. "
            "No visualization_data.json was written; rerun without --openai-enhance "
            "for raw export, or check adapter_enhance.yaml/API response."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(viz.model_dump_json(indent=2), encoding="utf-8")
    quality = viz.quality_report or {}
    enhancement_note = ""
    if use_openai_enhance:
        score = quality.get("overall_score")
        if quality:
            enhancement_note = f" | openai_quality={score}"
        else:
            enhancement_note = " | openai_quality=unavailable"
    print(
        f"[OK] {input_path.name} -> {output_path}\n"
        f"     topic={viz.topic} | timeline={len(viz.timeline)} methods={len(viz.methods)} "
        f"papers={len(viz.papers)} benchmarks={len(viz.benchmarks)} "
        f"frontiers={len(viz.frontiers)} resources={len(viz.resources)} "
        f"| nodes={len(viz.graph.nodes)} edges={len(viz.graph.edges)} "
        f"| needs_research={len(viz.needs_research)}{enhancement_note}"
    )


def _openai_enhancement_effective(viz) -> bool:
    quality = viz.quality_report or {}
    has_concept_explanations = bool(viz.overview.key_concept_explanations)
    has_no_broad_timeline = not any("Attention Is All You Need" == e.title for e in viz.timeline)
    has_benchmark_catalog = bool(viz.benchmarks) and all(
        b.url and b.description and not b.score and not b.year for b in viz.benchmarks
    )
    return (
        bool(quality.get("overall_score", 0))
        and has_concept_explanations
        and has_no_broad_timeline
        and has_benchmark_catalog
    )


def _resolve_input_path(value: str) -> Path:
    path = Path(value)
    candidates = [
        path,
        _PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _resolve_topic_dir(value: str) -> Path:
    """Accept either tmp/<topic>, an absolute dir, or just <topic>."""

    path = Path(value)
    candidates = [
        path,
        _PROJECT_ROOT / path,
        _PROJECT_ROOT / "tmp" / value,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # If the user passed a bare topic name, prefer the documented tmp/<topic>
    # location even before it exists so the skip message points at the right path.
    if len(path.parts) == 1:
        return _PROJECT_ROOT / "tmp" / value
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", nargs="?", help="Topic directory under tmp/. If omitted, process tmp/*.")
    parser.add_argument("--input", help="Explicit input step3_writer_done.json path.")
    parser.add_argument("--output", help="Explicit output visualization_data.json path.")
    parser.add_argument("--no-llm", action="store_true", help="Run script adapter only.")
    parser.add_argument("--openai-enhance", action="store_true", help="Enable OpenAI Responses API quality enhancement.")
    parser.add_argument("--enhance-config", default="adapter_enhance.yaml", help="OpenAI enhancement YAML config path.")
    args = parser.parse_args()

    use_llm = not args.no_llm

    if args.input:
        in_path = _resolve_input_path(args.input)
        out_path = Path(args.output) if args.output else in_path.parent / "visualization_data.json"
        _convert(
            in_path,
            out_path,
            use_llm=use_llm,
            use_openai_enhance=args.openai_enhance,
            enhance_config=args.enhance_config,
        )
        return

    if args.dir:
        dirs = [_resolve_topic_dir(args.dir)]
    else:
        tmp_root = _PROJECT_ROOT / "tmp"
        dirs = [d for d in tmp_root.iterdir() if d.is_dir()] if tmp_root.exists() else []

    if not dirs:
        print("No processable tmp directories found.")
        return
    if args.output and len(dirs) != 1:
        raise SystemExit("--output can only be used with --input or a single topic directory.")

    for d in dirs:
        in_path = d / "step3_writer_done.json"
        if not in_path.exists():
            print(f"[SKIP] {d} (no step3_writer_done.json)")
            continue
        out_path = Path(args.output) if args.output else d / "visualization_data.json"
        try:
            _convert(
                in_path,
                out_path,
                use_llm=use_llm,
                use_openai_enhance=args.openai_enhance,
                enhance_config=args.enhance_config,
            )
        except Exception as exc:
            print(f"[FAIL] {in_path}: {exc}")


if __name__ == "__main__":
    main()
