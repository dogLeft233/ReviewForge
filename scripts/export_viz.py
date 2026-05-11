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

    output_path.write_text(viz.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"[OK] {input_path.name} -> {output_path}\n"
        f"     topic={viz.topic} | timeline={len(viz.timeline)} methods={len(viz.methods)} "
        f"papers={len(viz.papers)} benchmarks={len(viz.benchmarks)} "
        f"frontiers={len(viz.frontiers)} resources={len(viz.resources)} "
        f"| nodes={len(viz.graph.nodes)} edges={len(viz.graph.edges)} "
        f"| needs_research={len(viz.needs_research)}"
    )


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
        in_path = Path(args.input)
        out_path = Path(args.output or in_path.parent / "visualization_data.json")
        _convert(
            in_path,
            out_path,
            use_llm=use_llm,
            use_openai_enhance=args.openai_enhance,
            enhance_config=args.enhance_config,
        )
        return

    if args.dir:
        dirs = [Path(args.dir)]
    else:
        tmp_root = _PROJECT_ROOT / "tmp"
        dirs = [d for d in tmp_root.iterdir() if d.is_dir()] if tmp_root.exists() else []

    if not dirs:
        print("No processable tmp directories found.")
        return

    for d in dirs:
        in_path = d / "step3_writer_done.json"
        if not in_path.exists():
            print(f"[SKIP] {d} (no step3_writer_done.json)")
            continue
        out_path = d / "visualization_data.json"
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
