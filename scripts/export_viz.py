"""把 tmp/<topic>/step3_writer_done.json 转成 UI 可用的 visualization_data.json。

默认两阶段：脚本规则化 → LLM 校验/补全。LLM 不可用或 --no-llm 时只跑脚本。

用法：
    # 全部转 + LLM
    python scripts/export_viz.py

    # 只跑脚本（不调 LLM）
    python scripts/export_viz.py --no-llm

    # 只转某个具体目录
    python scripts/export_viz.py tmp/ASR自动语音识别

    # 显式指定 IO
    python scripts/export_viz.py --input tmp/ASR/step3_writer_done.json --output tmp/ASR/viz.json
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

from src.adapter import convert, convert_with_llm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def _convert(input_path: Path, output_path: Path, *, use_llm: bool) -> None:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    viz = convert_with_llm(raw) if use_llm else convert(raw)
    output_path.write_text(viz.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"[OK] {input_path.name} -> {output_path}\n"
        f"     topic={viz.topic} | timeline={len(viz.timeline)} methods={len(viz.methods)} "
        f"papers={len(viz.papers)} benchmarks={len(viz.benchmarks)} "
        f"frontiers={len(viz.frontiers)} | nodes={len(viz.graph.nodes)} edges={len(viz.graph.edges)} "
        f"| needs_research={len(viz.needs_research)}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir", nargs="?", help="tmp 下的某个领域目录；不传则处理 tmp/*")
    p.add_argument("--input", help="直接指定输入 step3_writer_done.json 路径")
    p.add_argument("--output", help="直接指定输出 visualization_data.json 路径")
    p.add_argument("--no-llm", action="store_true", help="只跑脚本规则化，不调 LLM")
    args = p.parse_args()

    use_llm = not args.no_llm

    if args.input:
        in_path = Path(args.input)
        out_path = Path(args.output or in_path.parent / "visualization_data.json")
        _convert(in_path, out_path, use_llm=use_llm)
        return

    if args.dir:
        dirs = [Path(args.dir)]
    else:
        tmp_root = _PROJECT_ROOT / "tmp"
        dirs = [d for d in tmp_root.iterdir() if d.is_dir()] if tmp_root.exists() else []

    if not dirs:
        print("没有找到可处理的目录。")
        return

    for d in dirs:
        in_path = d / "step3_writer_done.json"
        if not in_path.exists():
            print(f"[SKIP] {d} (无 step3_writer_done.json)")
            continue
        out_path = d / "visualization_data.json"
        try:
            _convert(in_path, out_path, use_llm=use_llm)
        except Exception as exc:
            print(f"[FAIL] {in_path}: {exc}")


if __name__ == "__main__":
    main()
