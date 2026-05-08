#!/usr/bin/env python3
"""Explorer Agent 测试脚本"""

import sys, os, logging
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ["PYTHONPATH"] = str(_ROOT) + "/src"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)

from src.llm import LLM
from src.explorer import ExplorerAgent


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Explorer Agent 测试")
    parser.add_argument("topic", nargs="?", default="automatic speech recognition",
                        help="要探索的研究领域")
    args = parser.parse_args()

    llm = LLM(model="Qwen/Qwen3-8B")
    explorer = ExplorerAgent(llm=llm, verbose=True)

    print(f"\n{'='*60}")
    print(f"Explorer Agent 测试 — 领域: {args.topic}")
    print(f"{'='*60}\n")

    report = explorer.run(args.topic)

    print(f"\n{'='*60}")
    print("下游总结报告")
    print(f"{'='*60}\n")
    print(report.to_downstream_summary())

    print(f"\n{'='*60}")
    print("Benchmark URL 列表")
    print(f"{'='*60}\n")
    for url in report.get_all_benchmark_urls():
        print(f"  🔗 {url}")

    print(f"\n✅ 测试完成！共执行 {report.total_queries} 次搜索")


if __name__ == "__main__":
    main()
