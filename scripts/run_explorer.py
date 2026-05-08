#!/usr/bin/env python3
"""交互式 Explorer Agent — 输入领域即可探索"""

import logging
import sys
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ["PYTHONPATH"] = str(_ROOT) + "/src"
os.environ["BOCHA_API_KEY"] = "sk-grnzvqmqpizcjwszwfuyirfhwocbopgwhcibkitmrpsoauye"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)

from src.llm import LLM
from src.explorer import ExplorerAgent


def main():
    print("=" * 60)
    print("🔍 Explorer Agent — 领域探索")
    print("=" * 60)

    topic = input("\n📚 请输入要探索的研究领域（或回车使用默认示例）：\n> ").strip()

    if not topic:
        topic = "automatic speech recognition"

    print(f"\n⏳ 正在探索领域：{topic}，请稍候...\n")

    api_key = os.environ.get("BOCHA_API_KEY", "sk-grnzvqmqpizcjwszwfuyirfhwocbopgwhcibkitmrpsoauye")
    llm = LLM(api_key=api_key, model="Qwen/Qwen3-8B", timeout_seconds=300.0, max_tokens=4096)
    explorer = ExplorerAgent(llm=llm, verbose=True)

    try:
        report = explorer.run(topic)
    except Exception as e:
        print(f"❌ 探索失败：{e}")
        raise

    print(f"\n{'='*60}")
    print("📄 下游总结报告")
    print(f"{'='*60}\n")
    print(report.to_downstream_summary())

    urls = report.get_all_benchmark_urls()
    if urls:
        print(f"\n{'='*60}")
        print("🔗 Benchmark / Leaderboard URL")
        print(f"{'='*60}\n")
        for url in urls:
            print(f"  🔗 {url}")

    print(f"\n✅ 完成！共执行 {report.total_queries} 次搜索")


if __name__ == "__main__":
    main()
