#!/usr/bin/env python3
"""交互式综述流水线 — Explorer → Searcher → Writer

用法:
    # 完整流程（交互输入领域）
    python run_pipeline.py

    # 指定领域（无需交互）
    python run_pipeline.py --topic "LoRA fine-tuning"

    # 从 tmp 目录恢复（跳过已完成步骤）
    python run_pipeline.py --topic "LoRA" --resume

    # 列出所有已保存的 session
    python run_pipeline.py --list

    # 从指定 step 文件恢复
    python run_pipeline.py --topic "LoRA" --resume-from tmp/LoRA/step2_searcher_done.json

    # 指定起始步骤（调试用）
    python run_pipeline.py --topic "LoRA" --from-step searcher_done

    # 指定 log 级别
    python run_pipeline.py --topic "LoRA" --log-level DEBUG
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ── 项目路径设置（必须在 import src 之前）───────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ["PYTHONPATH"] = str(_ROOT / "src")

# ── 延迟导入（路径设置好之后再 import）────────────────────────────────────────

from src.core import CoreConfig, PipelineResult, ReviewForge

# ── API Keys（开发调试用）───────────────────────────────────────────────────

# 从环境变量读取，可传入 BOCHA_API_KEY / LLM_API_KEY 覆盖
_API_KEY = os.environ.get("BOCHA_API_KEY", "sk-grnzvqmqpizcjwszwfuyirfhwocbopgwhcibkitmrpsoauye")

# ── 入口 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReviewForge 综述流水线（开发调试用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--topic", "-t", type=str, default=None, help="要写综述的研究领域")
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认 INFO）",
    )
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="从 tmp/{topic}/ 最新中间文件恢复（需配合 --topic）",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        metavar="PATH",
        help="从指定中间结果文件恢复，如 tmp/LoRA/step2_searcher_done.json",
    )
    parser.add_argument(
        "--from-step",
        type=str,
        default=None,
        choices=["explorer_done", "searcher_done"],
        help="强制从指定步骤开始（调试用，会覆盖 --resume-from）",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="列出所有已保存的 session",
    )
    parser.add_argument(
        "--max-turns", type=int, default=2, help="MultiSourceSearcher 最大轮数（默认 2）"
    )
    parser.add_argument(
        "--max-results", type=int, default=50, help="每个数据源最大结果数（默认 50）"
    )
    parser.add_argument(
        "--tmp-root", type=str, default=None, help="中间结果根目录（默认 src/../tmp）"
    )
    args = parser.parse_args()

    # ── 日志配置 ────────────────────────────────────────────────────────────

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("run_pipeline")

    # ── 导入（延迟） ──────────────────────────────────────────────────────

    from src.config import settings
    from src.llm import LLM

    # ── LLM 初始化 ─────────────────────────────────────────────────────────

    llm = LLM(
        api_key=_API_KEY or settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        max_concurrency=settings.llm_max_concurrency,
        timeout_seconds=300.0,
        max_tokens=4096,
    )

    # ── Core 配置 ──────────────────────────────────────────────────────────

    config = CoreConfig(
        search_max_turns=args.max_turns,
        search_max_results=args.max_results,
        verbose=(args.log_level == "DEBUG"),
    )
    tmp_root = Path(args.tmp_root) if args.tmp_root else None
    rf = ReviewForge(llm=llm, config=config, tmp_root=tmp_root)

    # ── 列表模式 ──────────────────────────────────────────────────────────

    if args.list_sessions:
        sessions = rf.list_sessions()
        if not sessions:
            print("暂无已保存的 session")
            return
        print(f"{'Topic':<40} {'Step':<20} {'Path'}")
        print("-" * 80)
        for s in sessions:
            print(f"{s['topic']:<40} {s['current_step']:<20} {s['path']}")
        return

    # ── 恢复模式优先级 ───────────────────────────────────────────────────

    if args.resume_from:
        path = Path(args.resume_from)
        if not path.exists():
            print(f"❌ 文件不存在: {path}")
            return
        print(f"📂 从指定文件恢复: {path}")
        result = ReviewForge.resume_from(path)
        if result.step == "complete":
            print("✅ 流程已完成，直接返回结果")
            _print_result(result)
            return
        # 继续执行剩余步骤
        topic = result.topic
        rf_tmp_root = result.tmp_dir
    elif args.resume:
        if not args.topic:
            print("❌ --resume 需要配合 --topic 使用")
            return
        slug = _slugify(args.topic)
        tmp_root = Path(rf.tmp_root)
        tmp_dir = tmp_root / slug
        step_files = sorted(tmp_dir.glob("step*.json")) if tmp_dir.exists() else []
        if not step_files:
            print(f"❌ tmp/{slug}/ 下无中间文件，无法恢复。从头开始...")
            # fall through to normal run
            args.resume = False
        else:
            path = step_files[-1]
            print(f"📂 从最新中间文件恢复: {path}")
            result = ReviewForge.resume_from(path)
            if result.step == "complete":
                print("✅ 该 topic 已完成，直接返回结果")
                _print_result(result)
                return
            rf = ReviewForge(llm=llm, config=config, tmp_root=tmp_root)
            # 从对应 step 继续
            if result.step == "explorer_done":
                print("⏭️  Explorer 已完成，从 Step 2 (Searcher) 继续...")
                rf._resume_from_explorer_done(result, tmp_dir)
                _print_result(result)
                return
            elif result.step == "searcher_done":
                print("⏭️  Searcher 已完成，从 Step 3 (Writer) 继续...")
                rf._resume_from_searcher_done(result, tmp_dir)
                _print_result(result)
                return
    else:
        # ── 交互输入 ──────────────────────────────────────────────────────
        topic = args.topic
        if not topic:
            print("\n📚 ReviewForge 综述流水线")
            print("=" * 50)
            print("输入一个研究领域，我将帮你：")
            print("  1. 探索领域概况（Explorer）")
            print("  2. 检索相关文献（Searcher）")
            print("  3. 撰写综述（Writer）")
            print("=" * 50)
            topic = input("\n请输入要写综述的研究领域（回车使用默认示例）：\n> ").strip()

        if not topic:
            topic = "LoRA large language model fine-tuning"

    slug = _slugify(topic)
    print(f"\n{'='*60}")
    print(f"📝 主题：{topic}")
    print(f"🪣 Slug：{slug}")
    print(f"{'='*60}\n")

    # ── 执行流水线 ────────────────────────────────────────────────────────

    try:
        result = rf.run(topic)
    except Exception as e:
        logger.error("流水线执行失败: %s", e, exc_info=True)
        print(f"\n❌ 失败: {e}")
        sys.exit(1)

    # ── 输出结果 ──────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"✅ 综述流水线完成！")
    print(f"{'='*60}")

    _print_result(result)

    print(f"\n📁 中间结果：{result.tmp_dir}")
    print(f"   Explorer 耗时：{result.explorer_elapsed_seconds:.1f}s")
    if result.searcher_elapsed_seconds:
        print(f"   Searcher 耗时：{result.searcher_elapsed_seconds:.1f}s")
    if result.writer_elapsed_seconds:
        print(f"   Writer   耗时：{result.writer_elapsed_seconds:.1f}s")


# ── 辅助函数 ───────────────────────────────────────────────────────────────

def _slugify(topic: str) -> str:
    import re
    slug = re.sub(r"[\s\-]+", "_", topic.strip())
    slug = re.sub(r"[^\w\u4e00-\u9fa5_]", "", slug)
    slug = slug[:40]
    return slug or "untitled"


def _print_result(result: PipelineResult) -> None:
    """打印 PipelineResult 关键内容"""
    if result.explorer_report:
        er = result.explorer_report
        concepts = er.stage1_concepts[:5] if er.stage1_concepts else []
        classics_count = len(er.stage2_classics) if er.stage2_classics else 0
        benchmarks_count = len(er.stage3_benchmarks) if er.stage3_benchmarks else 0

        print(f"\n📊 Explorer 统计：")
        print(f"   总查询次数：{er.total_queries}")
        print(f"   核心概念：{len(concepts)} 条" if concepts else "   核心概念：（无）")
        for c in concepts:
            print(f"      · {c}")
        print(f"   经典论文：{classics_count} 篇")
        print(f"   Benchmark：{benchmarks_count} 个")

    if result.writer_report:
        wr = result.writer_report
        if wr.title:
            print(f"\n📄 综述标题：{wr.title}")
        if wr.abstract:
            print(f"\n📋 摘要：\n{wr.abstract[:300]}...")
        if wr.keywords:
            print(f"\n🔑 关键词：{', '.join(wr.keywords)}")


if __name__ == "__main__":
    main()
