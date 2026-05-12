"""Core Pipeline Skill — ReviewForge Explorer for ChatAgent (Async Version)

将 ReviewForge.run_explorer() 封装为异步任务，支持：
- 中间结果实时输出到 UI
- 完成后自动写入 tmp/{topic}/step1_explorer_done.json
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Annotated, Generator

from src.core import ReviewForge, ExplorerReport
from src.llm import LLM
from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


def _write_task_status(topic: str, status: dict[str, Any]) -> None:
    """写入任务状态文件，用于 UI 轮询"""
    import hashlib
    safe_name = hashlib.md5(topic.encode()).hexdigest()[:8]
    status_dir = f"tmp/{topic}"
    os.makedirs(status_dir, exist_ok=True)
    status_file = f"{status_dir}/task_status.json"
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status, ensure_ascii=False, indent=2)


def _run_explorer(topic: Annotated[str, "要探索的研究领域或主题"]) -> str:
    """同步版本（保留兼容）"""
    result = _run_explorer_sync(topic)
    return _format_explorer_result(result)


def _run_explorer_sync(topic: str) -> ExplorerReport:
    """同步执行领域探索"""
    api_key = getattr(settings, "llm_api_key", "") or ""
    model = getattr(settings, "llm_model", "Qwen/Qwen3-8B")
    base_url = getattr(settings, "llm_base_url", "https://api.siliconflow.cn/v1")

    llm = LLM(api_key=api_key, model=model, base_url=base_url, timeout_seconds=300)
    return ReviewForge.run_explorer(topic, llm, verbose=False)


def _run_explorer_async(topic: Annotated[str, "要探索的研究领域或主题"]) -> str:
    """异步版本：启动后台任务，立即返回任务ID，阶段进度实时写入 task_status.json

    用于 ChatAgent 的流式响应，UI 端轮询 task_status.json 获取进度。
    完成后写入 tmp/{topic}/step1_explorer_done.json（与手动运行格式一致）。
    """
    import uuid
    task_id = uuid.uuid4().hex[:8]

    # 初始化任务状态
    _write_task_status(topic, {
        "task_id": task_id,
        "topic": topic,
        "status": "running",
        "stage": "initializing",
        "progress": 0.0,
        "message": "🚀 正在初始化领域探索...",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    try:
        api_key = getattr(settings, "llm_api_key", "") or ""
        model = getattr(settings, "llm_model", "Qwen/Qwen3-8B")
        base_url = getattr(settings, "llm_base_url", "https://api.siliconflow.cn/v1")

        llm = LLM(api_key=api_key, model=model, base_url=base_url, timeout_seconds=300)

        # Stage 1
        _write_task_status(topic, {
            "task_id": task_id,
            "topic": topic,
            "status": "running",
            "stage": "stage1",
            "progress": 0.1,
            "message": "🔍 正在生成领域概况和核心概念（Stage 1/3）...",
        })

        stage1_result = ReviewForge._run_stage1(topic, llm, verbose=False)

        _write_task_status(topic, {
            "task_id": task_id,
            "topic": topic,
            "status": "running",
            "stage": "stage2",
            "progress": 0.4,
            "message": "📚 正在查找经典论文（Stage 2/3）...",
        })

        stage2_result = ReviewForge._run_stage2(topic, llm, stage1_result=stage1_result, verbose=False)

        _write_task_status(topic, {
            "task_id": task_id,
            "topic": topic,
            "status": "running",
            "stage": "stage3",
            "progress": 0.7,
            "message": "🏆 正在梳理 Benchmark（Stage 3/3）...",
        })

        stage3_result = ReviewForge._run_stage3(topic, llm, stage1_result=stage1_result, verbose=False)

        # 组装 ExplorerReport
        report = ExplorerReport(
            topic=topic,
            stage1_overview=stage1_result.get("overview", ""),
            stage2_classics=stage2_result.get("classics", []),
            stage3_benchmarks=stage3_result.get("benchmarks", []),
        )

        # 写入结果文件（供其他 Tab 加载）
        topic_dir = f"tmp/{topic}"
        os.makedirs(topic_dir, exist_ok=True)
        result_file = f"{topic_dir}/step1_explorer_done.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "topic": topic,
                "explorer_report": {
                    "topic": topic,
                    "stage1_overview": report.stage1_overview,
                    "stage2_classics": [
                        {"title": c.title, "year": c.year, "authors": c.authors,
                         "venue": c.venue, "key_idea": c.key_idea, "impact": c.impact}
                        for c in (report.stage2_classics or [])
                    ],
                    "stage3_benchmarks": [
                        {"name": b.name, "metric": b.metric, "dataset": b.dataset,
                         "description": b.description, "url": b.url}
                        for b in (report.stage3_benchmarks or [])
                    ],
                },
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, ensure_ascii=False, indent=2)

        _write_task_status(topic, {
            "task_id": task_id,
            "topic": topic,
            "status": "completed",
            "stage": "done",
            "progress": 1.0,
            "message": "✅ 领域探索完成！",
            "result_file": result_file,
        })

        return _format_explorer_result(report)

    except Exception as e:
        logger.error("run_explorer 失败: %s", e, exc_info=True)
        _write_task_status(topic, {
            "task_id": task_id,
            "topic": topic,
            "status": "failed",
            "stage": "error",
            "progress": 0.0,
            "message": f"❌ 探索失败: {e}",
            "error": str(e),
        })
        return f"（领域探索失败: {e}）"


def _format_explorer_result(report: ExplorerReport) -> str:
    """将 ExplorerReport 格式化为 Markdown 文本"""
    lines = [f"# 领域探索结果：{report.topic}\n"]

    if report.stage1_overview:
        lines.append("## 📖 领域概况")
        lines.append(report.stage1_overview.strip())
        lines.append("")

    if report.stage2_classics:
        lines.append("## 📚 经典论文")
        for i, cw in enumerate(report.stage2_classics, 1):
            lines.append(f"{i}. **{cw.title}** ({cw.year})")
            if cw.authors:
                lines.append(f"   作者: {cw.authors}")
            if cw.venue:
                lines.append(f"   场所: {cw.venue}")
            if cw.key_idea:
                lines.append(f"   核心思想: {cw.key_idea}")
            if cw.impact:
                lines.append(f"   影响: {cw.impact}")
            lines.append("")
    else:
        lines.append("## 📚 经典论文（暂无）")

    if report.stage3_benchmarks:
        lines.append("## 🏆 重要 Benchmark")
        for b in report.stage3_benchmarks:
            lines.append(f"- **{b.name}**")
            if b.metric:
                lines.append(f"  指标: {b.metric}")
            if b.dataset:
                lines.append(f"  数据集: {b.dataset}")
            if b.description:
                lines.append(f"  说明: {b.description}")
            if b.url:
                lines.append(f"  链接: {b.url}")
    else:
        lines.append("## 🏆 重要 Benchmark（暂无）")

    return "\n".join(lines)


def make_explorer_tool():
    """返回 LangChain StructuredTool（兼容旧接口）"""
    from langchain_core.tools import StructuredTool
    return StructuredTool(
        name="run_explorer",
        description="执行领域探索，获取研究领域的基本概况、核心概念、经典论文和 Benchmark。",
        func=_run_explorer,
        args_schema={
            "topic": {
                "type": "string",
                "description": "要探索的研究领域或主题",
                "examples": ["LoRA大模型微调", "扩散模型图像生成"],
            },
        },
    )
