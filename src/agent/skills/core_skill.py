"""Core Pipeline Skill — ReviewForge Explorer for ChatAgent

将 ReviewForge.run() 完整流程封装为工具，支持：
- run_explorer: 仅 Explorer 阶段
- run_pipeline: 完整流程 Explorer → Searcher → Writer
- 中间结果实时输出到 UI
- 阶段执行进度通知
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Annotated, Iterator

from src.config import settings
from src.logging_config import get_logger
from src.core import CoreConfig, ReviewForge, PipelineResult

logger = get_logger(__name__)


def _write_task_status(topic: str, status: dict[str, Any]) -> None:
    """写入任务状态文件，用于 UI 轮询"""
    slug = _slugify(topic)
    status_dir = f"tmp/{slug}"
    os.makedirs(status_dir, exist_ok=True)
    status_file = f"{status_dir}/task_status.json"
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def _log_network_env() -> None:
    """打印网络/代理环境诊断日志（仅打印一次）"""
    import os, httpx
    env_check = {
        "HTTP_PROXY": os.environ.get("HTTP_PROXY") or "(empty)",
        "HTTPS_PROXY": os.environ.get("HTTPS_PROXY") or "(empty)",
        "ALL_PROXY": os.environ.get("ALL_PROXY") or "(empty)",
    }
    has_proxy = any(v != "(empty)" for v in env_check.values())
    logger.info("[NetworkEnv] proxies configured: %s | env: %s", has_proxy, env_check)
    # httpx trust_env status
    c = httpx.Client(timeout=5.0)
    logger.info("[NetworkEnv] httpx trust_env=%s, HTTP_PROXY=%s", c._trust_env, os.environ.get("HTTP_PROXY", "(none)"))
    c.close()


def _build_llm():
    """从 settings 构建 LLM"""
    from src.llm import LLM
    api_key = getattr(settings, "llm_api_key", "") or ""
    model = getattr(settings, "llm_model", "Qwen/Qwen3-8B")
    base_url = getattr(settings, "llm_base_url", "https://api.siliconflow.cn/v1")
    return LLM(api_key=api_key, model=model, base_url=base_url, timeout_seconds=300)


def _build_rf():
    """构建 ReviewForge 实例（配置与 run_pipeline.py 一致）"""
    llm = _build_llm()
    config = CoreConfig(
        bfs_expand_layers=1,
        bfs_expand_papers_count=5,
        bfs_search_queries_count=3,
        bfs_search_papers_count=5,
        bfs_similarity_threshold=0.20,
        bfs_rerank_top_n=100,
        bfs_enabled=True,
    )
    return ReviewForge(llm=llm, config=config)


def _progressyield(
    topic: str,
    stage: str,
    progress: float,
    message: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """生成进度消息，并更新 task_status.json"""
    status = {
        "task_id": "pipeline",
        "topic": topic,
        "status": "running",
        "stage": stage,
        "progress": progress,
        "message": message,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        status.update(extra)
    _write_task_status(topic, status)
    return message


def _slugify(topic: str) -> str:
    """将主题转换为安全的目录名"""
    import re
    slug = re.sub(r"[\s\-]+", "_", topic.strip())
    slug = re.sub(r"[^\w一-龥_]", "", slug)
    slug = slug[:40]
    return slug or "untitled"


# ── 工具函数 ────────────────────────────────────────────────


def run_explorer(topic: Annotated[str, "要探索的研究领域或主题"]) -> str:
    """仅执行 Explorer 阶段（领域探索），返回格式化结果。"""
    try:
        _log_network_env()
        logger.info("run_explorer called with topic=%r", topic[:30])
        rf = _build_rf()
        result = rf.run(topic)

        lines = [f"# 领域探索结果：{topic}\n"]
        er = result.explorer_report
        if er.stage1_overview:
            lines.append("## 📖 领域概况")
            lines.append(er.stage1_overview.strip())
            lines.append("")
        if er.stage1_concepts:
            lines.append("## 💡 核心概念")
            for c in er.stage1_concepts:
                lines.append(f"- {c}")
            lines.append("")
        if er.stage2_classics:
            lines.append("## 📚 经典论文")
            for cw in er.stage2_classics:
                lines.append(f"- **{cw.title}** ({cw.year})")
            lines.append("")
        if er.stage3_benchmarks:
            lines.append("## 🏆 Benchmark")
            for b in er.stage3_benchmarks:
                lines.append(f"- **{b.name}** ({b.metric} on {b.dataset})")
        return "\n".join(lines)
    except Exception as e:
        logger.error("run_explorer 失败: %s", e, exc_info=True)
        return f"（领域探索失败: {e}）"


def run_explorer_async(topic: Annotated[str, "要探索的研究领域或主题"]) -> str:
    """异步执行 Explorer 阶段，立即返回任务ID，后台执行不阻塞。"""
    import uuid
    import threading
    task_id = uuid.uuid4().hex[:8]

    _write_task_status(topic, {
        "task_id": task_id,
        "topic": topic,
        "status": "running",
        "stage": "explorer",
        "progress": 0.0,
        "message": "🚀 正在初始化领域探索...",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    def _do():
        try:
            rf = _build_rf()

            _write_task_status(topic, {
                "task_id": task_id,
                "topic": topic,
                "status": "running",
                "stage": "stage1_explorer",
                "progress": 0.3,
                "message": "🔍 Stage 1/3：生成领域概况和核心概念...",
            })

            result = rf.run(topic)

            _write_task_status(topic, {
                "task_id": task_id,
                "topic": topic,
                "status": "completed",
                "stage": "explorer_done",
                "progress": 1.0,
                "message": "✅ 领域探索完成！",
                "result_file": f"tmp/{_slugify(topic)}/step1_explorer_done.json",
            })
        except Exception as e:
            logger.error("run_explorer_async 失败: %s", e)
            _write_task_status(topic, {
                "task_id": task_id,
                "topic": topic,
                "status": "failed",
                "stage": "error",
                "progress": 0.0,
                "message": f"❌ 探索失败: {e}",
                "error": str(e),
            })

    thread = threading.Thread(target=_do)
    thread.daemon = True
    thread.start()
    return f"⏳ 领域探索任务已启动（topic: {topic}），UI 将自动更新进度..."


def run_searcher(topic: Annotated[str, "要执行文献搜索的研究领域"]) -> str:
    """执行 Searcher 阶段（BFSSearch + MultiSource），依赖 tmp/{topic}/step1_explorer_done.json。"""
    try:
        rf = _build_rf()
        slug = _slugify(topic)
        tmp_dir = rf.tmp_root / slug

        step1_file = tmp_dir / "step1_explorer_done.json"
        if step1_file.exists():
            result = PipelineResult.load(step1_file)
            if result.step == "explorer_done":
                result = rf._resume_from_explorer_done(result, tmp_dir)
                papers_count = len(result.searcher_papers) if result.searcher_papers else 0
                return f"✅ Searcher 完成，获取 **{papers_count}** 篇论文，耗时 {result.searcher_elapsed_seconds:.1f}s"
            elif result.step == "complete":
                return f"该 topic 已完成全部流程（Explorer → Searcher → Writer）"
        return f"（未找到 tmp/{slug}/step1_explorer_done.json，请先运行 run_explorer）"
    except Exception as e:
        logger.error("run_searcher 失败: %s", e, exc_info=True)
        return f"（Searcher 失败: {e}）"


def run_pipeline(topic: Annotated[str, "要执行完整流水线的研究领域"]) -> str:
    """执行完整流水线：Explorer → Searcher → Writer。委托给 scripts/run_pipeline.py 执行。"""
    try:
        _log_network_env()
        logger.info("run_pipeline called with topic=%r", topic[:30])

        project_root = Path(__file__).resolve().parents[3]
        script_path = project_root / "scripts" / "run_pipeline.py"

        _write_task_status(topic, {
            "task_id": "pipeline",
            "topic": topic,
            "status": "running",
            "stage": "init",
            "progress": 0.0,
            "message": "🚀 完整流水线启动（Explorer → Searcher → Writer）...",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        slug = _slugify(topic)
        log_basename = f"run_pipeline_{slug}_{time.strftime('%Y%m%d_%H%M%S')}.log"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)

        result = subprocess.run(
            [sys.executable, str(script_path), "--topic", topic, "--log-file", log_basename],
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            logger.error("run_pipeline.py failed with code %d: %s", result.returncode, result.stderr)
            _write_task_status(topic, {
                "task_id": "pipeline",
                "topic": topic,
                "status": "failed",
                "stage": "error",
                "progress": 0.0,
                "message": f"❌ 流水线失败: {result.stderr or 'script exited with code ' + str(result.returncode)}",
                "error": result.stderr,
            })
            return f"（流水线失败: {result.stderr or result.stdout or 'exit code ' + str(result.returncode)}）"

        _write_task_status(topic, {
            "task_id": "pipeline",
            "topic": topic,
            "status": "completed",
            "stage": "complete",
            "progress": 1.0,
            "message": "🎉 完整流水线完成！",
        })

        try:
            import streamlit as st
            st.session_state["_pipeline_completed"] = {"topic": topic, "slug": slug}
        except Exception:
            pass

        return result.stdout if result.stdout else "✅ 流水线执行完成"

    except subprocess.TimeoutExpired:
        logger.error("run_pipeline timed out after 3600s")
        _write_task_status(topic, {
            "task_id": "pipeline",
            "topic": topic,
            "status": "failed",
            "stage": "error",
            "progress": 0.0,
            "message": "❌ 流水线超时（3600s）",
            "error": "timeout",
        })
        return "（流水线超时）"
    except Exception as e:
        logger.error("run_pipeline 失败: %s", e, exc_info=True)
        _write_task_status(topic, {
            "task_id": "pipeline",
            "topic": topic,
            "status": "failed",
            "stage": "error",
            "progress": 0.0,
            "message": f"❌ 流水线失败: {e}",
            "error": str(e),
        })
        return f"（流水线失败: {e}）"


def run_pipeline_async(topic: Annotated[str, "要执行完整流水线的研究领域"]) -> str:
    """异步执行完整流水线，立即返回任务ID，后台执行不阻塞。"""
    import uuid
    task_id = uuid.uuid4().hex[:8]

    _write_task_status(topic, {
        "task_id": task_id,
        "topic": topic,
        "status": "running",
        "stage": "init",
        "progress": 0.0,
        "message": "🚀 完整流水线已启动（Explorer → Searcher → Writer）...",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    project_root = Path(__file__).resolve().parents[3]
    script_path = project_root / "scripts" / "run_pipeline.py"

    def _do():
        try:
            slug = _slugify(topic)
            log_basename = f"run_pipeline_{slug}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(project_root)
            proc = subprocess.Popen(
                [sys.executable, str(script_path), "--topic", topic, "--log-file", log_basename],
                cwd=str(project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate(timeout=3600)

            if proc.returncode != 0:
                logger.error("run_pipeline_async failed: %s", stderr.decode())
                _write_task_status(topic, {
                    "task_id": task_id,
                    "topic": topic,
                    "status": "failed",
                    "stage": "error",
                    "progress": 0.0,
                    "message": f"❌ 流水线失败（exit code {proc.returncode}）",
                })
            else:
                logger.info("run_pipeline_async completed: %s", topic[:30])
                _write_task_status(topic, {
                    "task_id": task_id,
                    "topic": topic,
                    "status": "completed",
                    "stage": "complete",
                    "progress": 1.0,
                    "message": "🎉 完整流水线完成！",
                })
        except Exception as e:
            logger.error("run_pipeline_async 失败: %s", e)
            _write_task_status(topic, {
                "task_id": task_id,
                "topic": topic,
                "status": "failed",
                "stage": "error",
                "progress": 0.0,
                "message": f"❌ 流水线失败: {e}",
                "error": str(e),
            })

    import threading
    thread = threading.Thread(target=_do)
    thread.daemon = True
    thread.start()
    return f"⏳ 完整流水线任务已启动（topic: {topic}），后台执行中，进度请查看 task_status.json..."


def make_explorer_tool():
    """返回 LangChain StructuredTool（兼容旧接口）"""
    from langchain_core.tools import StructuredTool
    return StructuredTool(
        name="run_explorer",
        description="执行领域探索，获取研究领域的基本概况、核心概念、经典论文和 Benchmark。",
        func=run_explorer,
        args_schema={
            "topic": {
                "type": "string",
                "description": "要探索的研究领域或主题",
                "examples": ["LoRA大模型微调", "扩散模型图像生成"],
            },
        },
    )