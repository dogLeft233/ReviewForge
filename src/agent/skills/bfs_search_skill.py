"""BFS Search Skill — 深度论文发现 Skill for ChatAgent (Async Version)"""
from __future__ import annotations

import json
import os
import time
import hashlib
from typing import Annotated

from src.seacher.bfs_search import BFSSearcher
from src.llm import LLM
from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


def _write_task_status(question: str, status: dict) -> None:
    safe_name = hashlib.md5(question.encode()).hexdigest()[:8]
    topic_dir = f"tmp/BFS_{safe_name}"
    os.makedirs(topic_dir, exist_ok=True)
    with open(f"{topic_dir}/task_status.json", "w", encoding="utf-8") as f:
        json.dump(status, ensure_ascii=False, indent=2)


def _bfs_search(question: str, expand_layers: int = 2, search_papers_count: int = 20) -> str:
    """同步版本（保留兼容）"""
    result = _bfs_search_sync(question, expand_layers, search_papers_count)
    return _format_bfs_result(result, question)


def _bfs_search_sync(question: str, expand_layers: int, search_papers_count: int) -> list:
    from src.embedding import EmbeddingClient
    from src.reranker import RerankerClient

    api_key = getattr(settings, "llm_api_key", "") or ""
    model = getattr(settings, "llm_model", "Qwen/Qwen3-8B")
    base_url = getattr(settings, "llm_base_url", "https://api.siliconflow.cn/v1")
    llm = LLM(api_key=api_key, model=model, base_url=base_url)

    embed = EmbeddingClient()
    reranker = RerankerClient()

    searcher = BFSSearcher(
        llm=llm, embed=embed, reranker=reranker,
        search_papers_count=search_papers_count, expand_layers=expand_layers,
    )
    return searcher.search(question)


def _bfs_search_async(
    question: Annotated[str, "研究问题或主题"],
    expand_layers: Annotated[int, "BFS 扩展层数"] = 2,
    search_papers_count: Annotated[int, "每个搜索词取多少篇论文"] = 20,
) -> str:
    """异步 BFS 搜索 — 阶段进度写入 task_status.json，完成后写入 step2_searcher_done.json"""
    import uuid
    task_id = uuid.uuid4().hex[:8]
    safe_name = hashlib.md5(question.encode()).hexdigest()[:8]
    topic_dir = f"tmp/BFS_{safe_name}"
    os.makedirs(topic_dir, exist_ok=True)

    _write_task_status(question, {
        "task_id": task_id, "question": question,
        "status": "running", "stage": "llm_queries",
        "progress": 0.05,
        "message": "🧠 正在用 LLM 生成搜索词...",
    })

    try:
        from src.embedding import EmbeddingClient
        from src.reranker import RerankerClient

        api_key = getattr(settings, "llm_api_key", "") or ""
        model = getattr(settings, "llm_model", "Qwen/Qwen3-8B")
        base_url = getattr(settings, "llm_base_url", "https://api.siliconflow.cn/v1")
        llm = LLM(api_key=api_key, model=model, base_url=base_url)
        embed = EmbeddingClient()
        reranker = RerankerClient()

        _write_task_status(question, {
            "task_id": task_id, "question": question,
            "status": "running", "stage": "searching",
            "progress": 0.15,
            "message": f"🔍 正在搜索 arXiv（expand_layers={expand_layers}）...",
        })

        searcher = BFSSearcher(
            llm=llm, embed=embed, reranker=reranker,
            search_papers_count=search_papers_count, expand_layers=expand_layers,
        )
        papers = searcher.search(question)

        _write_task_status(question, {
            "task_id": task_id, "question": question,
            "status": "running", "stage": "saving",
            "progress": 0.9,
            "message": f"💾 找到 {len(papers)} 篇论文，写入结果文件...",
        })

        # 写入结果（与手动运行格式一致）
        result_file = f"{topic_dir}/step2_searcher_done.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "question": question,
                "total_papers": len(papers),
                "papers": [
                    {
                        "title": p.title, "paper_id": p.paper_id,
                        "abstract": p.abstract, "select_score": p.select_score,
                        "depth": p.depth, "source": p.source,
                    }
                    for p in papers
                ],
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, ensure_ascii=False, indent=2)

        _write_task_status(question, {
            "task_id": task_id, "question": question,
            "status": "completed", "stage": "done",
            "progress": 1.0,
            "message": f"✅ BFS 完成！找到 {len(papers)} 篇论文",
            "result_file": result_file,
        })

        return _format_bfs_result(papers, question)

    except Exception as e:
        logger.error("bfs_search 失败: %s", e, exc_info=True)
        _write_task_status(question, {
            "task_id": task_id, "question": question,
            "status": "failed", "stage": "error",
            "progress": 0, "message": f"❌ BFS 搜索失败: {e}",
            "error": str(e),
        })
        return f"（BFS 搜索失败: {e}）"


def _format_bfs_result(papers: list, question: str) -> str:
    if not papers:
        return f"（未找到与「{question}」相关的论文）"
    lines = [f"# BFS 论文发现结果：{question}\n"]
    lines.append(f"共找到 {len(papers)} 篇论文（按相关性排序）：\n")
    for i, p in enumerate(papers[:30], 1):
        score_note = f"⭐{p.select_score:.3f}" if p.select_score > 0 else ""
        lines.append(f"{i}. **{p.title}** {score_note}")
        if p.paper_id:
            lines.append(f"   arXiv: https://arxiv.org/abs/{p.paper_id}")
        if p.abstract:
            lines.append(f"   {p.abstract[:200]}")
        if p.depth > 0:
            lines.append(f"   [引文 depth={p.depth}, 来源: {p.source}]")
        lines.append("")
    return "\n".join(lines)


def make_bfs_search_tool():
    """返回 LangChain StructuredTool（兼容旧接口）"""
    from langchain_core.tools import StructuredTool
    return StructuredTool(
        name="bfs_search",
        description="深度论文发现，使用 BFS 广度优先搜索发现领域重要论文。",
        func=_bfs_search,
        args_schema={
            "question": {"type": "string", "description": "研究问题或主题"},
            "expand_layers": {"type": "int", "description": "BFS 扩展层数（默认2）", "default": 2},
            "search_papers_count": {"type": "int", "description": "每个搜索词取多少篇论文", "default": 20},
        },
    )
