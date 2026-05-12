"""Explorer Orchestrator — 协调 Explorer 三阶段流程

Stage1 先执行，Stage2 和 Stage3 并行执行，使用 asyncio.Semaphore(4)
控制全局 LLM 并发不超过 4。
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.explorer import ExplorerAgent
from src.explorer.explorer_report import ExplorerReport
from src.llm import LLM

logger = logging.getLogger(__name__)

# 全局并发信号量（模块级单例）
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore(max_concurrency: int) -> asyncio.Semaphore:
    """获取全局 semaphore 单例"""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max_concurrency)
    return _semaphore


class ExplorerOrchestrator:
    """Explorer 协调器——控制三阶段执行顺序和并发

    Stage1 先执行，Stage2 和 Stage3 并行执行。
    所有 LLM 调用受全局 semaphore(max_concurrency) 控制。

    用法:
        from src.agent import ExplorerOrchestrator
        from src.llm import LLM

        llm = LLM(api_key="sk-...", model="Qwen/Qwen3-8B")
        orch = ExplorerOrchestrator(llm)
        report = orch.run_sync("automatic speech recognition")
    """

    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self._max_concurrency = llm._cfg.max_concurrency
        self._executor = ThreadPoolExecutor(max_workers=self._max_concurrency)

    async def run(self, topic: str) -> ExplorerReport:
        """执行三阶段探索：

        1. Stage1 先执行
        2. Stage2 + Stage3 并行执行
        3. 汇总报告
        """
        report = ExplorerReport(topic=topic)

        # Stage 1（不受 semaphore 限制，串行执行）
        logger.info("[Orchestrator] Stage 1: 领域概况")
        stage1_result = await self._run_stage1_async(topic)
        report.stage1_overview = stage1_result["overview"]
        report.stage1_concepts = stage1_result["concepts"]
        report.stage1_search_results = stage1_result["raw"]
        report.total_queries = stage1_result["query_count"]

        # Stage 2 & 3 并行（受全局 semaphore 控制）
        logger.info("[Orchestrator] Stage 2 & 3 并行执行（max concurrency=%d)", self._max_concurrency)
        stage2_future = asyncio.create_task(self._run_stage2_async(topic))
        stage3_future = asyncio.create_task(self._run_stage3_async(topic))

        stage2_result = await stage2_future
        stage3_result = await stage3_future

        report.stage2_classics = stage2_result["classics"]
        report.stage2_timeline = stage2_result["timeline"]
        report.stage2_search_results = stage2_result["raw"]
        report.total_queries += stage2_result["query_count"]

        report.stage3_benchmarks = stage3_result["benchmarks"]
        report.stage3_state_of_art = stage3_result["sota"]
        report.stage3_trends = stage3_result["trends"]
        report.stage3_search_results = stage3_result["raw"]
        report.total_queries += stage3_result["query_count"]

        # Synthesis
        logger.info("[Orchestrator] 生成下游报告")
        loop = asyncio.get_event_loop()
        report.downstream_report = await loop.run_in_executor(
            self._executor,
            self._synthesize,
            topic,
            report,
        )

        logger.info("[Orchestrator] 完成，共执行 %d 次搜索", report.total_queries)
        return report

    async def _run_stage1_async(self, topic: str) -> dict[str, Any]:
        """异步执行 Stage1（在 executor 中运行）"""
        explorer = ExplorerAgent(llm=self._llm)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, explorer._run_stage1, topic)

    async def _run_stage2_async(self, topic: str) -> dict[str, Any]:
        """异步执行 Stage2（受全局并发限制）"""
        sem = _get_semaphore(self._max_concurrency)
        async with sem:
            logger.debug("[Orchestrator] Stage2 获取 semaphore，开始执行")
            explorer = ExplorerAgent(llm=self._llm)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, explorer._run_stage2, topic)

    async def _run_stage3_async(self, topic: str) -> dict[str, Any]:
        """异步执行 Stage3（受全局并发限制）"""
        sem = _get_semaphore(self._max_concurrency)
        async with sem:
            logger.debug("[Orchestrator] Stage3 获取 semaphore，开始执行")
            explorer = ExplorerAgent(llm=self._llm)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, explorer._run_stage3, topic)

    def _synthesize(self, topic: str, report: ExplorerReport) -> str:
        """综合报告（同步，在 executor 中运行）"""
        explorer = ExplorerAgent(llm=self._llm)
        return explorer._synthesize(topic, report)

    def run_sync(self, topic: str) -> ExplorerReport:
        """同步入口（用于非 async 上下文）"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.run(topic))
        finally:
            loop.close()