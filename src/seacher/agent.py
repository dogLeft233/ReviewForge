"""Searcher Agent — 主逻辑"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SearcherAgent:
    """Searcher Agent — 为研究问题生成 arXiv 搜索关键词

    用法:
        from reviewforge.src.llm import LLM
        from reviewforge.src.seacher import SearcherAgent

        llm = LLM(
            api_key="sk-...",
            model="Qwen/Qwen3-8B",
            base_url="https://api.siliconflow.cn/v1",
        )
        agent = SearcherAgent(llm=llm)
        result = agent.run("LoRA 在大模型微调中的应用")
    """

    llm: Any = field(repr=False)
    verbose: bool = False

    def run(self, question: str) -> str:
        """执行搜索关键词生成（原始版，支持网页搜索）"""
        from .tools import search_arxiv_keywords

        if self.verbose:
            logger.info("SearcherAgent 收到问题: %s", question)

        result = search_arxiv_keywords(question=question, llm=self.llm)

        if self.verbose:
            logger.info("生成结果:\n%s", result)

        return result

    def run_with_explorer_report(self, topic: str, er: Any) -> str:
        """基于 Explorer 初步调查结果，继续生成搜索关键词（不强制调用网页搜索）

        参数:
            topic: 用户输入的要写综述的领域
            er: ExplorerAgent 返回的初步调查结果（ExplorerReport）
        """
        from .tools import search_arxiv_keywords_with_explorer

        if self.verbose:
            logger.info(
                "SearcherAgent [with ExplorerReport] 收到领域: %s, "
                "stage1_overview=%s, stage2_classics=%d",
                topic,
                bool(er.stage1_overview),
                len(er.stage2_classics) if er.stage2_classics else 0,
            )

        result = search_arxiv_keywords_with_explorer(
            question=topic,
            er=er,
            llm=self.llm,
        )

        if self.verbose:
            logger.info("生成结果:\n%s", result)

        return result

    def __repr__(self) -> str:
        return f"SearcherAgent(verbose={self.verbose})"
