"""Searcher Agent — 主逻辑"""

from __future__ import annotations

import logging
import sys
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
        """执行搜索关键词生成

        参数:
            question: 用户的研究问题

        返回:
            生成的 arXiv 搜索关键词建议
        """
        from .tools import search_arxiv_keywords

        if self.verbose:
            logger.info("SearcherAgent 收到问题: %s", question)

        result = search_arxiv_keywords(question=question, llm=self.llm)

        if self.verbose:
            logger.info("生成结果:\\n%s", result)

        return result

    def __repr__(self) -> str:
        return f"SearcherAgent(verbose={self.verbose})"
