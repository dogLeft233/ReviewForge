"""Multi-Source Searcher Skill — 多源检索 Skill for ChatAgent"""

from typing import Any
from langchain_core.tools import StructuredTool
from src.seacher.tools.multi_source_searcher import MultiSourceSearcher
from src.llm import LLM, Message
from src.config import settings


def _multi_source_search(question: str) -> str:
    """执行多源检索，同时搜索 GitHub / HuggingFace 模型 / HuggingFace 数据集 / arXiv。

    适用场景：
    - 了解某技术/领域的最新实现和资源
    - 找 GitHub 上的开源项目
    - 找 HuggingFace 上的模型和数据集
    - 找最新的 arXiv 论文

    Args:
        question: 研究问题或技术主题

    Returns:
        多源搜索结果（各来源汇总）
    """
    api_key = getattr(settings, "llm_api_key", "")
    model = getattr(settings, "llm_model", "Qwen/Qwen3-8B")
    base_url = getattr(settings, "llm_base_url", "https://api.siliconflow.cn/v1")

    llm = LLM(api_key=api_key, model=model, base_url=base_url)
    searcher = MultiSourceSearcher(llm)

    result, _ = searcher.run(question, messages=None)
    return result


def make_multi_source_searcher_tool() -> StructuredTool:
    return StructuredTool(
        name="multi_source_search",
        description="""多源检索工具，同时搜索 GitHub（项目）、HuggingFace（模型+数据集）和 arXiv（论文）。

适用场景：
- 了解某技术/领域的最新实现和资源
- 找 GitHub 上的高质量开源项目
- 找 HuggingFace 上的预训练模型和数据集
- 找最新的 arXiv 论文（特别是 Survey/Review 类）
- 调研某领域现状时需要多维度的资源覆盖

每次搜索会自动：生成各平台的搜索词 → 并行执行 → 汇总结果。最多进行 2 轮搜索，第 2 轮后给出最终回答。""",
        func=_multi_source_search,
        args_schema={
            "question": {"type": "string", "description": "研究问题或技术主题"},
        },
    )