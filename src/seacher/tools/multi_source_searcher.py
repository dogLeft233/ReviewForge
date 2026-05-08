"""多源检索器 — GitHub + HuggingFace + arXiv 三路并行搜索

支持多轮对话，上一轮记忆通过 messages 列表保持。
每次调用 llm.chat() 传入完整历史，LLM 可自主决定调用哪些数据源。

用法:
    from src.seacher.tools.multi_source_searcher import MultiSourceSearcher
    from src.llm import LLM, Message

    llm = LLM(api_key="sk-...", model="Qwen/Qwen3-8B")
    searcher = MultiSourceSearcher(llm)
    messages = []  # 首次调用为空
    result, messages = searcher.run("LoRA fine-tuning 最新实现", messages=messages)
    # 下次调用传入同一 messages 即可保持上下文
    result2, messages = searcher.run("能用在视觉Transformer上吗", messages=messages)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.llm import Message

logger = logging.getLogger(__name__)

# ── 搜索最佳姿势（运行时加载）────────────────────────────────

_SEARCH_TIPS_PATH = Path(__file__).parent.parent / "docs" / "最佳搜索姿势.md"


def _load_search_tips() -> str:
    """加载搜索技巧文档"""
    if _SEARCH_TIPS_PATH.exists():
        return _SEARCH_TIPS_PATH.read_text(encoding="utf-8")
    # 回退：内置最常用技巧
    return """
## GitHub 搜索运算符
- in:name,description,readme  字段搜索
- stars:>N  forks:>N  语言筛选
- pushed:>YYYY-MM-DD  时间筛选
- 组合示例: "transformer" in:name stars:>5000 language:python pushed:>2024-01-01

## HuggingFace 搜索
- search_models(query) / search_datasets(query)  关键字搜索
- sort="downloads"  按下载量排序
- task=  任务类型: text-generation, image-classification 等
- framework=  框架: pytorch, tensorflow

## arXiv 搜索
- ti:标题 abs:摘要 cat:分类（全字段搜索，不要混用 ti:/cat:）
- 布尔: AND OR ANDNOT
- 日期: submittedDate:[YYYYMMDD TO YYYYMMDD]
"""


# ── LLM 输出解析 ───────────────────────────────────────────────

_PAT_GITHUB = re.compile(r"```\s*github\s+(.+?)```\s*", re.DOTALL | re.IGNORECASE)
_PAT_HF_MODEL = re.compile(r"```\s*hf\s+models?\s+(.+?)```\s*", re.DOTALL | re.IGNORECASE)
_PAT_HF_DATASET = re.compile(r"```\s*hf\s+datasets?\s+(.+?)```\s*", re.DOTALL | re.IGNORECASE)
_PAT_ARXIV = re.compile(r"```\s*arxiv\s+(.+?)```\s*", re.DOTALL | re.IGNORECASE)


def _parse_search_queries(text: str) -> dict[str, list[str]]:
    """从 LLM 回复中解析各类搜索指令

    Returns:
        {
            "github": ["query1", "query2"],
            "hf_models": ["..."],
            "hf_datasets": ["..."],
            "arxiv": ["..."],
        }
    """
    result: dict[str, list[str]] = {
        "github": [],
        "hf_models": [],
        "hf_datasets": [],
        "arxiv": [],
    }

    for m in _PAT_GITHUB.finditer(text):
        q = m.group(1).strip()
        if q:
            result["github"].append(q)

    for m in _PAT_HF_MODEL.finditer(text):
        q = m.group(1).strip()
        if q:
            result["hf_models"].append(q)

    for m in _PAT_HF_DATASET.finditer(text):
        q = m.group(1).strip()
        if q:
            result["hf_datasets"].append(q)

    for m in _PAT_ARXIV.finditer(text):
        q = m.group(1).strip()
        if q:
            result["arxiv"].append(q)

    return result


# ── 执行搜索 ─────────────────────────────────────────────────

def _search_github(query: str, max_results: int = 10) -> str:
    """执行 GitHub 搜索"""
    try:
        from src.retrievers import GithubRetriever

        with GithubRetriever() as gh:
            results = gh.search_resources(query, max_results=max_results)
        if not results:
            return f"（GitHub 无结果: {query}）"
        lines = [f"## GitHub: {query}"]
        for r in results[:8]:
            lines.append(f"- [{r.title}]({r.url})  ⭐{r.likes_or_stars or '?'}  🍴{r.forks or '?'}")
            if r.description:
                lines.append(f"  {r.description[:150]}")
        return "\n".join(lines)
    except Exception as e:
        return f"（GitHub 搜索失败: {e})"


def _search_hf_models(query: str, max_results: int = 10) -> str:
    """执行 HuggingFace 模型搜索"""
    try:
        from src.retrievers import HuggingFaceRetriever

        with HuggingFaceRetriever() as hf:
            results = hf.search_models(query, max_results=max_results)
        if not results:
            return f"（HuggingFace 模型无结果: {query}）"
        lines = [f"## HuggingFace 模型: {query}"]
        for r in results[:8]:
            lines.append(f"- [{r.title}]({r.url})  ⬇{r.downloads or '?'}")
            if r.description:
                lines.append(f"  {r.description[:150]}")
        return "\n".join(lines)
    except Exception as e:
        return f"（HuggingFace 模型搜索失败: {e})"


def _search_hf_datasets(query: str, max_results: int = 10) -> str:
    """执行 HuggingFace 数据集搜索"""
    try:
        from src.retrievers import HuggingFaceRetriever

        with HuggingFaceRetriever() as hf:
            results = hf.search_datasets(query, max_results=max_results)
        if not results:
            return f"（HuggingFace 数据集无结果: {query}）"
        lines = [f"## HuggingFace 数据集: {query}"]
        for r in results[:8]:
            lines.append(f"- [{r.title}]({r.url})  ⬇{r.downloads or '?'}")
            if r.description:
                lines.append(f"  {r.description[:150]}")
        return "\n".join(lines)
    except Exception as e:
        return f"（HuggingFace 数据集搜索失败: {e})"


def _search_arxiv(query: str, max_results: int = 10) -> str:
    """执行 arXiv 搜索"""
    try:
        from src.retrievers import ArxivRetriever

        with ArxivRetriever() as arxiv:
            results = arxiv.search(query, max_results=max_results)
        if not results:
            return f"（arXiv 无结果: {query}）"
        lines = [f"## arXiv: {query}"]
        for r in results[:8]:
            lines.append(f"- [{r.title}]({r.url})")
            if r.abstract:
                lines.append(f"  {r.abstract[:200]}")
        return "\n".join(lines)
    except Exception as e:
        return f"（arXiv 搜索失败: {e})"


def _execute_all_queries(queries: dict[str, list[str]]) -> str:
    """执行所有搜索查询，合并结果"""
    sections: list[str] = []

    for q in queries.get("github", []):
        sections.append(_search_github(q))

    for q in queries.get("hf_models", []):
        sections.append(_search_hf_models(q))

    for q in queries.get("hf_datasets", []):
        sections.append(_search_hf_datasets(q))

    for q in queries.get("arxiv", []):
        sections.append(_search_arxiv(q))

    if not sections:
        return "（未识别到有效搜索指令）"

    return "\n\n".join(sections)


# ── 系统提示词 ───────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """你是一个研究助手，负责为用户问题生成多源检索关键词。

## 你的任务
1. 分析用户问题，理解其研究意图
2. 生成 2-4 个 GitHub 搜索词（覆盖不同角度：官方实现/高质量项目/最新活跃）
3. 生成 2-4 个 HuggingFace 模型搜索词（任务类型/应用场景）
4. 生成 1-2 个 HuggingFace 数据集搜索词
5. 生成 1-3 个 arXiv 搜索词（精准主题/最新趋势）

## 搜索规则
- **最多生成 2 轮搜索指令**：第 1 轮生成全面覆盖的搜索词；收到结果后第 2 轮可细化/补充
- **第 2 轮之后必须给出最终回答**，不再生成新的搜索指令（不要再说"我继续搜索"）
- 如果搜索结果已足够回答问题，第 2 轮直接输出回答即可

## 搜索运算符参考
{search_tips}

## 输出格式（严格按此格式，每类资源一个代码块）
```
github transformers pytorch stars:>5000 pushed:>2024-01-01
```
```
hf models text-generation llm
```
```
hf datasets instruction-tuning
```
```
arxiv lora fine-tuning 2024
```

**直接输出搜索指令，不要在代码块之外写任何解释文字。**"""


# ── 主搜索器 ─────────────────────────────────────────────────

_MAX_TURNS = 3


class MultiSourceSearcher:
    """多源检索器 — 保留对话历史，多轮工具调用

    用法:
        searcher = MultiSourceSearcher(llm)
        messages = []  # 首次调用为空
        result, messages = searcher.run("LoRA fine-tuning 实现", messages=messages)
        # 下次调用传入同一 messages 即可保持上下文
        result2, messages = searcher.run("能用在视觉Transformer上吗", messages=messages)
    """

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self._tips = _load_search_tips()

    def run(
        self,
        question: str,
        messages: list[Message] | None = None,
        *,
        max_search_turns: int = 2,
        max_total_turns: int = 4,
    ) -> tuple[str, list[Message]]:
        """执行多源搜索

        参数:
            question: 用户当前问题
            messages: 对话历史（首次为空 list[]，后续传入同一列表保持记忆）
            max_search_turns: 最大搜索迭代轮数（默认2轮，之后强制进入回答模式）
            max_total_turns: 最大总迭代轮数

        返回:
            (最终回答文本, 更新后的 messages 列表)
            messages 可用于下一轮调用以保持上下文
        """
        if messages is None:
            messages = []

        # 构建首条 user 消息
        first_user = (
            f"用户问题：{question}\n\n"
            f"请为这个问题生成多源检索关键词。"
        )
        if not messages:
            messages.append(Message(role="user", content=first_user))
        else:
            messages.append(Message(role="user", content=question))

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(search_tips=self._tips)
        search_round = 0

        for turn in range(1, max_total_turns + 1):
            logger.info("MultiSourceSearcher 第 %d 轮 (search_round=%d)", turn, search_round)

            reply = self.llm.chat(
                external_prompt=system_prompt,
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )
            logger.debug("LLM 回复前300字: %s", reply[:300])
            messages.append(Message(role="assistant", content=reply))

            # 解析搜索指令
            queries = _parse_search_queries(reply)
            total_queries = sum(len(v) for v in queries.values())

            if total_queries == 0 or search_round >= max_search_turns:
                # 无搜索指令 或 达到搜索轮数上限 → 给出最终回答
                logger.info("第 %d 轮得到最终回答 (search_round=%d)", turn, search_round)
                return reply, messages

            search_round += 1
            logger.info("解析到 %d 条搜索指令 (search_round=%d): %s", total_queries, search_round, {
                k: v for k, v in queries.items() if v
            })

            # 执行所有搜索
            results_text = _execute_all_queries(queries)
            logger.info("搜索执行完成，共 %d 字符", len(results_text))

            # 注入结果，继续下一轮
            messages.append(
                Message(
                    role="user",
                    content=f"[搜索结果]\n{results_text}\n\n" + (
                        "请基于以上结果给出最终回答。" if search_round >= max_search_turns
                        else "请基于以上结果给出最终回答，或者如果信息不足可以补充搜索（最多再搜索1轮）。"
                    ),
                )
            )

        # 超过最大轮数
        logger.warning("达到最大轮数 %d，终止", max_total_turns)
        final_text = messages[-1].content if messages else ""
        return (
            "（已达到最大迭代次数，以下是已收集的结果）\n\n" + final_text,
            messages,
        )
