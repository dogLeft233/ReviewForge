"""Planner — LLM 驱动的检索规划、覆盖评估与洞察提取

职责边界：
- 不直接调用检索器（由 Manager 协调）
- 只负责「思考和分析」—— 想查什么、查得怎么样、有什么洞察
- 输出结构化的 RetrievalPlan / CoverageEvaluation / SynthesisResult

典型调用流程：
  1. planner.plan_retrieval("大数据处理技术综述")
     → 返回 RetrievalPlan（含多组子查询和检索策略）

  2. manager.execute_plan(plan)  # 调用所有检索器
     → 返回 CurationReport

  3. planner.evaluate_coverage(topic, report)
     → 返回 CoverageEvaluation（含补搜建议）

  4. (可选) planner.synthesize_insights(topic, report)
     → 返回 SynthesisResult（含写作建议）
"""

import logging
from typing import Any

from src.planner.llm import LLMClient
from src.planner.prompts import get_prompt
from src.planner.schemas import (
    CoverageEvaluation,
    RetrievalPlan,
    SynthesisResult,
)
from src.planner.searcher import SearchContext, WebSearcher

logger = logging.getLogger(__name__)


class Planner:
    """检索规划与评估器"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        web_searcher: WebSearcher | None = None,
    ) -> None:
        self.llm = llm_client or LLMClient()
        self.searcher = web_searcher or WebSearcher()

    # ── 联网预搜索 ──

    def search_web(
        self,
        topic: str,
        additional_queries: list[str] | None = None,
    ) -> SearchContext:
        """Step 0 (可选): 联网搜索主题，获取真实世界的研究现状

        在调用 plan_retrieval 之前先搜索 Web，
        让 LLM 基于实际内容而非训练数据生成子方向。

        Args:
            topic: 综述主题
            additional_queries: 附加搜索词，如 ["survey", "最新进展"]

        Returns:
            SearchContext 对象（可传入 plan_retrieval 的 web_context 参数）
        """
        return self.searcher.search_topic(
            topic=topic,
            additional_queries=additional_queries,
        )

    def analyze_web_context(
        self,
        topic: str,
        search_context: SearchContext,
    ) -> dict:
        """用 LLM 将搜索结果分析为结构化研究概览

        可选步骤，在 plan_retrieval 前对搜索结果做一次 LLM 分析。

        Args:
            topic: 综述主题
            search_context: search_web() 返回的结果

        Returns:
            结构化分析（方向、热点、新兴趋势等）
        """
        if search_context.is_empty():
            return {}

        prompt = get_prompt(
            "analyze_web_context",
            topic=topic,
            search_results=search_context.to_prompt_block(),
        )

        try:
            return self.llm.chat_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是研究趋势分析专家。"
                            "请严格按照用户要求的 JSON 格式输出，不要添加任何额外文字。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
            )
        except (RuntimeError, ValueError):
            logger.warning("Web context analysis failed, proceeding without it")
            return {}

    def plan_retrieval(
        self,
        topic: str,
        web_context: SearchContext | str | None = None,
        analyze_context: bool = False,
    ) -> RetrievalPlan:
        """Step 1: 分析主题，生成检索规划

        基于 LLM 对主题的理解，产出：
        - 主题分析（核心概念、子方向）
        - 多组搜索查询（中英文、分源）
        - 经典/前沿工作搜索策略
        - 需要的资源类型

        如果传入了 web_context（来自 search_web() 的结果），
        会自动使用 v2 prompt 将联网搜索结果注入为上下文，
        让 LLM 基于真实世界的研究现状来规划，减少幻觉。

        Args:
            topic: 综述主题，如 "大数据处理技术综述"
            web_context: 来自 search_web() 的搜索结果（可选）
            analyze_context: 如果为 True，先用 LLM 分析搜索结果为结构化概览

        Returns:
            RetrievalPlan 对象

        Raises:
            ValueError: LLM 返回无效 JSON 或缺少关键字段
            RuntimeError: LLM 调用失败（重试耗尽）
        """
        # 如果传了 web_context 但需要先分析，执行 LLM 分析
        if web_context and analyze_context and not isinstance(web_context, str):
            analysis = self.analyze_web_context(topic, web_context)
            if analysis:
                # 将分析结果格式化为 prompt 上下文
                lines = [f"研究方向: {', '.join(analysis.get('identified_directions', []))}"]
                if analysis.get("hot_directions"):
                    lines.append(f"热点: {', '.join(analysis['hot_directions'])}")
                if analysis.get("emerging_trends"):
                    lines.append(f"新兴趋势: {', '.join(analysis['emerging_trends'])}")
                if analysis.get("key_papers_projects"):
                    lines.append("关键资源:")
                    for kp in analysis["key_papers_projects"]:
                        lines.append(f"  - {kp.get('name')} ({kp.get('type')}): {kp.get('description')}")
                web_context_str = "\n".join(lines)
            else:
                web_context_str = web_context.to_prompt_block() if hasattr(web_context, 'to_prompt_block') else str(web_context)
        elif isinstance(web_context, SearchContext):
            web_context_str = web_context.to_prompt_block()
        elif isinstance(web_context, str):
            web_context_str = web_context
        else:
            web_context_str = None

        # 选择提示词版本
        if web_context_str:
            prompt = get_prompt("plan_retrieval", version="v2", topic=topic, web_context=web_context_str)
        else:
            prompt = get_prompt("plan_retrieval", topic=topic)

        try:
            raw = self.llm.chat_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是学术综述检索规划专家。"
                            "请严格按照用户要求的 JSON 格式输出，不要添加任何额外文字。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"LLM retrieval planning failed: {e}") from e

        if not raw or not isinstance(raw, dict):
            raise ValueError(
                f"LLM returned empty or non-dict response for topic {topic!r}"
            )

        plan = RetrievalPlan.from_dict(topic, raw)

        if not plan.queries:
            logger.warning(
                "LLM returned plan with no queries for topic %r: %s",
                topic,
                raw,
            )

        return plan

    def plan_with_search(
        self,
        topic: str,
        additional_queries: list[str] | None = None,
        analyze_context: bool = False,
        deep_fetch: bool = False,
        max_fetch_pages: int = 3,
    ) -> RetrievalPlan:
        """端到端：联网搜索 → (可选深度抓取) → (可选分析) → 检索规划

        一键完成预搜索和规划的全流程。
        相当于 search_web() + plan_retrieval(web_context=...)。

        Args:
            topic: 综述主题
            additional_queries: 附加搜索词
            analyze_context: 是否用 LLM 分析搜索结果
            deep_fetch: 是否深入抓取搜索结果页面（需要 httpx）
            max_fetch_pages: 最多抓取的页面数

        Returns:
            RetrievalPlan 对象
        """
        logger.info("plan_with_search: searching web for topic=%r", topic)

        if deep_fetch:
            logger.info("plan_with_search: deep fetch enabled (max_pages=%d)", max_fetch_pages)
            ctx = self.searcher.deep_search(topic, additional_queries, max_pages=max_fetch_pages)
        else:
            ctx = self.search_web(topic, additional_queries)

        if ctx.is_empty():
            logger.info("plan_with_search: no web results, falling back to knowledge-only plan")
            return self.plan_retrieval(topic)

        logger.info(
            "plan_with_search: got %d results (%d fetched)",
            len(ctx.results),
            len(ctx.deep_results),
        )

        # 如果有 deep_results，用 to_deep_prompt_block 提供更丰富的上下文
        if ctx.deep_results:
            web_context_str = ctx.to_deep_prompt_block()
            return self.plan_retrieval(topic=topic, web_context=web_context_str)
        else:
            return self.plan_retrieval(
                topic=topic,
                web_context=ctx,
                analyze_context=analyze_context,
            )

    def evaluate_coverage(
        self,
        topic: str,
        total_papers: int = 0,
        classic_count: int = 0,
        frontier_count: int = 0,
        github_count: int = 0,
        benchmark_count: int = 0,
        dataset_count: int = 0,
        categories: str = "",
        insights: str = "",
    ) -> CoverageEvaluation:
        """Step 2: 评估检索覆盖度，发现缺失方向

        基于已检索到的论文和资源的统计，判断覆盖度并提出补搜建议。

        Args:
            topic: 综述主题
            total_papers: 检索到的论文总数
            classic_count: 高引用"经典"论文数
            frontier_count: 2023 年后的前沿论文数
            github_count: GitHub 项目数
            benchmark_count: Benchmark 数
            dataset_count: 数据集数
            categories: 已涉及的方法分类列表（逗号分隔或文本描述）
            insights: 当前提取到的洞察

        Returns:
            CoverageEvaluation 对象
        """
        prompt = get_prompt(
            "evaluate_coverage",
            topic=topic,
            total_papers=str(total_papers),
            classic_count=str(classic_count),
            frontier_count=str(frontier_count),
            github_count=str(github_count),
            benchmark_count=str(benchmark_count),
            dataset_count=str(dataset_count),
            categories=categories or "暂无分类信息",
            insights=insights or "暂无洞察",
        )

        try:
            raw = self.llm.chat_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是学术综述覆盖度评估专家。"
                            "请严格按照用户要求的 JSON 格式输出，不要添加任何额外文字。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Coverage evaluation failed: {e}") from e

        evaluation = CoverageEvaluation.from_dict(raw if isinstance(raw, dict) else {})
        return evaluation

    def synthesize_insights(
        self,
        topic: str,
        papers_summary: str = "",
        resources_summary: str = "",
    ) -> SynthesisResult:
        """Step 3: 从检索结果中提取洞察

        分析论文之间的关系、方法演进、热点和空白，为写作提供建议。

        Args:
            topic: 综述主题
            papers_summary: 论文列表文本（每行：标题 | 年份 | 引用 | 分类 | 会议）
            resources_summary: 资源列表文本

        Returns:
            SynthesisResult 对象
        """
        prompt = get_prompt(
            "synthesize_insights",
            topic=topic,
            papers_summary=papers_summary or "暂无论文数据",
            resources_summary=resources_summary or "暂无资源数据",
        )

        try:
            raw = self.llm.chat_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是学术综述洞察提取专家。"
                            "请严格按照用户要求的 JSON 格式输出，不要添加任何额外文字。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2500,
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Insight synthesis failed: {e}") from e

        result = SynthesisResult.from_dict(raw if isinstance(raw, dict) else {})
        return result

    def full_plan(
        self,
        topic: str,
        papers_summary: str = "",
        resources_summary: str = "",
    ) -> dict[str, Any]:
        """端到端完整流程：规划 → (若有数据则)评估 → (若有数据则)洞察

        便捷方法，覆盖典型使用场景。
        更灵活的方式是分别调用 plan_retrieval / evaluate_coverage / synthesize_insights。

        Args:
            topic: 综述主题
            papers_summary: (可选) 已检索的论文摘要文本
            resources_summary: (可选) 已检索的资源摘要文本

        Returns:
            {"plan": RetrievalPlan, "evaluation": CoverageEvaluation|None,
             "synthesis": SynthesisResult|None}
        """
        logger.info("full_plan: planning retrieval for topic=%r", topic)
        plan = self.plan_retrieval(topic)

        evaluation: CoverageEvaluation | None = None
        synthesis: SynthesisResult | None = None

        if papers_summary:
            logger.info("full_plan: evaluating coverage")
            evaluation = self.evaluate_coverage(
                topic=topic,
                categories=", ".join(plan.estimated_coverage),
            )
            if evaluation.needs_supplement:
                logger.info(
                    "full_plan: coverage %s, %d supplemental queries suggested",
                    evaluation.overall_assessment,
                    len(evaluation.supplementary_queries),
                )

        if papers_summary or resources_summary:
            logger.info("full_plan: synthesizing insights")
            synthesis = self.synthesize_insights(
                topic=topic,
                papers_summary=papers_summary,
                resources_summary=resources_summary,
            )

        return {
            "plan": plan,
            "evaluation": evaluation,
            "synthesis": synthesis,
        }
