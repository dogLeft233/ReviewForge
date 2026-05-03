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

logger = logging.getLogger(__name__)


class Planner:
    """检索规划与评估器"""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient()

    def plan_retrieval(self, topic: str) -> RetrievalPlan:
        """Step 1: 分析主题，生成检索规划

        基于 LLM 对主题的理解，产出：
        - 主题分析（核心概念、子方向）
        - 多组搜索查询（中英文、分源）
        - 经典/前沿工作搜索策略
        - 需要的资源类型

        Args:
            topic: 综述主题，如 "大数据处理技术综述"

        Returns:
            RetrievalPlan 对象

        Raises:
            ValueError: LLM 返回无效 JSON 或缺少关键字段
            RuntimeError: LLM 调用失败（重试耗尽）
        """
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
