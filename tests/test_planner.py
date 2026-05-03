"""Planner 单元与集成测试"""

import json
import pytest
from dataclasses import fields

from src.planner import Planner, RetrievalPlan, CoverageEvaluation, LLMClient
from src.planner.prompts import get_prompt, PROMPT_REGISTRY
from src.planner.schemas import QuerySpec, SupplementaryQuery, SynthesisResult

# ═══════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════

SAMPLE_PLAN_JSON = {
    "topic_analysis": {
        "core_concepts": ["MapReduce", "分布式计算"],
        "sub_directions": ["批处理", "流处理"],
        "focus_level": "broad",
    },
    "queries": [
        {
            "query": "big data processing survey",
            "language": "en",
            "target_sources": ["arxiv", "semantic_scholar"],
            "rationale": "经典综述论文",
            "priority": 1,
        },
        {
            "query": "大数据处理 技术综述",
            "language": "zh",
            "target_sources": ["dblp", "arxiv"],
            "rationale": "中文文献",
            "priority": 2,
        },
    ],
    "classic_search_strategy": "搜索高引用经典论文",
    "frontier_search_strategy": "搜索2023年后论文",
    "resource_types_needed": ["github_repo", "dataset"],
    "estimated_coverage": ["批处理系统", "流处理系统"],
}

SAMPLE_EVAL_JSON = {
    "overall_assessment": "insufficient",
    "strengths": ["批处理覆盖较好"],
    "gaps": ["缺乏流处理相关论文", "缺少benchmark比较"],
    "missing_classics": ["MapReduce原始论文"],
    "missing_frontiers": ["2024年LLM数据处理"],
    "supplementary_queries": [
        {
            "query": "stream processing benchmark",
            "target_sources": ["arxiv", "github"],
            "rationale": "流处理基准测试",
        }
    ],
    "resource_gaps": ["benchmark代码"],
    "confidence": 0.75,
}

SAMPLE_SYNTHESIS_JSON = {
    "evolution_paths": [
        {
            "path_name": "批处理演进",
            "papers": ["MapReduce", "Spark", "Flink"],
            "description": "从Hadoop到Spark的演进",
        }
    ],
    "hot_topics": ["实时流处理", "AI集成"],
    "open_problems": ["数据倾斜", "资源调度"],
    "cross_references": [
        {"paper_a": "Spark", "paper_b": "MapReduce", "relationship": "builds_on"}
    ],
    "writing_recommendations": [
        {
            "section": "引言",
            "suggested_papers": ["MapReduce"],
            "key_message": "大数据处理的起源",
        }
    ],
}


# ═══════════════════════════════════════════════
# 基础功能测试（不依赖 API）
# ═══════════════════════════════════════════════

class TestPrompts:
    """提示词模板测试"""

    def test_get_known_prompt(self):
        """确保已知提示词可以正常获取"""
        for name in PROMPT_REGISTRY:
            kwargs = {"topic": "test"}
            if name == "evaluate_coverage":
                kwargs.update(
                    total_papers="10", classic_count="5", frontier_count="3",
                    github_count="2", benchmark_count="1", dataset_count="0",
                    categories="分类A", insights="洞察1",
                )
            elif name == "synthesize_insights":
                kwargs.update(papers_summary="论文列表", resources_summary="资源列表")
            for version in PROMPT_REGISTRY[name]:
                prompt = get_prompt(name, version, **kwargs)
                assert "test" in prompt, f"{name}/{version} 未正确格式化"
                assert len(prompt) > 50, f"{name}/{version} 内容过短"

    def test_get_prompt_with_multiple_params(self):
        """多参数格式化"""
        prompt = get_prompt(
            "evaluate_coverage",
            topic="test",
            total_papers="10",
            classic_count="5",
            frontier_count="3",
            github_count="2",
            benchmark_count="1",
            dataset_count="0",
            categories="批处理",
            insights="无",
        )
        assert "test" in prompt
        assert "10" in prompt

    def test_unknown_prompt_name(self):
        """未知提示词名称应抛出 KeyError"""
        with pytest.raises(KeyError):
            get_prompt("nonexistent_prompt")

    def test_unknown_prompt_version(self):
        """未知版本应抛出 KeyError"""
        with pytest.raises(KeyError):
            get_prompt("plan_retrieval", version="v99", topic="test")

    def test_missing_format_param(self):
        """缺少格式化参数应抛出 ValueError"""
        with pytest.raises(ValueError):
            get_prompt("plan_retrieval")  # 缺少 topic


class TestSchemas:
    """数据模型测试"""

    def test_retrieval_plan_from_dict(self):
        """RetrievalPlan 可以从字典正确构造"""
        plan = RetrievalPlan.from_dict("大数据处理", SAMPLE_PLAN_JSON)
        assert plan.topic == "大数据处理"
        assert len(plan.queries) == 2
        assert plan.queries[0].language == "en"
        assert plan.queries[1].language == "zh"
        assert "arxiv" in plan.queries[0].target_sources
        assert plan.estimated_coverage == ["批处理系统", "流处理系统"]

    def test_retrieval_plan_empty_queries(self):
        """空查询列表应被正确处理"""
        data = {
            "topic_analysis": {"core_concepts": [], "sub_directions": [], "focus_level": "focused"},
            "queries": [],
            "classic_search_strategy": "",
            "frontier_search_strategy": "",
            "resource_types_needed": [],
            "estimated_coverage": [],
        }
        plan = RetrievalPlan.from_dict("test", data)
        assert plan.queries == []

    def test_retrieval_plan_missing_fields(self):
        """缺失字段应使用默认值"""
        plan = RetrievalPlan.from_dict("test", {})
        assert plan.queries == []
        assert plan.topic_analysis == {}
        assert plan.resource_types_needed == []

    def test_coverage_evaluation_from_dict(self):
        """CoverageEvaluation 可以从字典正确构造"""
        eval_ = CoverageEvaluation.from_dict(SAMPLE_EVAL_JSON)
        assert eval_.overall_assessment == "insufficient"
        assert len(eval_.supplementary_queries) == 1
        assert eval_.needs_supplement is True
        assert eval_.confidence == 0.75

    def test_coverage_evaluation_adequate(self):
        """覆盖度足够时不需要补搜"""
        data = SAMPLE_EVAL_JSON.copy()
        data["overall_assessment"] = "adequate"
        data["supplementary_queries"] = []
        eval_ = CoverageEvaluation.from_dict(data)
        assert eval_.needs_supplement is False

    def test_coverage_evaluation_empty(self):
        """空字典应使用默认值"""
        eval_ = CoverageEvaluation.from_dict({})
        assert eval_.overall_assessment == "insufficient"
        assert not eval_.supplementary_queries
        assert eval_.confidence == 0.0

    def test_synthesis_from_dict(self):
        """SynthesisResult 可以从字典正确构造"""
        syn = SynthesisResult.from_dict(SAMPLE_SYNTHESIS_JSON)
        assert len(syn.evolution_paths) == 1
        assert syn.evolution_paths[0].path_name == "批处理演进"
        assert len(syn.writing_recommendations) == 1
        assert syn.writing_recommendations[0].section == "引言"

    def test_query_spec_defaults(self):
        """QuerySpec 默认值"""
        q = QuerySpec(query="test")
        assert q.language == "en"
        assert q.priority == 1
        assert q.target_sources == []

    def test_supplementary_query_defaults(self):
        """SupplementaryQuery 默认值"""
        q = SupplementaryQuery(query="test")
        assert q.target_sources == []


class TestLLMClient:
    """LLM 客户端测试"""

    def test_init_no_api_key(self, monkeypatch):
        """无 API key 应抛出 ValueError"""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            LLMClient()

    def test_init_with_env_key(self, monkeypatch):
        """环境变量中的 API key 应被正确读取（LLMClient 直接检查 os.environ）"""
        monkeypatch.setenv("LLM_API_KEY", "test-key-123")
        client = LLMClient()
        assert client.api_key == "test-key-123"
        monkeypatch.delenv("LLM_API_KEY", raising=False)

    def test_init_with_direct_key(self):
        """直接传入的 API key 应优先"""
        client = LLMClient(api_key="direct-key", base_url="http://localhost:8080/v1")
        assert client.api_key == "direct-key"

    def test_extract_json_direct(self):
        """直接解析整个 JSON 字符串"""
        result = LLMClient._extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_extract_json_embedded(self):
        """从文字中提取 JSON 块"""
        content = '以下是结果：\n{"key": "value"}\n以上'
        result = LLMClient._extract_json(content)
        assert result == {"key": "value"}

    def test_extract_json_nested(self):
        """嵌套 JSON 应正确提取"""
        content = '结果如下：\n{"outer": {"inner": [1, 2, 3]} }\n结束'
        result = LLMClient._extract_json(content)
        assert "outer" in result
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_extract_json_empty(self):
        """空字符串应抛出 ValueError"""
        with pytest.raises(ValueError):
            LLMClient._extract_json("")

    def test_extract_json_invalid(self):
        """无法提取 JSON 时应抛出 ValueError"""
        with pytest.raises(ValueError):
            LLMClient._extract_json("这不是 JSON，只是一段普通文本")


# ═══════════════════════════════════════════════
# 集成测试（需要实际 LLM API）
# ═══════════════════════════════════════════════

@pytest.mark.integration
class TestPlannerIntegration:
    """Planner 集成测试——需要 LLM_API_KEY 环境变量"""

    def _has_api_key(self) -> bool:
        import os
        return bool(os.environ.get("LLM_API_KEY"))

    @pytest.fixture
    def planner(self):
        return Planner()

    def _run_test_or_skip(self, test_fn, *args, **kwargs):
        import os
        if not os.environ.get("LLM_API_KEY"):
            pytest.skip("需要 LLM_API_KEY 环境变量")
        return test_fn(*args, **kwargs)

    def test_planner_connectivity(self, planner):
        """验证 LLM API 连通性"""
        if not self._has_api_key():
            pytest.skip("需要 LLM_API_KEY 环境变量")

        reply = planner.llm.chat(
            messages=[{"role": "user", "content": "回复OK"}],
            max_tokens=20,
        )
        assert reply, "LLM 返回了空回复"
        assert len(reply) > 0

    def test_plan_retrieval_simple(self, planner):
        """生成简单主题的检索规划"""
        if not self._has_api_key():
            pytest.skip("需要 LLM_API_KEY 环境变量")

        plan = planner.plan_retrieval("大数据处理技术综述")

        # 验证规划基本结构
        assert plan.topic == "大数据处理技术综述"
        assert len(plan.queries) > 0, "规划应包含至少一个查询"

        # 验证每个查询的完整性
        for i, q in enumerate(plan.queries):
            assert q.query, f"查询 {i} 缺少关键词"
            assert q.target_sources, f"查询 {i} 缺少目标源"
            assert q.language in ("en", "zh"), f"查询 {i} 语言字段异常"

        # 验证主题分析
        assert plan.topic_analysis.get("core_concepts"), "应包含核心概念分析"
        assert plan.topic_analysis.get("sub_directions"), "应包含子方向分析"

    def test_plan_retrieval_english(self, planner):
        """英文主题"""
        if not self._has_api_key():
            pytest.skip("需要 LLM_API_KEY 环境变量")

        plan = planner.plan_retrieval("Distributed Machine Learning Systems")
        assert len(plan.queries) > 0
        # 至少有一个英文查询
        en_queries = [q for q in plan.queries if q.language == "en"]
        assert en_queries, "英文主题应生成英文查询"

    def test_plan_retrieval_multiple_sources(self, planner):
        """验证查询分配到多个源"""
        if not self._has_api_key():
            pytest.skip("需要 LLM_API_KEY 环境变量")

        plan = planner.plan_retrieval("Real-time Stream Processing")
        all_sources = set()
        for q in plan.queries:
            all_sources.update(q.target_sources)
        assert len(all_sources) >= 2, (
            f"查询应覆盖至少 2 个检索源，当前: {all_sources}"
        )

    def test_evaluate_coverage(self, planner):
        """覆盖度评估"""
        if not self._has_api_key():
            pytest.skip("需要 LLM_API_KEY 环境变量")

        eval_ = planner.evaluate_coverage(
            topic="大数据处理",
            total_papers=15,
            classic_count=3,
            frontier_count=5,
            github_count=4,
            benchmark_count=1,
            dataset_count=2,
            categories="批处理,流处理,内存计算",
            insights="MapReduce到Spark演进清晰",
        )
        assert eval_.overall_assessment in ("adequate", "insufficient", "critical_gaps")
        assert isinstance(eval_.confidence, (int, float))
        assert 0 <= eval_.confidence <= 1

    def test_synthesize_insights(self, planner):
        """洞察提取"""
        if not self._has_api_key():
            pytest.skip("需要 LLM_API_KEY 环境变量")

        result = planner.synthesize_insights(
            topic="大数据处理",
            papers_summary="MapReduce | 2004 | 20000+ | 批处理 | OSDI\n"
                          "Spark | 2012 | 10000+ | 批处理/内存计算 | NSDI\n"
                          "Flink | 2015 | 5000+ | 流处理 | VLDB",
            resources_summary="Apache Spark | github | 分布式计算框架\n"
                            "Apache Flink | github | 流处理框架",
        )
        assert len(result.evolution_paths) > 0 or len(result.hot_topics) > 0

    def test_full_plan_without_papers(self, planner):
        """端到端流程——无论文数据时只做规划"""
        if not self._has_api_key():
            pytest.skip("需要 LLM_API_KEY 环境变量")

        result = planner.full_plan("大数据处理")
        assert "plan" in result
        assert result["plan"].topic == "大数据处理"
        assert len(result["plan"].queries) > 0
        # 没有论文数据，不应触发评估和洞察
        assert result["evaluation"] is None
        assert result["synthesis"] is None


# ═══════════════════════════════════════════════
# 端到端测试（完整流程，需要 API）
# ═══════════════════════════════════════════════

@pytest.mark.e2e
class TestPlannerE2E:
    """端到端测试——模拟真实使用场景"""

    def test_realistic_workflow(self):
        """模拟真实工作流：规划 → 人工检索 → 评估 → 洞察"""
        import os
        if not os.environ.get("LLM_API_KEY"):
            pytest.skip("需要 LLM_API_KEY 环境变量")

        planner = Planner()

        # Step 1: Plan
        plan = planner.plan_retrieval("分布式机器学习训练框架")
        assert len(plan.queries) >= 2

        # 输出规划概览
        print(f"\n[规划结果] 主题: {plan.topic}")
        print(f"  核心概念: {plan.topic_analysis.get('core_concepts', [])}")
        print(f"  子方向: {plan.topic_analysis.get('sub_directions', [])}")
        print(f"  查询数: {len(plan.queries)}")
        for q in plan.queries:
            print(f"    [{q.priority}] {q.query} ({q.language}) → {q.target_sources}")
        print(f"  经典策略: {plan.classic_search_strategy}")
        print(f"  前沿策略: {plan.frontier_search_strategy}")
        print(f"  所需资源: {plan.resource_types_needed}")

        # Step 2: Evaluate (simulated retrieval results)
        eval_ = planner.evaluate_coverage(
            topic=plan.topic,
            total_papers=20,
            classic_count=5,
            frontier_count=8,
            github_count=6,
            benchmark_count=2,
            dataset_count=3,
            categories=", ".join(plan.estimated_coverage),
            insights="参数服务器和AllReduce架构是主流",
        )
        print(f"\n[评估结果] 覆盖度: {eval_.overall_assessment}")
        print(f"  置信度: {eval_.confidence}")
        print(f"  优势: {eval_.strengths}")
        print(f"  缺口: {eval_.gaps}")
        print(f"  补搜建议: {len(eval_.supplementary_queries)} 个")
        if eval_.supplementary_queries:
            for sq in eval_.supplementary_queries:
                print(f"    {sq.query} → {sq.target_sources}: {sq.rationale}")

        # Step 3: Synthesize
        syn = planner.synthesize_insights(
            topic=plan.topic,
            papers_summary=(
                "TensorFlow | 2015 | 15000+ | 深度学习框架 | OSDI\n"
                "PyTorch | 2016 | 12000+ | 深度学习框架 | NeurIPS\n"
                "Horovod | 2017 | 3000+ | 分布式训练 | ATC\n"
                "DeepSpeed | 2020 | 2000+ | 分布式优化 | MLSys"
            ),
            resources_summary=(
                "Horovod | github | 分布式训练框架\n"
                "DeepSpeed | github | 训练优化库\n"
                "MLPerf | benchmark | 机器学习性能基准"
            ),
        )
        print(f"\n[洞察结果]")
        print(f"  演进路径: {[ep.path_name for ep in syn.evolution_paths]}")
        print(f"  热点: {syn.hot_topics}")
        print(f"  开放问题: {syn.open_problems}")
        print(f"  写作建议: {len(syn.writing_recommendations)} 条")
        for wr in syn.writing_recommendations:
            print(f"    [{wr.section}] {wr.key_message}")

        # 最终断言
        assert len(syn.evolution_paths) > 0 or len(syn.hot_topics) > 0
