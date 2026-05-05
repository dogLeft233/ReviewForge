"""Planner 单元与集成测试"""

import json
import pytest
from dataclasses import fields

from src.planner import Planner, RetrievalPlan, CoverageEvaluation, LLMClient
from src.planner import searcher
from src.planner.prompts import get_prompt, PROMPT_REGISTRY
from src.planner.schemas import (
    QuerySpec, SupplementaryQuery, SynthesisResult,
    Outline, OutlineSection,
)

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
            elif name == "generate_outline":
                kwargs.update(
                    papers_summary="论文列表", resources_summary="资源列表",
                )
            elif name == "analyze_web_context":
                kwargs.update(search_results="搜索结果摘要")
            for version in PROMPT_REGISTRY[name]:
                extra = {}
                if name == "plan_retrieval" and version == "v2":
                    extra = {"web_context": "搜索上下文"}
                prompt = get_prompt(name, version, **(kwargs | extra))
                assert "test" in prompt, f"{name}/{version} 未正确格式化"
                assert len(prompt) > 50, f"{name}/{version} 内容过短"
                if name == "plan_retrieval" and version == "v2":
                    assert "搜索上下文" in prompt, f"v2 未包含 web_context 内容"

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

    # ── Outline 模型 ──

    def test_outline_section_from_dict(self):
        """OutlineSection 可以从字典正确构造"""
        data = {
            "id": "2.1",
            "title": "批处理框架演进",
            "level": 2,
            "description": "从Hadoop MapReduce到Spark的演进脉络",
            "evidence_required": ["MapReduce原始论文", "Spark论文"],
            "coverage_status": "partial",
            "coverage_rationale": "找到了Spark论文但缺少MapReduce",
            "supplementary_queries": ["MapReduce原始论文"],
            "child_sections": [
                {
                    "id": "2.1.1",
                    "title": "Hadoop MapReduce",
                    "level": 3,
                    "description": "经典批处理模型",
                    "evidence_required": [],
                    "coverage_status": "insufficient",
                    "coverage_rationale": "未检索到",
                    "supplementary_queries": ["MapReduce Dean 2004"],
                    "child_sections": [],
                }
            ],
        }
        s = OutlineSection.from_dict(data)
        assert s.id == "2.1"
        assert s.title == "批处理框架演进"
        assert s.level == 2
        assert s.coverage_status == "partial"
        assert s.supplementary_queries == ["MapReduce原始论文"]
        assert len(s.child_sections) == 1
        assert s.child_sections[0].id == "2.1.1"
        assert s.child_sections[0].coverage_status == "insufficient"

    def test_outline_section_defaults(self):
        """OutlineSection 缺失字段应使用默认值"""
        s = OutlineSection.from_dict({})
        assert s.id == ""
        assert s.level == 1
        assert s.coverage_status == "unknown"
        assert s.child_sections == []
        assert s.supplementary_queries == []

    def test_outline_section_to_text(self):
        """OutlineSection.to_text() 输出格式"""
        s = OutlineSection(
            id="1", title="引言", level=1,
            description="研究背景", coverage_status="sufficient",
        )
        text = s.to_text()
        assert "✅" in text
        assert "1" in text
        assert "引言" in text
        assert "研究背景" in text

    def test_outline_section_to_text_with_children(self):
        """带子章节的 to_text"""
        child = OutlineSection(
            id="1.1", title="研究背景", level=2,
            description="大数据时代背景", coverage_status="partial",
            supplementary_queries=["大数据发展报告"],
        )
        parent = OutlineSection(
            id="1", title="引言", level=1,
            coverage_status="partial", child_sections=[child],
        )
        text = parent.to_text()
        assert "⚠️" in text
        assert "1.1" in text
        assert "补搜" in text

    def test_outline_from_dict(self):
        """Outline 可以从字典正确构造"""
        data = {
            "abstract": "综述大数据处理技术的发展脉络",
            "overall_coverage": 0.65,
            "gap_summary": "缺少流处理相关论文",
            "supplementary_queries": ["流处理 benchmark"],
            "sections": [
                {
                    "id": "1",
                    "title": "引言",
                    "level": 1,
                    "description": "研究背景",
                    "evidence_required": [],
                    "coverage_status": "sufficient",
                    "coverage_rationale": "",
                    "supplementary_queries": [],
                    "child_sections": [],
                }
            ],
        }
        o = Outline.from_dict("大数据处理技术综述", data)
        assert o.topic == "大数据处理技术综述"
        assert o.abstract
        assert o.overall_coverage == 0.65
        assert o.gap_summary
        assert len(o.supplementary_queries) == 1
        assert len(o.sections) == 1
        assert o.sections[0].title == "引言"

    def test_outline_empty(self):
        """空字典应使用默认值"""
        o = Outline.from_dict("test", {})
        assert o.topic == "test"
        assert o.sections == []
        assert o.overall_coverage == 0.0

    def test_outline_needs_supplement_true(self):
        """有章节不足时应触发补搜"""
        o = Outline(
            topic="test", sections=[
                OutlineSection(id="1", title="A", coverage_status="sufficient"),
                OutlineSection(id="2", title="B", coverage_status="insufficient"),
            ],
        )
        assert o.needs_supplement is True

    def test_outline_needs_supplement_false(self):
        """所有章节充足时不应触发补搜"""
        o = Outline(
            topic="test", sections=[
                OutlineSection(id="1", title="A", coverage_status="sufficient"),
                OutlineSection(id="2", title="B", coverage_status="sufficient"),
            ],
        )
        assert o.needs_supplement is False

    def test_outline_to_full_text(self):
        """to_full_text 应产出结构化文本"""
        o = Outline(
            topic="大数据处理",
            abstract="本文综述大数据处理技术",
            overall_coverage=0.7,
            gap_summary="缺少流处理内容",
            sections=[
                OutlineSection(id="1", title="引言", coverage_status="sufficient"),
                OutlineSection(id="2", title="核心技术", coverage_status="partial"),
            ],
        )
        text = o.to_full_text()
        assert "大数据处理" in text
        assert "70%" in text or "0.7" in text
        assert "缺漏" in text
        assert "✅" in text
        assert "⚠️" in text

    # ── Outline 缺口查询提取 ──

    def test_collect_gap_queries_empty(self):
        """无缺口时返回空列表"""
        o = Outline(topic="test", sections=[
            OutlineSection(id="1", title="A", coverage_status="sufficient"),
        ])
        queries, section_map = o.collect_gap_queries()
        assert queries == []
        assert section_map == {}

    def test_collect_gap_queries_global_only(self):
        """仅全局缺口"""
        o = Outline(
            topic="test",
            supplementary_queries=["big data survey"],
        )
        queries, section_map = o.collect_gap_queries()
        assert queries == ["big data survey"]
        assert section_map == {}

    def test_collect_gap_queries_from_sections(self):
        """从各章节提取缺口"""
        o = Outline(topic="test", sections=[
            OutlineSection(
                id="2", title="B", coverage_status="partial",
                supplementary_queries=["cloud storage"],
                child_sections=[
                    OutlineSection(
                        id="2.2", title="B2", coverage_status="insufficient",
                        supplementary_queries=["MinIO architecture"],
                    ),
                ],
            ),
            OutlineSection(id="3", title="C", coverage_status="sufficient"),
        ])
        queries, section_map = o.collect_gap_queries()
        assert "cloud storage" in queries
        assert "MinIO architecture" in queries
        assert section_map["2"] == ["cloud storage"]
        assert section_map["2.2"] == ["MinIO architecture"]
        assert "3" not in section_map  # 充足章节无查询

    def test_collect_gap_queries_dedup(self):
        """重复查询只出现一次"""
        o = Outline(
            topic="test",
            supplementary_queries=["data lakehouse"],
            sections=[
                OutlineSection(
                    id="5", title="E", coverage_status="partial",
                    supplementary_queries=["data lakehouse"],
                ),
            ],
        )
        queries, section_map = o.collect_gap_queries()
        assert queries == ["data lakehouse"]  # 去重
        assert "5" in section_map  # 仍记录章节映射


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
        import os
        if not os.environ.get("LLM_API_KEY"):
            pytest.skip("需要 LLM_API_KEY 环境变量")
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

    def test_generate_outline(self, planner):
        """生成细纲 + 覆盖评估"""
        if not self._has_api_key():
            pytest.skip("需要 LLM_API_KEY 环境变量")

        # 模拟有论文数据的情况
        outline = planner.generate_outline(
            topic="大数据处理技术综述",
            papers_summary=(
                "MapReduce | 2004 | 20000+ | 批处理 | OSDI\n"
                "Spark | 2012 | 10000+ | 批处理/内存计算 | NSDI\n"
                "Flink | 2015 | 5000+ | 流处理 | VLDB\n"
                "TensorFlow | 2015 | 15000+ | 深度学习框架 | OSDI\n"
                "DeepSpeed | 2020 | 2000+ | 分布式优化 | MLSys"
            ),
            resources_summary=(
                "Apache Spark | github | 分布式计算框架\n"
                "Apache Flink | github | 流处理框架\n"
                "Horovod | github | 分布式训练框架\n"
                "MLPerf | benchmark | 机器学习性能基准"
            ),
        )

        # 验证细纲基本结构
        assert outline.topic == "大数据处理技术综述"
        assert len(outline.sections) > 0, "细纲应包含至少一个章节"

        # 验证各章节结构
        for i, section in enumerate(outline.sections):
            assert section.id, f"章节 {i} 缺少 id"
            assert section.title, f"章节 {i} 缺少 title"
            assert section.level == 1, f"一级章节 {i} level 应为 1"
            assert section.coverage_status in (
                "sufficient", "partial", "insufficient", "unknown"
            ), f"章节 {i} 覆盖状态异常: {section.coverage_status}"

        # 验证覆盖度指标
        assert 0 <= outline.overall_coverage <= 1

        # 输出细纲预览
        print(f"\n{'='*60}")
        print(f"[细纲预览] 主题: {outline.topic}")
        print(f"  摘要: {outline.abstract}")
        print(f"  整体覆盖度: {outline.overall_coverage:.0%}")
        if outline.gap_summary:
            print(f"  缺漏概况: {outline.gap_summary}")
        print(f"  章节数: {len(outline.sections)}")
        for s in outline.sections:
            print(f"    [{s.coverage_status}] {s.id} {s.title}")
            print(f"      └ {s.description[:80] if s.description else '(无描述)'}")
            if s.child_sections:
                for c in s.child_sections:
                    print(f"      ├ [{c.coverage_status}] {c.id} {c.title}")
        if outline.supplementary_queries:
            print(f"  补搜建议 ({len(outline.supplementary_queries)} 条):")
            for q in outline.supplementary_queries:
                print(f"    - {q}")

    def test_generate_outline_empty(self, planner):
        """无论文数据时也要能生成细纲"""
        if not self._has_api_key():
            pytest.skip("需要 LLM_API_KEY 环境变量")

        outline = planner.generate_outline(
            topic="大数据处理技术综述",
        )

        assert outline.topic == "大数据处理技术综述"
        assert len(outline.sections) > 0
        # 无论文数据，各节状态至少不是空的
        for s in outline.sections:
            assert s.coverage_status in (
                "sufficient", "partial", "insufficient", "unknown"
            )

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


# ═══════════════════════════════════════════════
# WebSearcher 单元测试
# ═══════════════════════════════════════════════


class TestSearchResult:
    """SearchResult 数据类测试"""

    def test_minimal(self):
        r = searcher.SearchResult()
        assert r.title == ""
        assert r.description == ""

    def test_from_dict_full(self):
        r = searcher.SearchResult.from_dict({
            "title": "Big Data Survey 2024",
            "url": "https://example.com/paper",
            "description": "A comprehensive survey",
            "summary": "Detailed summary here",
            "siteName": "arXiv",
            "publishedDate": "2024-01-15",
        })
        assert r.title == "Big Data Survey 2024"
        assert r.summary == "Detailed summary here"
        assert r.site_name == "arXiv"
        assert r.published_date == "2024-01-15"

    def test_from_dict_summary_fallback(self):
        r = searcher.SearchResult.from_dict({
            "title": "ML Trends",
            "description": "Falls back to this",
        })
        assert r.summary == "Falls back to this"


class TestSearchContext:
    """SearchContext 数据类测试"""

    def test_empty(self):
        ctx = searcher.SearchContext(topic="test")
        assert ctx.is_empty()
        assert ctx.to_prompt_block() == "(无搜索结果)"

    def test_with_results(self):
        results = [
            searcher.SearchResult(title="Paper 1", description="Desc 1", site_name="arXiv"),
            searcher.SearchResult(title="Paper 2", description="Desc 2", published_date="2024-03-01"),
        ]
        ctx = searcher.SearchContext(topic="ML", results=results, total_estimated=10)
        assert not ctx.is_empty()
        block = ctx.to_prompt_block()
        assert "Paper 1" in block
        assert "Paper 2" in block
        assert "arXiv" in block
        assert "2024-03-01" in block
        assert "ML" in block


class TestWebSearcher:
    """WebSearcher 类测试"""

    def test_script_not_found(self):
        ws = searcher.WebSearcher(script_path="/nonexistent/path/search.js")
        ctx = ws.search("test")
        assert ctx.is_empty()
        assert ctx.topic == "test"

    def test_search_topic_with_additional_empty_fallback(self):
        ws = searcher.WebSearcher(script_path="/nonexistent/path/search.js")
        ctx = ws.search_topic("test", additional_queries=["survey"])
        assert ctx.is_empty()
        assert ctx.topic == "test"

    def test_to_prompt_block_empty(self):
        ws = searcher.WebSearcher(script_path="/nonexistent/path/search.js")
        ctx = ws.search("anything")
        assert ctx.to_prompt_block() == "(无搜索结果)"

    def test_deep_search_fallback_on_no_script(self):
        ws = searcher.WebSearcher(script_path="/nonexistent/path/search.js")
        ctx = ws.deep_search("test", max_pages=2)
        assert ctx.is_empty()
        assert len(ctx.deep_results) == 0


# ═══════════════════════════════════════════════
# WebPageFetcher 单元测试
# ═══════════════════════════════════════════════

from src.planner.fetcher import FetchedPage, WebPageFetcher, _TextExtractor


class TestFetchedPage:
    """FetchedPage 数据类测试"""

    def test_success_property(self):
        p = FetchedPage(url="https://example.com", status_code=200)
        assert p.success

    def test_failure_property(self):
        p = FetchedPage(url="https://example.com", status_code=0, error="err")
        assert not p.success

    def test_error_result(self):
        p = FetchedPage.error_result("https://x.com", "连接失败")
        assert p.url == "https://x.com"
        assert p.error == "连接失败"
        assert p.status_code == 0

    def test_summary_error(self):
        p = FetchedPage.error_result("https://x.com", "超时")
        assert "抓取失败" in p.summary()
        assert "超时" in p.summary()

    def test_summary_success(self):
        p = FetchedPage(url="https://x.com", title="Test", text="Hello world", status_code=200)
        s = p.summary(max_chars=50)
        assert "Test" in s
        assert "Hello" in s


class TestTextExtractor:
    """HTML 文本提取测试"""

    def test_simple_text(self):
        e = _TextExtractor()
        e.feed("<p>Hello World</p>")
        assert e.get_text() == "Hello World"

    def test_skip_script(self):
        e = _TextExtractor()
        e.feed("<p>Hello</p><script>x=1</script><p>World</p>")
        assert "Hello World" == e.get_text()

    def test_block_spacing(self):
        e = _TextExtractor()
        e.feed("<p>A</p><p>B</p>")
        assert "A B" in e.get_text() or "A\n\nB" in e.get_text()


class TestWebPageFetcher:
    """WebPageFetcher 单元测试"""

    def test_bad_url(self):
        f = WebPageFetcher(timeout=3)
        p = f.fetch("https://nonexistent-domain-12345.test/")
        assert not p.success
        assert p.error

    def test_fetch_results_empty(self):
        f = WebPageFetcher(timeout=3)
        pages = f.fetch_results([])
        assert pages == []

    def test_fetch_results_max(self):
        f = WebPageFetcher(timeout=3)
        pages = f.fetch_results(["a", "b", "c"], max_pages=1)
        # 最多尝试 1 个 URL，但会失败
        assert len(pages) == 0
