"""提示词测试——验证动态示例无锚定偏差"""

from src.searcher.prompts import (
    _extract_example_term,
    _build_example_section,
    build_query_generation_prompt,
)
from src.searcher.models import DomainProfile


class TestExtractExampleTerm:
    """_extract_example_term 提取逻辑"""

    def test_no_profile_short_topic(self):
        term = _extract_example_term("ASR")
        assert term == "ASR"

    def test_no_profile_multi_word(self):
        term = _extract_example_term("Large Language Model")
        assert term == "Large Language Model"

    def test_no_profile_long_topic(self):
        term = _extract_example_term("automatic speech recognition transformer")
        assert term == "automatic speech"

    def test_profile_core_concept_takes_priority(self):
        profile = DomainProfile(topic="ASR", core_concepts=["automatic speech recognition"])
        term = _extract_example_term("ASR", domain_profile=profile)
        assert term == "automatic speech recognition"

    def test_profile_empty_core_concepts_falls_back(self):
        profile = DomainProfile(topic="ASR")
        term = _extract_example_term("ASR", domain_profile=profile)
        assert term == "ASR"


class TestBuildExampleSection:
    """_build_example_section 动态生成"""

    def test_asr_example_no_transformer(self):
        """验证 ASR 的示例 JSON 中不含 'transformer'"""
        section = _build_example_section("ASR")
        assert "transformer" not in section.lower(), (
            f"示例不应含 transformer, 实际={section[:200]}"
        )
        # 验证确实包含 topic 相关词
        assert "ASR" in section

    def test_rag_example_uses_rag(self):
        """验证 RAG 的示例 JSON 使用 RAG"""
        section = _build_example_section("RAG")
        assert "RAG" in section
        assert "transformer" not in section.lower()

    def test_profile_terms_used(self):
        """验证 domain_profile 核心概念被用作示例术语"""
        profile = DomainProfile(topic="ASR", core_concepts=["speech recognition"])
        section = _build_example_section("ASR", domain_profile=profile)
        assert "speech recognition" in section
        assert "ASR" not in section  # core_concepts[0] 优先于 topic

    def test_diff_topics_generate_different_examples(self):
        """不同 topic 生成不同示例"""
        a = _build_example_section("ASR")
        b = _build_example_section("RAG")
        assert a != b


class TestBuildQueryGenerationPrompt:
    """build_query_generation_prompt 整体验证"""

    @staticmethod
    def _extract_example_region(prompt: str) -> str:
        """提取 OUTPUT FORMAT 示例部分（"BASED ON" 到 "CONSTRAINTS" 之间）"""
        start = prompt.find("BASED ON THE ACTUAL TOPIC")
        end = prompt.find("CONSTRAINTS")
        if start >= 0 and end > start:
            return prompt[start:end]
        return prompt

    def test_asr_example_has_no_transformer(self):
        """验证 ASR 提示词的示例部分不含 transformer"""
        prompt = build_query_generation_prompt(
            topic="ASR", domain_profile_text="",
        )
        example = self._extract_example_region(prompt)
        assert "transformer" not in example.lower(), (
            f"示例区域不应含 transformer: ...{example[:200]}..."
        )
        assert "ASR" in example

    def test_prompt_includes_example_topic_info(self):
        """验证提示词说明示例基于实际 topic"""
        prompt = build_query_generation_prompt(
            topic="ASR", domain_profile_text="",
        )
        assert "BASED ON THE ACTUAL TOPIC" in prompt

    def test_diff_topics_no_anchor(self):
        """不同 topic 生成不同示例—不相互锚定"""
        a = build_query_generation_prompt(topic="ASR", domain_profile_text="")
        b = build_query_generation_prompt(topic="RAG", domain_profile_text="")
        assert a != b
        assert "ASR" in a
        assert "RAG" in b
        # 示例部分也不含对方 topic 词
        assert "RAG" not in self._extract_example_region(a)
        assert "ASR" not in self._extract_example_region(b)
