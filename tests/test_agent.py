#!/usr/bin/env python3
"""ChatAgent 测试程序

用法:
    cd /mnt/e/Documents/github_clone/ReviewForge
    python -m tests.test_agent

验证流程：
    1. 导入 ChatAgent 和 VisualizationData
    2. 用测试问题调用 agent
    3. 打印生成结果（流式输出）
    4. 人工验证回答质量
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent import ChatAgent
from src.visualizer.schema import VisualizationData, Overview, Method, Paper


def _create_sample_data() -> VisualizationData:
    """创建示例 VisualizationData"""
    return VisualizationData(
        topic="Large Language Model Fine-tuning",
        overview=Overview(
            definition="大语言模型微调是指在预训练模型基础上，通过特定任务数据进一步训练以适应下游任务的技术。",
            core_questions=[
                "如何高效微调大模型？",
                "LoRA 与全量微调的区别是什么？",
                "哪些任务最适合微调？",
            ],
            key_concepts=["LoRA", "QLoRA", "Adapter", "Prefix Tuning"],
        ),
        methods=[
            Method(id="m1", name="LoRA", category="Parameter-Efficient Fine-Tuning",
                   description="低秩适配器，通过学习低秩矩阵减少可训练参数"),
            Method(id="m2", name="QLoRA", category="Quantized Fine-Tuning",
                   description="量化+LoRA，可在单卡微调 65B 模型"),
        ],
        papers=[
            Paper(id="p1", title="LoRA: Low-Rank Adaptation of Large Language Models",
                  year=2021, authors="Hu et al.", venue="ICLR 2022"),
        ],
    )


def main():
    print("=" * 60)
    print("ChatAgent 测试")
    print("=" * 60)

    print("\n[1] 初始化 ChatAgent...")
    try:
        agent = ChatAgent(session_id="test_session")
        print("    ✓ ChatAgent 初始化成功")
    except Exception as e:
        print(f"    ❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[2] 准备示例数据...")
    data = _create_sample_data()
    print(f"    Topic: {data.topic}")
    print("    ✓ 数据准备完成")

    TEST_QUESTIONS = [
        "解释 LoRA 的原理",
        "LoRA 与全量微调相比有什么优势？",
        "QLoRA 是如何实现单卡微调大模型的？",
    ]

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'=' * 60}")
        print(f"测试 {i}/{len(TEST_QUESTIONS)}")
        print(f"{'=' * 60}")
        print(f"问题: {question}")
        print("-" * 60)

        try:
            print("\n流式输出:")
            print(">>> ", end="", flush=True)
            full_response = ""
            for chunk in agent.ask_iter(question, data):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()
            print(f"\n完整回答长度: {len(full_response)} 字符")
        except Exception as e:
            print(f"\n❌ 调用失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("测试完成")
    print("=" * 60)
    print("\n验证提示:")
    print("  1. Agent 是否正确调用了 web_search/web_fetch？")
    print("  2. Memory 是否保留了多轮对话历史？")
    print("  3. 流式输出是否正常工作？")


if __name__ == "__main__":
    main()
