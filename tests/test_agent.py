#!/usr/bin/env python3
"""Searcher Agent 测试程序

用法:
    cd /mnt/e/Documents/ReviewForge/src
    python3 seacher/test_agent.py

验证流程：
    1. 导入 SearcherAgent 和 LLM
    2. 用测试问题调用 agent
    3. 打印生成结果（包含语义查询描述 + arXiv 关键词）
    4. 人工验证关键词是否合理
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm import LLM
from src.seacher import SearcherAgent


def main():
    API_KEY = "sk-grnzvqmqpizcjwszwfuyirfhwocbopgwhcibkitmrpsoauye"
    MODEL = "Qwen/Qwen3-8B"
    BASE_URL = "https://api.siliconflow.cn/v1"

    TEST_QUESTIONS = [
        "LoRA 在大模型微调中的应用与最新进展",
        "扩散模型（Diffusion Model）在图像生成领域的最新研究",
        "Transformer 架构在自然语言处理中的改进与变体",
    ]

    print("=" * 60)
    print("Searcher Agent 测试")
    print("=" * 60)

    print("\n[1] 初始化 LLM...")
    llm = LLM(
        api_key=API_KEY,
        model=MODEL,
        base_url=BASE_URL,
        temperature=0.1,
        max_tokens=1500,
    )
    print(f"    模型: {MODEL}")
    print("    ✓ LLM 初始化成功")

    print("\n[2] 初始化 SearcherAgent...")
    agent = SearcherAgent(llm=llm, verbose=True)
    print("    ✓ Agent 初始化成功")

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'=' * 60}")
        print(f"测试 {i}/{len(TEST_QUESTIONS)}")
        print(f"{'=' * 60}")
        print(f"问题: {question}")
        print("-" * 60)

        try:
            result = agent.run(question)
            print("\n生成结果:\n")
            print(result)
        except Exception as e:
            print(f"\n❌ 调用失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("测试完成")
    print("=" * 60)
    print("\n人工验证提示:")
    print("  1. 语义查询描述是否完整、具体、有语义信息量？")
    print("  2. arXiv 关键词是否符合 ti:/au:/abs:/cat: 规范？")
    print("  3. 将关键词复制到 https://arxiv.org/search 验证相关性")


if __name__ == "__main__":
    main()