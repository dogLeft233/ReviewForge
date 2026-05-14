#!/usr/bin/env python3
"""测试 BFS 搜索模块 via ChatAgent 工具调用（DEBUG 模式）"""

import sys
import os
import logging

# ── 项目路径设置 ──────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ["PYTHONPATH"] = os.path.join(_ROOT, "src")

# ── DEBUG 日志配置 ────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
# 第三方库日志调成 INFO 减少干扰
for mod in ["httpx", "urllib3", "httpcore", "hpack", "charset_normalizer"]:
    logging.getLogger(mod).setLevel(logging.WARNING)

logger = logging.getLogger("test_bfs_agent")

# ── 导入 ──────────────────────────────────────────────────────
from src.agent.chat_agent import ChatAgent
from src.visualizer.schema import VisualizationData, Overview
from src.config import settings

def main():
    topic = "ASR automatic speech recognition"

    print(f"\n{'='*60}")
    print(f"🧪 测试 BFS Search via ChatAgent 工具调用")
    print(f"📋 Topic: {topic}")
    print(f"{'='*60}\n")

    # 初始化 ChatAgent
    agent = ChatAgent(
        session_id="test_bfs",
        system_prompt=(
            "你是一个学术研究 AI，可以用工具深度搜索论文。"
            "当用户提出研究领域时，优先使用 bfs_search 工具进行深度论文发现。"
            "请用中文回答。"
        ),
    )

    # 构建空的 VisualizationData（测试用）
    empty_data = VisualizationData(
        overview=Overview(
            definition=f"研究领域：{topic}",
            core_questions=[],
            key_concepts=[],
        ),
        timeline=[],
        methods=[],
        papers=[],
        benchmarks=[],
        frontiers=[],
        graph={"nodes": [], "edges": []},
    )

    # ── 直接调用 bfs_search 工具 ──────────────────────────────
    print("📡 调用 bfs_search 工具...\n")

    question = f"使用 BFS 搜索 '{topic}' 领域的论文，找出重要论文和引用关系"
    answer = agent.ask(question, empty_data)

    print(f"\n{'='*60}")
    print("📤 搜索结果:")
    print(f"{'='*60}")
    print(answer[:3000])
    if len(answer) > 3000:
        print(f"\n... (共 {len(answer)} 字符，已截断)")

if __name__ == "__main__":
    main()