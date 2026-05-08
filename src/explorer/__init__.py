"""Explorer Agent — 领域探索 Agent

执行三阶段探索：
1. 了解领域概况与基本概念
2. 了解该领域的经典工作和历史阶段
3. 寻找 leaderboard 和 benchmark 了解前沿

最终输出：下游模型使用总结报告
"""

from .agent import ExplorerAgent
from .explorer_report import ExplorerReport

__all__ = ["ExplorerAgent", "ExplorerReport"]
