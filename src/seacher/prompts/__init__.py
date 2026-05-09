"""Prompts — 分文件存储的提示词"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(filename: str) -> str:
    """加载 prompt 文件内容"""
    path = PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


SYSTEM_PROMPT = load_prompt("system.txt")
USER_PROMPT_TEMPLATE = load_prompt("user.txt")
USER_PROMPT_NO_SEARCH = load_prompt("user_no_search.txt")
