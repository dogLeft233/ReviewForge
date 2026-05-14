"""全局配置——从 config.json 读取，环境变量覆盖 API keys

用法:
    from src.config import settings
    settings.bocha_api_key  # 优先取 env var BOCHA_API_KEY，再回退 config.json
    settings.max_retries    # 取自 config.json
"""

import json
import os
from pathlib import Path
from typing import Any


def _find_config() -> Path:
    """从项目根目录定位 config.json"""
    return Path(__file__).parent.parent / "config.json"


def _load_config() -> dict[str, Any]:
    path = _find_config()
    if not path.exists():
        raise FileNotFoundError(
            f"config.json not found at {path}. "
            "Create one from config.example.json or check PROJECT_ROOT."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 环境变量覆盖映射 ───────────────────────────────────────
_ENV_OVERRIDES: dict[str, str] = {
    "bocha_api_key": "BOCHA_API_KEY",
    "semantic_scholar_api_key": "S2_API_KEY",
    "github_token": "GITHUB_TOKEN",
    "serper_api_key": "SERPER_API_KEY",
    "serpapi_api_key": "SERPAPI_API_KEY",
    "huggingface_token": "HF_TOKEN",
    "llm_api_key": "LLM_API_KEY",
    "reranker_api_key": "RERANKER_API_KEY",
    "arxiv_email": "ARXIV_EMAIL",
    "adapter_enhance_api_key": "ADAPTER_ENHANCE_API_KEY",
}


class Settings:
    """类型安全的配置访问器——属性名即 config.json 字段名

    优先级: 环境变量 > config.json
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        # 1. 环境变量覆盖（仅敏感字段）
        if name in _ENV_OVERRIDES:
            if env_val := os.environ.get(_ENV_OVERRIDES[name]):
                return env_val

        # 2. config.json 配置值
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Settings has no field '{name}'")


settings = Settings(_load_config())
