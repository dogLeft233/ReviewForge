"""全局配置"""

from dataclasses import dataclass, field
from os import environ
from pathlib import Path


@dataclass(slots=True)
class Settings:
    """所有可配置项，统一管理"""

    # ── 检索器 API Keys ──
    bocha_api_key: str = field(
        default_factory=lambda: environ.get("BOCHA_API_KEY", ""),
    )
    semantic_scholar_api_key: str = field(
        default_factory=lambda: environ.get("S2_API_KEY", ""),
    )
    github_token: str = field(
        default_factory=lambda: environ.get("GITHUB_TOKEN", ""),
    )
    serper_api_key: str = field(
        default_factory=lambda: environ.get("SERPER_API_KEY", ""),
    )
    # Wikipedia 不需要 API Key（免费）

    # ── 重试 ──
    max_retries: int = 3
    retry_delay_seconds: float = 2.0

    # ── 检索默认参数 ──
    default_max_results: int = 50
    request_timeout_seconds: float = 30.0

    # ── 缓存 ──
    cache_dir: str = field(
        default_factory=lambda: str(
            Path.home() / ".reviewforge" / "cache"
        )
    )
    cache_ttl_hours: int = 24

    # ── 路径 ──
    output_dir: str = field(
        default_factory=lambda: str(
            Path.home() / ".reviewforge" / "output"
        )
    )

    # ── LLM ──
    llm_api_key: str = field(
        default_factory=lambda: environ.get("LLM_API_KEY", "")
    )
    llm_base_url: str = field(
        default="https://api.siliconflow.cn/v1"
    )
    llm_model: str = field(default="Qwen/Qwen3-8B")
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2


settings = Settings()
