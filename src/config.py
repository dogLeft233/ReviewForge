"""全局配置"""

from dataclasses import dataclass, field
from os import environ
from pathlib import Path


@dataclass(slots=True)
class Settings:
    """所有可配置项，统一管理"""

    # ── Semantic Scholar ──
    semantic_scholar_api_key: str = field(
        default_factory=lambda: environ.get("S2_API_KEY", "")
    )

    # ── GitHub ──
    github_token: str = field(
        default_factory=lambda: environ.get("GITHUB_TOKEN", "")
    )

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


settings = Settings()
