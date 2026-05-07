"""YAML 配置加载 + 环境变量展开"""

import re
from pathlib import Path
from typing import Any

import yaml


class SearcherConfig:
    """MetaSearcher 配置"""

    def __init__(
        self,
        sources: dict[str, Any],
        fusion: dict[str, Any],
        dedup: dict[str, Any],
        rerank: dict[str, Any],
    ) -> None:
        self.sources = sources
        self.fusion = fusion
        self.dedup = dedup
        self.rerank = rerank

    @classmethod
    def from_file(cls, path: str | Path) -> "SearcherConfig":
        """从 YAML 文件加载配置，环境变量 ${VAR} 会自动展开"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        raw = p.read_text(encoding="utf-8")
        raw = _expand_env(raw)
        data = yaml.safe_load(raw)

        return cls(
            sources=data.get("sources", {}),
            fusion=data.get("fusion", {}),
            dedup=data.get("dedup", {}),
            rerank=data.get("rerank", {}),
        )


def _expand_env(text: str) -> str:
    """将 ${VAR} 替换为环境变量值"""
    pattern = re.compile(r"\$\{([^}]+)\}")

    def replacer(m: re.Match) -> str:
        var = m.group(1)
        import os

        return os.environ.get(var, m.group(0))

    return pattern.sub(replacer, text)