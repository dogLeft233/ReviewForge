"""UI 端 JSON 加载器 + 内置 ASR demo fallback。

`load_json(file_or_path)` — 接受 streamlit UploadedFile 或文件路径。
`validate_data(data: dict)` — 走 Pydantic 校验。
`load_demo_data()`         — 找一份本地 visualization_data.json 顶包；找不到就给空骨架。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.adapter.schema import VisualizationData


def validate_data(data: dict[str, Any]) -> VisualizationData:
    return VisualizationData.model_validate(data)


def load_json(source: Any) -> VisualizationData:
    """source 可以是 streamlit UploadedFile、Path、str 或 bytes。"""

    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return validate_data(json.loads(raw))

    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8")
        return validate_data(json.loads(text))

    if isinstance(source, bytes):
        return validate_data(json.loads(source.decode("utf-8")))

    raise TypeError(f"unsupported source type: {type(source)!r}")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _find_demo_path() -> Path | None:
    root = _project_root() / "tmp"
    if not root.exists():
        return None
    # 优先 ASR；否则任何一个领域目录下的 visualization_data.json
    preferred = root / "ASR自动语音识别" / "visualization_data.json"
    if preferred.exists():
        return preferred
    for d in sorted(root.iterdir()):
        if d.is_dir():
            cand = d / "visualization_data.json"
            if cand.exists():
                return cand
    return None


def load_demo_data() -> VisualizationData:
    p = _find_demo_path()
    if p is None:
        return VisualizationData(topic="Demo（无可用数据）")
    try:
        return load_json(p)
    except Exception:
        return VisualizationData(topic="Demo（加载失败）")
