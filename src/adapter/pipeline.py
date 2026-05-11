"""adapter 包对外的两个入口。

- convert(raw)              — 仅跑脚本转换
- convert_with_llm(raw, ...) — 脚本 + LLM 校验/补全

LLM 客户端可以由调用方传入；不传则用 src.config.settings 自动建。
"""

from __future__ import annotations

import logging
from typing import Any

from src.adapter.schema import VisualizationData
from src.adapter.script_adapter import writer_json_to_visualization

logger = logging.getLogger(__name__)


def convert(raw: dict[str, Any]) -> VisualizationData:
    """规则化转换，不调 LLM。"""
    return writer_json_to_visualization(raw)


def convert_with_llm(
    raw: dict[str, Any],
    llm: Any | None = None,
) -> VisualizationData:
    """先跑脚本转换，再过一遍 LLM refiner。LLM 不可用时静默退回脚本结果。"""
    viz = writer_json_to_visualization(raw)

    if llm is None:
        try:
            from src.config import settings
            from src.llm import LLM

            llm = LLM(
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                temperature=settings.llm_temperature,
                max_tokens=4000,
                timeout_seconds=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
        except Exception as exc:
            logger.warning("无法初始化 LLM，跳过 refine：%s", exc)
            return viz

    try:
        from src.adapter.llm_refiner import refine_with_llm

        return refine_with_llm(viz, raw, llm)
    except Exception as exc:
        logger.warning("LLM refine 失败，回退脚本结果：%s", exc)
        return viz
