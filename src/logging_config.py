"""统一日志配置——集中管理项目所有模块的日志行为

用法:
    from src.logging_config import get_logger, set_log_level

    logger = get_logger(__name__)
    logger.debug("debug message")
    logger.info("info message")

    set_log_level("DEBUG")  # 实时调整所有 logger 级别
"""

import logging
import sys
from typing import Final

# 支持的日志级别
LOG_LEVELS: list[str] = ["DEBUG", "INFO", "WARNING", "ERROR"]

# 默认级别
DEFAULT_LOG_LEVEL: Final[str] = "INFO"

# 全局 logger 字典（按名称缓存）
_LOGGERS: dict[str, logging.Logger] = {}

# 全局级别标志
_global_level: str = DEFAULT_LOG_LEVEL


# ─────────────────────────────────────────────────────────────────────────────
# 格式化
# ─────────────────────────────────────────────────────────────────────────────


def _make_formatter() -> logging.Formatter:
    """标准日志格式：时间 | 级别 | 模块名 | 消息"""
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 全局级别控制
# ─────────────────────────────────────────────────────────────────────────────


def set_log_level(level: str) -> None:
    """设置所有 logger 的日志级别（实时生效）

    Args:
        level: "DEBUG" | "INFO" | "WARNING" | "ERROR"
    """
    global _global_level
    if level not in LOG_LEVELS:
        raise ValueError(f"Invalid log level: {level}. Must be one of {LOG_LEVELS}")
    _global_level = level

    numeric = getattr(logging, level)
    for logger in _LOGGERS.values():
        logger.setLevel(numeric)


def get_log_level() -> str:
    """获取当前全局日志级别"""
    return _global_level


# ─────────────────────────────────────────────────────────────────────────────
# 获取 logger
# ─────────────────────────────────────────────────────────────────────────────


def get_logger(name: str) -> logging.Logger:
    """获取或创建指定名称的 logger

    Args:
        name: 通常传入 __name__（模块路径）

    Returns:
        配置好的 logger 实例
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)

    # 避免重复添加 handler（logging.getLogger 可能返回已有 handler 的 logger）
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_make_formatter())
        logger.addHandler(handler)

    # 设置级别
    numeric = getattr(logging, _global_level)
    logger.setLevel(numeric)

    # 防止传播到根 logger（避免重复输出）
    logger.propagate = False

    _LOGGERS[name] = logger
    return logger