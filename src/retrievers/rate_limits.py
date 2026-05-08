"""速率限制基础设施

两层设计：
1. SourceRateLimit — 每类数据源的分级限速静态配置
2. RateLimiter — 运行时令牌桶限速器，由 BaseRetriever._get() 自动调用
"""

import time
from dataclasses import dataclass
from threading import Lock

import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 静态配置
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class SourceRateLimit:
    """单个 source 的速率配置"""

    # 最小请求间隔（秒），0 = 不限制
    min_interval_seconds: float = 0.0
    # 突发并发上限（超过则排队）
    burst: int = 1
    # 是否强制使用 key 才能调用
    requires_key: bool = False
    # 有 key 时的并发上限
    key_concurrency: int = 3
    # 备注
    note: str = ""


SOURCE_RATE_LIMITS: dict[str, SourceRateLimit] = {
    "arxiv": SourceRateLimit(
        min_interval_seconds=5.0,
        burst=1,
        requires_key=False,
        key_concurrency=2,
        note="无 key 限 1req/3s；注册邮箱后可降低至 1req/2s",
    ),
    "github": SourceRateLimit(
        min_interval_seconds=6.0,  # 无 key: 10 req/min
        burst=1,
        requires_key=False,
        key_concurrency=5,
        note="有 GITHUB_TOKEN: search 5000 req/hour；无 key 10 req/min",
    ),
    "huggingface": SourceRateLimit(
        min_interval_seconds=1.0,
        burst=1,
        requires_key=False,
        key_concurrency=3,
        note="Hub API 1 req/s，无硬性限速但建议保守",
    ),
    "semantic_scholar": SourceRateLimit(
        min_interval_seconds=1.0,
        burst=1,
        requires_key=False,
        key_concurrency=3,
        note="无 key: 1000 req/s 共享（生产限速）；有 S2_API_KEY: 1 RPS intro tier；申请进阶 key 可达 10 RPS+",
    ),
    "dblp": SourceRateLimit(
        min_interval_seconds=1.0,
        burst=2,
        requires_key=False,
        key_concurrency=3,
        note="DBLP 未公开限速，频繁请求会触发 500",
    ),
    "serper": SourceRateLimit(
        min_interval_seconds=1.0,
        burst=1,
        requires_key=True,
        key_concurrency=2,
        note="无 SERPER_API_KEY 时 retriever 内部跳过",
    ),
    "bocha": SourceRateLimit(
        min_interval_seconds=1.0,
        burst=1,
        requires_key=True,
        key_concurrency=2,
        note="无 BOCHA_API_KEY 时 retriever 内部跳过",
    ),
    "hackernews": SourceRateLimit(
        min_interval_seconds=0.0,
        burst=1,
        requires_key=False,
        key_concurrency=2,
        note="Algolia HN API free tier 10k/month",
    ),
    "wikipedia": SourceRateLimit(
        min_interval_seconds=0.0,
        burst=5,
        requires_key=False,
        key_concurrency=5,
        note="Wikipedia API 相对宽松",
    ),
}


def get_source_config(name: str) -> SourceRateLimit:
    """获取 source 速率配置，不存在时返回宽松默认值"""
    return SOURCE_RATE_LIMITS.get(name, SourceRateLimit())


# ──────────────────────────────────────────────
# 运行时限速器
# ──────────────────────────────────────────────


class RateLimiter:
    """基于最小间隔的同步限速器

    用法:
        limiter = RateLimiter(min_interval_seconds=3.0)
        limiter.wait()  # 阻塞直到允许下次请求
    """

    def __init__(self, min_interval_seconds: float = 0.0) -> None:
        self._interval = min_interval_seconds
        self._last: float = 0.0
        self._lock = Lock()

    def wait(self) -> None:
        """阻塞直到允许发出下一次请求"""
        if self._interval <= 0:
            return

        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self._interval:
                sleep_for = self._interval - elapsed
                logger.debug("rate_limiter: sleeping %.2fs", sleep_for)
                time.sleep(sleep_for)
            self._last = time.monotonic()

    @property
    def min_interval(self) -> float:
        return self._interval

    @min_interval.setter
    def min_interval(self, value: float) -> None:
        self._interval = value
