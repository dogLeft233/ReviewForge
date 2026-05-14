"""速率限制基础设施

三层设计：
1. SourceRateLimit — 每类数据源的静态速率配置
2. RateLimiter — 线程安全的全局令牌桶限速器
3. get_source_limiter() — 全局单例，同 source 的所有实例共享同一把锁

关键修复：
- _last 初始化为负值，确保第一个请求也必须等待 min_interval
- penalize() 支持 429 后惩罚性延长封锁时间
- _injected_now 允许模拟模式（测试时绕过真实 HTTP）
"""

import time
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 静态配置
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class SourceRateLimit:
    """单个 source 的速率配置"""

    min_interval_seconds: float = 0.0
    burst: int = 1
    requires_key: bool = False
    key_concurrency: int = 3
    note: str = ""


SOURCE_RATE_LIMITS: dict[str, SourceRateLimit] = {
    "arxiv": SourceRateLimit(
        min_interval_seconds=10.0,   # 无 key 限 1req/3s，保守取 10s
        burst=1,
        requires_key=False,
        key_concurrency=2,
        note="无 key 限 1req/3s；注册邮箱后可降低至 1req/2s",
    ),
    "github": SourceRateLimit(
        min_interval_seconds=6.0,
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
        note="无 key: 1000 req/s 共享；S2_API_KEY: 1 RPS；申请进阶 key 可达 10 RPS+",
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
    "ar5iv": SourceRateLimit(
        min_interval_seconds=1.0,
        burst=1,
        requires_key=False,
        key_concurrency=1,
        note="建议 1-2s 间隔，无明确官方限速，全局锁保护，429 后重试",
    ),
}


def get_source_config(name: str) -> SourceRateLimit:
    """获取 source 速率配置，不存在时返回宽松默认值"""
    return SOURCE_RATE_LIMITS.get(name, SourceRateLimit())


# ──────────────────────────────────────────────
# 运行时限速器
# ──────────────────────────────────────────────


class RateLimiter:
    """基于最小间隔的同步限速器（线程安全）

    关键设计：
    - _last 初始化为负值（-interval）：第一个请求也必须等待 interval，
      避免 _last=0 导致首次请求立即通过的问题
    - _penalty_until：惩罚期，429 等情况下临时延长封锁时间
    - _injected_now：模拟模式，允许外部注入时间戳，绕过真实 HTTP 延迟
    """

    def __init__(
        self,
        min_interval_seconds: float = 0.0,
        *,
        _init_last: Optional[float] = None,
    ) -> None:
        self._interval = min_interval_seconds
        # 关键修复：初始化为负值，确保第一个请求也必须等待 interval
        # _init_last 仅用于测试注入自定义起始时间
        self._last: float = (
            _init_last if _init_last is not None else -min_interval_seconds
        )
        self._penalty_until: float = 0.0    # 惩罚期截止时间（monotonic）
        self._lock = Lock()
        self._injected_now: Optional[float] = None  # 模拟时间注入

    def _now(self) -> float:
        """返回当前时间（支持模拟注入）"""
        if self._injected_now is not None:
            return self._injected_now
        return time.monotonic()

    def wait(self) -> None:
        """阻塞直到允许发出下一次请求

        伪代码:
            deadline = max(_last + _interval, _penalty_until)
            if _now() < deadline:
                sleep deadline - _now()
            _last = _now()
        """
        if self._interval <= 0 and self._penalty_until <= 0:
            return

        with self._lock:
            now = self._now()
            # ── 惩罚期：独立分支，sleep 后直接 return ──
            if self._penalty_until > 0 and now < self._penalty_until:
                sleep_for = self._penalty_until - now
                logger.debug("rate_limiter: penalty sleep %.2fs (penalty_until=%.2f, now=%.2f)",
                             sleep_for, self._penalty_until, now)
                time.sleep(sleep_for)
                self._last = self._now()
                return

            # ── 普通间隔 ──
            elapsed = now - self._last
            if elapsed <= self._interval:
                sleep_for = self._interval - elapsed
                logger.debug("rate_limiter: sleeping %.2fs (elapsed=%.2f, interval=%.2f)",
                             sleep_for, elapsed, self._interval)
                time.sleep(sleep_for)

            self._last = self._now()

    def penalize(self, extra_seconds: float) -> None:
        """触发惩罚：临时延长封锁时间（429 时由 BaseRetriever 调用）"""
        with self._lock:
            now = self._now()
            new_penalty = now + extra_seconds
            if new_penalty > self._penalty_until:
                self._penalty_until = new_penalty
                logger.warning("rate_limiter: penalize +%.1fs until %.2f", extra_seconds, self._penalty_until)

    def set_sim_time(self, t: float) -> None:
        """设置模拟时间（用于测试，注入时间戳）"""
        self._injected_now = t

    def reset_sim_time(self) -> None:
        """清除模拟时间，恢复真实时钟"""
        self._injected_now = None

    @property
    def min_interval(self) -> float:
        return self._interval

    @min_interval.setter
    def min_interval(self, value: float) -> None:
        self._interval = value

    @property
    def penalty_until(self) -> float:
        return self._penalty_until


# ──────────────────────────────────────────────
# 全局限速器单例（进程级共享）
# ──────────────────────────────────────────────

_GLOBAL_LIMITERS: dict[str, RateLimiter] = {}
_GLOBAL_LIMITERS_LOCK = Lock()


def get_source_limiter(source_name: str) -> RateLimiter:
    """获取指定 source 的全局单例 RateLimiter

    所有同名 source 的 retriever 实例共享同一把锁，
    实现进程级并发限速（解决 ThreadPoolExecutor 场景下的限流失效问题）。
    """
    if source_name not in _GLOBAL_LIMITERS:
        config = get_source_config(source_name)
        with _GLOBAL_LIMITERS_LOCK:
            if source_name not in _GLOBAL_LIMITERS:
                _GLOBAL_LIMITERS[source_name] = RateLimiter(config.min_interval_seconds)
    return _GLOBAL_LIMITERS[source_name]


# ──────────────────────────────────────────────
# 模拟模式上下文管理器（测试用）
# ──────────────────────────────────────────────

def set_sim_time_for_source(source_name: str, t: float) -> None:
    """为指定 source 的全局限速器注入模拟时间"""
    limiter = get_source_limiter(source_name)
    limiter.set_sim_time(t)


def reset_sim_time_for_source(source_name: str) -> None:
    """清除指定 source 的模拟时间"""
    limiter = get_source_limiter(source_name)
    limiter.reset_sim_time()
