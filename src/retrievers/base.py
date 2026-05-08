"""检索器基类 — 所有数据驱动的统一抽象"""

from abc import ABC, abstractmethod
from typing import Any

import httpx

from src.config import settings
from src.retrievers.exceptions import RateLimitError, AuthenticationError
from src.models import PaperCard, ResourceCard
from src.retrievers.rate_limits import RateLimiter, get_source_config, get_source_limiter

import logging

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """所有检索器的基类

    子类只需:
    1. 设置 name 类变量（与 SOURCE_RATE_LIMITS 的 key 对应）
    2. 实现 search() 或 search_resources()
    3. 内部发请求用 self._get(url) — 自动处理重试 + 速率限制
    """

    name: str = ""
    """数据源名称，须与 SOURCE_RATE_LIMITS 的 key 匹配"""

    def __init__(self) -> None:
        self._client = httpx.Client(
            timeout=settings.request_timeout_seconds,
            follow_redirects=True,
        )
        # 使用全局单例限速器（进程级共享，所有同名 source 共用同一把锁）
        # 在 ThreadPoolExecutor 并发场景下，防止各自创建独立 RateLimiter 导致限流失效
        self._rate_limiter = get_source_limiter(self.name)
        self._max_retries = settings.max_retries
        self._retry_delay = settings.retry_delay_seconds

    # ── 抽象接口 ──

    @abstractmethod
    def search(self, query: str, max_results: int | None = None) -> list[PaperCard]:
        """检索论文"""
        ...

    def search_resources(
        self, query: str, max_results: int | None = None
    ) -> list[ResourceCard]:
        """检索资源（非论文型数据源覆盖此项）"""
        return []

    # ── 公共方法 ──

    def close(self) -> None:
        self._client.close()

    # ── 受保护的 HTTP 方法 ──

    def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        """带重试和速率限制的 GET 请求

        自动:
        1. 调用 self._rate_limiter.wait() 确保不超频
        2. 对 429/403/网络错误做指数退避重试
        """
        self._rate_limiter.wait()

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.get(url, **kwargs)
            except httpx.TimeoutException:
                logger.warning("%s: attempt %d timeout", self.name, attempt)
                if attempt < self._max_retries:
                    import time

                    time.sleep(self._retry_delay * attempt)
                continue

            match resp.status_code:
                case 200:
                    return resp
                case 429:
                    logger.warning(
                        "%s: 429 rate limited, retry %d after %.1fs",
                        self.name, attempt, self._retry_delay * attempt,
                    )
                    # 通知全局限速器进入惩罚期，防止后续并发请求继续撞墙
                    self._rate_limiter.penalize(self._retry_delay * (attempt + 1))
                    if attempt < self._max_retries:
                        import time
                        time.sleep(self._retry_delay * attempt)
                    else:
                        raise RateLimitError(
                            f"{self.name}: rate limited (gave up after {self._max_retries} attempts)"
                        )
                case 403:
                    raise AuthenticationError(f"{self.name}: forbidden")
                case _:
                    if attempt < self._max_retries:
                        import time

                        delay = self._retry_delay * attempt
                        logger.warning(
                            "%s: %d, retry %d after %.1fs",
                            self.name,
                            resp.status_code,
                            attempt,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        resp.raise_for_status()

        raise RuntimeError(
            f"{self.name}: all {self._max_retries} attempts failed"
        )

    # ── 上下文管理 ──

    def __enter__(self) -> "BaseRetriever":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
