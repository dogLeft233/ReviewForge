"""检索器基类"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import httpx

from src.config import settings
from src.exceptions import RateLimitError, AuthenticationError
from src.models import PaperCard, ResourceCard

import logging

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """所有检索器的基类"""

    name: str = ""
    """检索器名称，子类覆盖"""

    def __init__(self) -> None:
        self._client = httpx.Client(
            timeout=settings.request_timeout_seconds,
            follow_redirects=True,
        )

    @abstractmethod
    def search(self, query: str, max_results: int | None = None) -> list[PaperCard]:
        """检索论文"""
        ...

    def search_resources(self, query: str) -> list[ResourceCard]:
        """检索资源（可选覆盖）"""
        return []

    def close(self) -> None:
        self._client.close()

    def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        """带重试和速率限制处理的 GET 请求"""
        for attempt in range(1, settings.max_retries + 1):
            try:
                resp = self._client.get(url, **kwargs)
            except httpx.TimeoutException:
                logger.warning("%s: attempt %d timeout", self.name, attempt)
                continue

            match resp.status_code:
                case 200:
                    return resp
                case 429:
                    raise RateLimitError(f"{self.name}: rate limited")
                case 403:
                    raise AuthenticationError(f"{self.name}: forbidden")
                case _:
                    if attempt < settings.max_retries:
                        import time

                        delay = settings.retry_delay_seconds * attempt
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

        raise RuntimeError(f"{self.name}: all {settings.max_retries} attempts failed")

    def __enter__(self) -> "BaseRetriever":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
