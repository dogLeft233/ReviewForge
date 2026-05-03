"""异常层次定义"""

import logging

logger = logging.getLogger(__name__)


class ReviewForgeError(Exception):
    """应用基异常"""

    def __init__(self, message: str, cause: Exception | None = None):
        self.cause = cause
        super().__init__(message)


class ValidationError(ReviewForgeError):
    """输入验证错误"""


class ConfigurationError(ReviewForgeError):
    """配置错误"""


class RetrieverError(ReviewForgeError):
    """检索器基异常"""


class RateLimitError(RetrieverError):
    """API 速率限制"""


class AuthenticationError(RetrieverError):
    """认证失败"""


class EmptyResultError(RetrieverError):
    """检索结果为空"""
