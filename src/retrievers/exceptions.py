"""Retrievers 包内部异常——检索相关异常集中定义在此"""


class RetrieverError(Exception):
    """检索器基异常"""


class RateLimitError(RetrieverError):
    """API 速率限制"""


class AuthenticationError(RetrieverError):
    """认证失败"""


class EmptyResultError(RetrieverError):
    """检索结果为空"""
