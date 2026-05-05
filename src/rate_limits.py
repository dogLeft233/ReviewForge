"""搜索源速率配置 — 各 API 的分级限速参数

Sources:
- arxiv       https://info.arxiv.org/help/api/start
- semantic_scholar  https://api.semanticscholar.org
- dblp        https://dblp.org
- github      https://docs.github.com/en/rest/rate-limit
- huggingface https://huggingface.co/docs/hub/rate_limit
- hackernews  https://hn.algolia.com/api
- serper      https://serper.dev
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceRateLimit:
    """单个 source 的速率配置"""

    # 每多少秒允许 1 次请求（0 = 无限制）
    min_interval_seconds: float = 0.0
    # 突发并发上限（超过则排队）
    burst: int = 1
    # 是否强制使用 key 才能调用
    requires_key: bool = False
    # 无 key 时的并发上限
    no_key_concurrency: int = 1
    # 有 key 时的并发上限
    key_concurrency: int = 3
    # 备注
    note: str = ""


# ── 分级配置 ────────────────────────────────────────────────
#
#  tier         无 key              有 key
#  ─────────────────────────────────────────────────────────
#  arxiv       1 req / 3s           1 req / 2s (注册邮箱后)
#  s2          1 req / 1s           100 req / hour (free tier)
#  dblp        ~1 req / 1s (无官方)  同左
#  github      10 req / min (search) 5000 req / hour (search)
#  hf          1 req / 1s           25 req / min (无明确认证)
#  hn/algolia  ~1 req / 1s           同左 (free tier 10k/month)
#  serper      1 req / 1s (free 2500/mo) 1 req / 1s
#
SOURCE_RATE_LIMITS: dict[str, SourceRateLimit] = {
    # 论文检索源
    "arxiv": SourceRateLimit(
        min_interval_seconds=3.0,  # 无 key: 1 req / 3s（严格）
        burst=1,
        requires_key=False,
        no_key_concurrency=1,
        key_concurrency=2,
        note="无 key 限 1req/3s；注册邮箱后 1req/2s（给 api@arxiv.org 发邮件）",
    ),
    "semantic_scholar": SourceRateLimit(
        min_interval_seconds=1.0,  # 无 key: 1 req / s（大概）
        burst=1,
        requires_key=False,
        no_key_concurrency=1,
        key_concurrency=5,
        note="有 S2_API_KEY: free tier 100 req/hour；无 key 更慢",
    ),
    "dblp": SourceRateLimit(
        min_interval_seconds=1.0,  # 无明确文档，保守设 1 req / s
        burst=2,
        requires_key=False,
        no_key_concurrency=2,
        key_concurrency=3,
        note="DBLP 官方未公开限速，但频繁请求会触发 500；建议加 delay",
    ),
    # 资源检索源
    "github": SourceRateLimit(
        min_interval_seconds=0.0,  # 不做 interval 限速，用 burst 控制并发
        burst=1,                     # 无 key: 10 req/min → burst=1
        requires_key=False,
        no_key_concurrency=1,
        key_concurrency=5,
        note="有 GITHUB_TOKEN: search 5000 req/hour；无 key 10 req/min",
    ),
    "huggingface": SourceRateLimit(
        min_interval_seconds=1.0,
        burst=1,
        requires_key=False,
        no_key_concurrency=1,
        key_concurrency=3,
        note="Hub API 1 req/s 无明确认证，burst 保持 1",
    ),
    "hackernews": SourceRateLimit(
        min_interval_seconds=0.0,  # Algolia free: 10k/month ≈ 0.2 req/s，burst 控制
        burst=1,
        requires_key=False,
        no_key_concurrency=1,
        key_concurrency=2,
        note="Algolia HN API free tier 10k/month，burst 保持 1",
    ),
    # 搜索源
    "serper": SourceRateLimit(
        min_interval_seconds=1.0,  # free 2500 queries/month ≈ 1 req / 10s
        burst=1,
        requires_key=True,  # 必须 key，否则直接报错
        no_key_concurrency=0,  # 0 = 完全禁用（无 key 时不调用）
        key_concurrency=2,
        note="无 SERPER_API_KEY 时 retriever 内部跳过；需要注册 serper.dev",
    ),
}


def get_source_config(name: str) -> SourceRateLimit:
    """获取 source 速率配置，不存在时返回宽松默认值"""
    return SOURCE_RATE_LIMITS.get(name, SourceRateLimit())
