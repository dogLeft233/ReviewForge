"""SiliconFlow Rerank API 调用"""

import os
from typing import Any

import httpx

from src.searcher.result import SearchResult


SILICONFLOW_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"


def rerank(
    query: str,
    results: list[SearchResult],
    api_key: str | None = None,
    model: str = "BAAI/bge-reranker-v2-m3",
    top_n: int = 10,
) -> list[SearchResult]:
    """调用 SiliconFlow Rerank API 对结果重排序

    Args:
        query: 搜索查询
        results: 待排序的 SearchResult 列表
        api_key: SiliconFlow API Key（默认从环境变量 SILICONFLOW_API_KEY 读取）
        model: Rerank 模型
        top_n: 返回前 N 条结果

    Returns:
        按 relevance_score 降序排列的 SearchResult 列表
    """
    if not results:
        return []

    key = api_key or os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        # 无 API Key 时降级为 RRF 结果
        return results

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": [r.title + " " + r.snippet for r in results],
        "top_n": min(top_n, len(results)),
        "return_documents": False,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(SILICONFLOW_RERANK_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # 从响应中提取排序后的 index，按 index 重新排列
        reranked: list[SearchResult] = []
        for item in data.get("results", []):
            idx = item["index"]
            score = item["relevance_score"]
            r = results[idx]
            reranked.append(
                SearchResult(
                    title=r.title,
                    url=r.url,
                    snippet=r.snippet,
                    rank_score=round(score, 4),
                    source=r.source,
                    category=r.category,
                    published_date=r.published_date,
                    domain=r.domain,
                )
            )
        return reranked

    except Exception as e:
        # Rerank 失败，降级为原始结果
        return results