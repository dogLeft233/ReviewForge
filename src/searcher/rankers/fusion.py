"""RRF 融合算法 — Reciprocal Rank Fusion"""

from collections import defaultdict

from src.searcher.result import SearchResult


def rrf_fusion(results_by_source: dict[str, list[SearchResult]], k: int = 60) -> list[SearchResult]:
    """多源结果 RRF 融合

    Args:
        results_by_source: {source_name: [SearchResult]} — 每个源的排序结果
        k: RRF 公式参数，默认 60（对排名靠后的结果做平滑）

    Returns:
        按 RRF 分数降序排列的合并列表
    """
    scores: dict[str, float] = defaultdict(float)
    url_to_result: dict[str, SearchResult] = {}

    for source, results in results_by_source.items():
        for rank, result in enumerate(results, start=1):
            url = result.url
            rrf_score = 1 / (k + rank)
            scores[url] += rrf_score
            if url not in url_to_result:
                url_to_result[url] = result

    # 按分数降序排列
    sorted_urls = sorted(scores.keys(), key=lambda u: scores[u], reverse=True)

    fused: list[SearchResult] = []
    for url in sorted_urls:
        r = url_to_result[url]
        fused.append(
            SearchResult(
                title=r.title,
                url=r.url,
                snippet=r.snippet,
                rank_score=round(scores[url], 4),
                source=r.source,
                category=r.category,
                published_date=r.published_date,
                domain=r.domain,
            )
        )
    return fused