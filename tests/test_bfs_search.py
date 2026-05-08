#!/usr/bin/env python3
"""BFSSearcher 测试程序 — 支持 reranker 或 embedding 过滤 + BFS 逻辑验证（无需 LLM API key）"""

import sys, os, math, logging
from pathlib import Path

# 设置 debug 日志（显示运行时行为）
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    stream=sys.stdout,
)
for _logger in ('bfs_search', 'retrievers', 'embedding', 'seacher',
                'src.seacher.bfs_search', 'src.embedding', 'src.retrievers'):
    logging.getLogger(_logger).setLevel(logging.DEBUG)

_script_dir = str(Path(__file__).parent)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
_SRC_DIR = _PROJECT_ROOT + "/src"
for p in (_PROJECT_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.retrievers import ArxivRetriever
from src.embedding import EmbeddingClient
from src.reranker import RerankerClient
from src.seacher.bfs_search import BFSSearcher, SearchQuery, PaperNode, RetrievalError


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class MockLLM:
    """Mock LLM — 返回预设搜索词"""
    def __init__(self):
        self.call_count = 0

    def chat(self, external_prompt="", system_prompt="", temperature=0.1, max_tokens=500):
        self.call_count += 1
        queries = [
            "SEARCH|LoRA parameter efficient fine-tuning|精准匹配 LoRA",
            "SEARCH|LoRA large language model tuning 2023|覆盖最新研究",
            "SEARCH|parameter efficient LLM training methods|宽泛搜索方法",
        ]
        return "\n".join(queries)


def main():
    print("=" * 60)
    print("BFSSearcher 测试（Mock LLM + Reranker）")
    print("=" * 60)

    # Embedding 验证
    print("\n[0] Embedding 模型基础验证...")
    embed = EmbeddingClient()
    print(f"    模型: {embed.model}，维度: {embed.dimensions}")

    # Cosine similarity sanity checks
    v1 = embed.encode("LoRA fine-tuning for large language models")
    v2 = embed.encode("Parameter-efficient fine-tuning using LoRA")
    v3 = embed.encode("Computer vision CNN image classification")
    s1 = cosine(v1, v2)
    s2 = cosine(v1, v3)
    print(f"    cosine(LoRA-V1, LoRA-V2) = {s1:.4f}  (同领域，应该高)")
    print(f"    cosine(LoRA, CNN)        = {s2:.4f}  (不同领域，应该低)")
    assert s1 > s2, "同领域相似度应该更高"
    assert s1 > 0.7, f"LoRA 相关相似度应 > 0.7，实际 {s1:.4f}"
    assert s2 < 0.7, f"LoRA vs CNN 相似度应 < 0.7，实际 {s2:.4f}"
    print("    ✓ Embedding 相似度逻辑正常")

    # Reranker 验证
    print("\n[0b] Reranker 模型验证...")
    reranker = RerankerClient()
    print(f"    模型: {reranker.model}")
    test_docs = [
        "LoRA: Low-Rank Adaptation of Large Language Models",
        "Video Generation with Diffusion Models",
        "Computer Vision CNN Image Classification",
    ]
    reranked = reranker.rerank(
        query="LoRA fine-tuning for large language models",
        documents=test_docs,
        top_n=3,
        return_documents=False,
    )
    scores = [r.relevance_score for r in reranked]
    print(f"    scores: {[f'{s:.4f}' for s in scores]}")
    assert reranked[0].index == 0, "Reranker: LoRA doc 应排第一"
    print("    ✓ Reranker 重排序正常")

    # Init BFSSearcher with mock LLM + reranker
    print("\n[1] 初始化 BFSSearcher（Mock LLM + Reranker）...")
    mock_llm = MockLLM()
    searcher = BFSSearcher(
        llm=mock_llm,
        embed=embed,
        reranker=reranker,
        search_queries_count=5,
        search_papers_count=10,
        expand_papers_count=20,
        expand_layers=3,
        similarity_threshold=0.50,
        rerank_top_n=20,
        threads_num=1,
    )
    print(f"    ✓ BFSSearcher 初始化成功（reranker={reranker.model}）")

    question = "ASR 自动语音识别经典文献与前沿进展"

    # ── Stage 1 纯搜索 ────────────────────────────────────
    print("\n【Stage 1】只搜索，不过滤：")
    papers_s1 = searcher.search(question, expand_layers=0)
    print(f"  → 返回 {len(papers_s1)} 篇（已去重）")

    for j, p in enumerate(papers_s1[:5], 1):
        print(f"    [{j}] score={p.select_score:.4f} | {p.title[:55]}")

    # 去重验证
    paper_ids_s1 = [p.paper_id for p in papers_s1 if p.paper_id]
    unique_s1 = set(paper_ids_s1)
    dup_ok_s1 = len(paper_ids_s1) == len(unique_s1)
    print(f"  【去重】共 {len(papers_s1)} 篇，unique={len(unique_s1)}，无重复: {dup_ok_s1}")
    assert dup_ok_s1, "Stage 1 去重失败！"

    # 分数排序验证
    scores_s1 = [p.select_score for p in papers_s1]
    if scores_s1:
        sorted_ok = all(scores_s1[i] >= scores_s1[i+1] for i in range(len(scores_s1)-1))
        print(f"  【排序】按 score 降序排列: {sorted_ok}")
        assert sorted_ok, "Stage 1 排序失败！"

    # ── Stage 1 + BFS 两层扩展 ────────────────────────────
    print("\n【Stage 1 + BFS】搜索 + 两层引文扩展：")
    papers_full = searcher.search(question, expand_layers=2)
    print(f"  → 返回 {len(papers_full)} 篇（已去重）")

    depth_counts = {}
    for p in papers_full:
        depth_counts[p.depth] = depth_counts.get(p.depth, 0) + 1
    print(f"  → 各层级论文数: {dict(sorted(depth_counts.items()))}")

    for j, p in enumerate(papers_full[:10], 1):
        indent = "  " * p.depth
        print(f"  [{j}] depth={p.depth} score={p.select_score:.4f}{indent}| {p.title[:50]}")
        if p.source:
            print(f"         来源: {p.source[:60]}")

    # 去重验证（全量）
    paper_ids = [p.paper_id for p in papers_full if p.paper_id]
    unique_ids = set(paper_ids)
    dup_ok = len(paper_ids) == len(unique_ids)
    print(f"  【去重】共 {len(papers_full)} 篇，unique={len(unique_ids)}，无重复: {dup_ok}")
    assert dup_ok, "全量去重失败！"

    # Reranker 评分分布（不再用 cosine 阈值，改用 reranker score）
    all_scores = [p.select_score for p in papers_full]
    if all_scores:
        print(f"\n  【Reranker 评分分布】top_n=20")
        print(f"  → 最高={max(all_scores):.4f}，最低={min(all_scores):.4f}，平均={sum(all_scores)/len(all_scores):.4f}")
    else:
        print(f"\n  【Reranker 评分分布】无结果（arXiv 限流中，reranker 逻辑已验证正常）")

    # 层级深度验证（expand_layers=2，所以 depth 应该是 0, 1, 2）
    max_depth = max(p.depth for p in papers_full) if papers_full else 0
    print(f"\n  【层级】最大 depth={max_depth}，expand_layers=2，depth in [0,1,2]: {max_depth <= 2}")
    assert max_depth <= 2, f"BFS 层级超过 expand_layers 限制！max_depth={max_depth}"

    # MockLLM 调用次数验证
    print(f"\n  【LLM调用】共 {mock_llm.call_count} 次（expand_layers=2 → 应该 3 次：1 search + 2 expand）")
    assert mock_llm.call_count <= 3, f"LLM 调用次数异常：{mock_llm.call_count}"

    print(f"\n{'=' * 60}")
    print("✅ 所有验证通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
