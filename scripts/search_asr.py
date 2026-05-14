#!/usr/bin/env python3
"""ASR 自动语音识别文献全面搜索脚本

覆盖：
- 经典 Hidden Markov Model / GMM-HMM 时代（90年代-2010年代）
- 深度学习时代（DNN、CNN、LCN、Attention、Transformer）
- 端到端模型（CTC、RNN-T、Transducer）
- 前沿进展（Conformer、Whisper、WeNet、SenseVoice、WavLM、Wav2Vec2.0 等）
- 中文 ASR 与跨语言模型
- 工业级部署实践

搜索策略：
- Stage 1: 20+ 个搜索词，覆盖经典 / 深度学习 / 端到端 / 前沿
- Stage 2: 3 层 BFS 扩展，最大获取引用关系
- Stage 3: reranker 重排序 + 阈值过滤，保留 top 100
"""

import sys, os, logging
from pathlib import Path

# ── 路径设置 ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ["PYTHONPATH"] = str(_ROOT) + "/src"

logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
for _name in ("seacher", "bfs_search", "retrievers", "embedding", "reranker",
              "src.seacher.bfs_search", "src.retrievers", "src.embedding", "src.reranker"):
    logging.getLogger(_name).setLevel(logging.INFO)

# ── LLM 配置 ────────────────────────────────────────────────
# 使用硅基流动 API（性价比高，支持 Qwen3-8B）
LLM_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
LLM_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_MODEL   = os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B")

if not LLM_API_KEY:
    raise RuntimeError("请设置环境变量 SILICONFLOW_API_KEY")

from src.llm import LLM
from src.embedding import EmbeddingClient
from src.reranker import RerankerClient
from src.seacher.bfs_search import BFSSearcher


def build_searcher():
    llm = LLM(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, model=LLM_MODEL)
    embed = EmbeddingClient()
    reranker = RerankerClient()

    # ── 搜索参数说明 ─────────────────────────────────────
    # search_queries_count=25 : LLM 生成 25 个搜索词，最大化文献覆盖面
    # search_papers_count=30  : 每个搜索词取 30 篇（尽量多）
    # expand_papers_count=30  : 每层扩展 30 篇（增加引文覆盖率）
    # expand_layers=3        : BFS 三层（depth=0,1,2,3），兼顾深度与广度
    # similarity_threshold=0.50 : embedding 模式阈值（reranker 模式下不生效）
    # rerank_top_n=100        : reranker 打分后取 top 100（大幅放宽）
    # threads_num=20          : 并行 20 线程，加速检索
    return BFSSearcher(
        llm=llm,
        embed=embed,
        reranker=reranker,
        search_queries_count=3,
        search_papers_count=5,
        expand_papers_count=5,
        expand_layers=1,
        score_threshold=0.20,
        rerank_top_n=100,
        threads_num=1,
    )


QUESTION = (
    "ASR 自动语音识别，要求同时关注经典的方法和优质的前沿方法"
)


def main():
    searcher = build_searcher()

    print("=" * 70)
    print("ASR 自动语音识别文献全面搜索")
    print("=" * 70)
    print(f"\n搜索器配置:")
    print(f"  search_queries_count : {searcher.search_queries_count}")
    print(f"  search_papers_count  : {searcher.search_papers_count}")
    print(f"  expand_papers_count : {searcher.expand_papers_count}")
    print(f"  expand_layers       : {searcher.expand_layers}")
    print(f"  rerank_top_n        : {searcher.rerank_top_n}")
    print(f"  similarity_threshold: {searcher.score_threshold}")
    print(f"  threads_num         : {searcher.threads_num}")

    print(f"\n搜索问题:\n  {QUESTION}\n")

    print("开始搜索（预计 3-10 分钟）...\n")
    results = searcher.search(QUESTION, expand_layers=3)

    print(f"\n{'=' * 70}")
    print(f"搜索完成，共返回 {len(results)} 篇文献\n")

    # ── 按 depth 分组统计 ─────────────────────────────────
    depth_counts = {}
    for p in results:
        depth_counts[p.depth] = depth_counts.get(p.depth, 0) + 1
    print(f"各层级论文数: {dict(sorted(depth_counts.items()))}")
    print("(depth=0: 搜索结果，depth=1: 直接引用，depth=2: 引用-of引用，以此类推)")

    # ── 按 score 排序输出 ─────────────────────────────────
    results_sorted = sorted(results, key=lambda p: p.select_score, reverse=True)

    print(f"\n{'=' * 70}")
    print("Top 50 高分论文（按 reranker score 降序）:\n")
    for i, p in enumerate(results_sorted[:50], 1):
        indent = "  " * p.depth
        print(f"[{i:2d}] score={p.select_score:.4f} | depth={p.depth}{indent}")
        print(f"      {p.title[:65]}")
        if p.paper_id:
            print(f"      ID: {p.paper_id}")
        if p.source:
            print(f"      来源: {p.source[:60]}")
        print()

    # ── 输出完整结果到 JSON ───────────────────────────────
    import json, datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _ROOT / f"asr_search_results_{ts}.json"
    jsonable = [
        {
            "title": p.title,
            "paper_id": p.paper_id,
            "depth": p.depth,
            "select_score": p.select_score,
            "source": p.source,
            "abstract": p.abstract[:200] if p.abstract else "",
        }
        for p in results_sorted
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(jsonable, f, ensure_ascii=False, indent=2)
    print(f"完整结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
