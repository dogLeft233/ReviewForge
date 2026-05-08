"""BFS 论文搜索器 — 仿 PaSa 架构，支持 reranker 或 embedding 相似度过滤

核心流程（双阶段 BFS）：
  Stage 1 (search):  用 LLM 生成多个搜索词 → 并行搜索 → reranker/embedding 过滤 → papers_queue
  Stage 2 (expand):  对每篇论文查引用关系 → reranker/embedding 过滤 → 推入下一层 papers_queue

过滤机制：
  - reranker 模式：重排序后取 top_n（优先）
  - embedding 模式：cosine similarity >= 0.50 才纳入

用法:
    from src.seacher.bfs_search import BFSSearcher
    from src.llm import LLM
    from src.embedding import EmbeddingClient
    from src.reranker import RerankerClient

    llm = LLM(api_key="...", model="Qwen/Qwen3-8B")
    embed = EmbeddingClient()
    reranker = RerankerClient()          # 优先使用 reranker
    searcher = BFSSearcher(llm=llm, embed=embed, reranker=reranker)

    results = searcher.search("LoRA 大模型微调")
    results = searcher.search("扩散模型图像生成", expand_layers=2)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── 超参数默认常量 ────────────────────────────────────────────
DEFAULT_SEARCH_QUERIES = 5       # LLM 生成多少个搜索词
DEFAULT_SEARCH_PAPERS = 20       # 每个搜索词取多少篇论文
DEFAULT_EXPAND_PAPERS = 15       # 每层最多扩展多少篇论文
DEFAULT_EXPAND_LAYERS = 2        # BFS 最大层数（0=只用搜索，1=搜索+一层引文）
DEFAULT_THREADS = 20             # 并行线程数
SIMILARITY_THRESHOLD = 0.50      # 嵌入相似度阈值

# ── 异常 ─────────────────────────────────────────────────────


class BFSSearchError(Exception):
    """BFS 搜索基异常"""


class RetrievalError(BFSSearchError):
    """检索失败"""


# ── 数据结构 ─────────────────────────────────────────────────


@dataclass(slots=True)
class SearchQuery:
    """LLM 生成的单个搜索词"""

    query: str           # 搜索词/句
    reason: str = ""    # 搜索理由


@dataclass(slots=True)
class PaperNode:
    """论文节点（BFS 树中的节点）"""

    title: str
    paper_id: str        # arXiv ID（去重 key）
    abstract: str = ""
    depth: int = 0      # BFS 层级（root=0，引用=1，引用-of引用=2...）
    select_score: float = 0.0  # 评分（reranker score 或 embedding cosine）
    child: dict[str, list["PaperNode"]] = field(default_factory=dict)  # section →子论文
    source: str = ""     # 来源描述
    cited_by: list["PaperNode"] = field(default_factory=list)  # 该论文的参考文献

    # 用于去重的哈希 key
    @property
    def dedup_key(self) -> str:
        return self.paper_id or self.title.lower()


# ── 核心搜索器 ────────────────────────────────────────────────


class BFSSearcher:
    """BFS 论文搜索引擎（PaSa 风格，reranker 替代 embedding 做相关性过滤）

    Attributes:
        llm: LLM 实例（用于生成搜索词）
        embed: 嵌入模型客户端（reranker 未提供时用于 cosine 过滤）
        reranker: 重排序客户端（优先；reranker 未提供则降级到 embedding）
        search_queries_count: 每轮生成多少个搜索词
        search_papers_count: 每个搜索词取多少篇论文
        expand_papers_count: 每层最多扩展多少篇论文
        expand_layers: BFS 最大层数
        similarity_threshold: 相似度过滤阈值（embedding 模式专用）
        rerank_top_n: 重排序后保留前 n 条（reranker 模式，None=全部）
    """

    def __init__(
        self,
        llm: Any,
        embed: Any,
        reranker: Any = None,
        *,
        search_queries_count: int = DEFAULT_SEARCH_QUERIES,
        search_papers_count: int = DEFAULT_SEARCH_PAPERS,
        expand_papers_count: int = DEFAULT_EXPAND_PAPERS,
        expand_layers: int = DEFAULT_EXPAND_LAYERS,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        rerank_top_n: int | None = 20,
        threads_num: int = DEFAULT_THREADS,
    ) -> None:
        self.llm = llm
        self.embed = embed
        self.reranker = reranker
        self.search_queries_count = search_queries_count
        self.search_papers_count = search_papers_count
        self.expand_papers_count = expand_papers_count
        self.expand_layers = expand_layers
        self.similarity_threshold = similarity_threshold
        self.rerank_top_n = rerank_top_n
        self.threads_num = threads_num

        logger.debug(
            "BFSSearcher init: embed=%s reranker=%s threshold=%.2f layers=%d",
            getattr(embed, "model", "?"),
            getattr(reranker, "model", "?") if reranker else None,
            similarity_threshold, expand_layers,
        )

    # ── 主入口 ──────────────────────────────────────────────

    def search(
        self,
        question: str,
        expand_layers: int | None = None,
    ) -> list[PaperNode]:
        """执行完整 BFS 搜索流程

        参数:
            question: 用户研究问题
            expand_layers: BFS 最大层数（None=用默认值）

        返回:
            按 select_score 排序的 PaperNode 列表
        """
        layers = expand_layers if expand_layers is not None else self.expand_layers

        # ── Stage 1: 生成搜索词 + 并行搜索 ──────────────────
        search_queries = self._generate_queries(question)
        logger.info("Stage 1: 生成了 %d 个搜索词", len(search_queries))

        papers_queue = self._search_all(search_queries)
        logger.info("Stage 1 完成: papers_queue 累计 %d 篇", len(papers_queue))

        # ── Stage 2: BFS 引文扩展 ────────────────────────────
        if layers > 0:
            papers_queue = self._bfs_expand(papers_queue, layers)

        # ── 全局去重 ─────────────────────────────────────────
        papers = self._deduplicate(papers_queue)

        # ── 按相似度排序 ─────────────────────────────────────
        papers.sort(key=lambda p: p.select_score, reverse=True)

        return papers

    # ── Stage 1: 查询生成 + 搜索 ────────────────────────────

    def _generate_queries(self, question: str) -> list[SearchQuery]:
        """调用 LLM 生成多个搜索词描述（单轮，无工具调用）"""
        prompt = (
            f"你是一名学术搜索研究员。根据研究问题，生成 {self.search_queries_count} 个搜索词/句，"
            f"覆盖不同搜索角度（宽泛/精准/作者/分类/最新趋势）。\n\n"
            f"研究问题：{question}\n\n"
            f"输出格式（每行一个）：SEARCH|搜索词|搜索理由\n"
            f"示例：\n"
            f"SEARCH|ti:LoRA fine-tuning large language models|精准匹配 LoRA 在 LLM 场景|"
        )

        reply = self.llm.chat(
            external_prompt=prompt,
            system_prompt="你只需输出搜索词，不要解释。",
            temperature=0.3,
            max_tokens=500,
        )

        queries: list[SearchQuery] = []
        for line in reply.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 2 and parts[0] == "SEARCH":
                queries.append(SearchQuery(query=parts[1].strip(), reason=parts[2].strip() if len(parts) > 2 else ""))
            elif line.startswith("SEARCH|"):
                # 兼容没有明确字段前缀的格式
                rest = line[7:].strip()
                queries.append(SearchQuery(query=rest, reason=""))

        if not queries:
            # 保底：用原问题作搜索词
            queries.append(SearchQuery(query=question, reason="原始问题"))
            logger.warning("LLM 未生成有效搜索词，用原始问题代替")

        return queries

    def _search_all(self, search_queries: list[SearchQuery]) -> list[PaperNode]:
        """并行执行所有搜索词，按阈值过滤后合并"""
        all_papers: list[PaperNode] = []

        with ThreadPoolExecutor(max_workers=self.threads_num) as pool:
            futures = {
                pool.submit(self._search_one_query, sq): sq
                for sq in search_queries
            }

            for future in as_completed(futures):
                sq = futures[future]
                try:
                    papers = future.result()
                    all_papers.extend(papers)
                    logger.debug("搜索词 '%s' → %d 篇论文通过阈值", sq.query, len(papers))
                except Exception as e:
                    logger.warning("搜索词 '%s' 失败: %s", sq.query, e)

        return all_papers

    def _search_one_query(self, sq: SearchQuery) -> list[PaperNode]:
        """对单个搜索词执行：retriever 搜索 → reranker/embedding 过滤"""
        from src.retrievers import ArxivRetriever

        try:
            with ArxivRetriever() as retriever:
                results = retriever.search(sq.query, max_results=self.search_papers_count)
        except Exception as e:
            raise RetrievalError(f"检索失败: {sq.query}") from e

        if not results:
            return []

        if self.reranker:
            return self._score_by_reranker_search(sq.query, results)
        return self._score_by_embedding(sq.query, results)

    def _score_by_reranker_search(
        self, query: str, results: list[Any]
    ) -> list[PaperNode]:
        """用 reranker 对搜索结果打分，取 top_n"""
        docs = [r.abstract or "" for r in results]
        reranked = self.reranker.rerank(
            query=query,
            documents=docs,
            top_n=self.rerank_top_n,
            return_documents=True,
        )
        paper_map = {i: r for i, r in enumerate(results)}
        nodes: list[PaperNode] = []
        for r in reranked:
            result = paper_map.get(r.index)
            if not result:
                continue
            nodes.append(PaperNode(
                title=result.title,
                paper_id=result.url or "",
                abstract=result.abstract or "",
                depth=0,
                select_score=r.relevance_score,
                source=f"Search: {query}",
            ))
        return nodes

    def _score_by_embedding(
        self, query: str, results: list[Any]
    ) -> list[PaperNode]:
        """用 embedding cosine 相似度过滤"""
        abstracts = [r.abstract or "" for r in results]
        query_vec = self.embed.encode(query)
        abstract_vecs = self.embed.encode_batch(abstracts)
        nodes: list[PaperNode] = []
        for result, abstract_vec in zip(results, abstract_vecs):
            score = self._cosine(query_vec, abstract_vec)
            if score >= self.similarity_threshold:
                nodes.append(PaperNode(
                    title=result.title,
                    paper_id=result.url or "",
                    abstract=result.abstract or "",
                    depth=0,
                    select_score=score,
                    source=f"Search: {query}",
                ))
        return nodes

    # ── Stage 2: BFS 引文扩展 ────────────────────────────────

    def _bfs_expand(self, initial_papers: list[PaperNode], layers: int) -> list[PaperNode]:
        """BFS 引文扩展（从 root 论文出发，逐层扩展引用关系）"""
        papers_queue: list[PaperNode] = list(initial_papers)
        all_papers: list[PaperNode] = list(initial_papers)
        visited_ids: set[str] = {p.dedup_key for p in papers_queue if p.paper_id}

        for depth in range(1, layers + 1):
            logger.info("BFS 扩展 depth=%d，当前队列 %d 篇", depth, len(papers_queue))

            # 取 top N 按 score 排序的论文扩展
            batch = sorted(papers_queue, key=lambda p: p.select_score, reverse=True)[: self.expand_papers_count]

            # 对 batch 每篇论文查引文
            next_batch: list[PaperNode] = []
            refs_nodes = self._fetch_refs_batch(batch)

            for parent, refs in zip(batch, refs_nodes):
                for ref_node in refs:
                    key = ref_node.dedup_key
                    if key and key not in visited_ids:
                        ref_node.depth = depth
                        ref_node.source = f"Expand[{parent.title[:30]}][{parent.dedup_key}]"
                        next_batch.append(ref_node)
                        visited_ids.add(key)
                        all_papers.append(ref_node)

            papers_queue = next_batch
            if not papers_queue:
                break

        return all_papers

    def _fetch_refs_batch(self, batch: list[PaperNode]) -> list[list[PaperNode]]:
        """并行获取一批论文的参考文献列表"""
        results: list[list[PaperNode]] = []

        with ThreadPoolExecutor(max_workers=self.threads_num) as pool:
            futures = {pool.submit(self._fetch_refs, p): p for p in batch}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.warning("获取参考文献失败: %s", e)
                    results.append([])

        return results

    def _fetch_refs(self, paper: PaperNode) -> list[PaperNode]:
        """获取单篇论文的参考文献节点

        实现策略（按优先级）:
        1. Semantic Scholar API — 真实引文关系图（优先，有 paper_id 时）
        2. arXiv 文本搜索 — 标题 + "related work citations"（降级备用，
           使用 reranker 或 embedding 过滤）
        """
        nodes: list[PaperNode] = []
        paper_id = paper.paper_id or ""

        # ── 策略 1：真实引文图（Semantic Scholar）─────────────
        if paper_id:
            raw_id = _extract_arxiv_id(paper.paper_id) if paper.paper_id else ""
            s2_id = f"ArXiv:{raw_id}" if raw_id and not raw_id.startswith("ArXiv:") else raw_id
            try:
                from src.retrievers import SemanticScholarRetriever

                with SemanticScholarRetriever() as s2:
                    refs = s2.get_references(s2_id, max_results=10)
                if refs:
                    for r in refs:
                        ref_arxiv = _extract_arxiv_id(r.url)
                        node = PaperNode(
                            title=r.title,
                            paper_id=ref_arxiv or r.url or r.title,
                            abstract=r.abstract or "",
                            depth=paper.depth + 1,
                            select_score=0.0,  # 引文关系不依赖评分
                            source=f"Ref: [{paper.title[:30]}]",
                        )
                        nodes.append(node)
                    logger.debug(
                        "论文 '%s' 借 S2 获取 %d 篇参考文献",
                        paper.title[:30], len(nodes),
                    )
                    return nodes
            except Exception as e:
                logger.debug(
                    "S2 引文查找失败 [%s]: %s，回退文本搜索", paper.title[:30], e
                )

        # ── 策略 2：降级文本搜索（保持向后兼容）──────────────
        try:
            from src.retrievers import ArxivRetriever

            with ArxivRetriever() as retriever:
                related = retriever.search(
                    f"{paper.title} related work citations",
                    max_results=10,
                )
        except Exception as e:
            logger.debug("论文 '%s' 参考文献搜索失败: %s", paper.title[:30], e)
            return []

        if not related:
            return []

        if self.reranker:
            nodes = self._score_refs_by_reranker(paper, related)
        else:
            nodes = self._score_refs_by_embedding(paper, related)

        time.sleep(3)
        return nodes

    def _score_refs_by_reranker(
        self, paper: PaperNode, related: list[Any]
    ) -> list[PaperNode]:
        """用 reranker 打分参考文献"""
        docs = [r.abstract or "" for r in related]
        reranked = self.reranker.rerank(
            query=paper.title,
            documents=docs,
            top_n=self.rerank_top_n,
            return_documents=True,
        )
        paper_map = {i: r for i, r in enumerate(related)}
        nodes: list[PaperNode] = []
        for r in reranked:
            result = paper_map.get(r.index)
            if not result:
                continue
            ref_arxiv = _extract_arxiv_id(result.url)
            nodes.append(PaperNode(
                title=result.title,
                paper_id=ref_arxiv or result.url or result.title,
                abstract=result.abstract or "",
                depth=paper.depth + 1,
                select_score=r.relevance_score,
                source=f"Ref: [{paper.title[:30]}]",
            ))
        return nodes

    def _score_refs_by_embedding(
        self, paper: PaperNode, related: list[Any]
    ) -> list[PaperNode]:
        """用 embedding cosine 相似度过滤参考文献"""
        abstracts = [r.abstract or "" for r in related]
        paper_vec = self.embed.encode(paper.title)
        nodes: list[PaperNode] = []
        for r, abstract_vec in zip(related, self.embed.encode_batch(abstracts)):
            score = self._cosine(paper_vec, abstract_vec)
            if score >= self.similarity_threshold:
                ref_arxiv = _extract_arxiv_id(r.url)
                nodes.append(PaperNode(
                    title=r.title,
                    paper_id=ref_arxiv or r.url or r.title,
                    abstract=r.abstract or "",
                    depth=paper.depth + 1,
                    select_score=score,
                    source=f"Ref: [{paper.title[:30]}]",
                ))
        return nodes

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _deduplicate(papers: list[PaperNode]) -> list[PaperNode]:
        """全局去重（按 dedup_key）"""
        seen: dict[str, PaperNode] = {}
        for p in papers:
            key = p.dedup_key
            if key not in seen or p.select_score > seen[key].select_score:
                seen[key] = p
        return list(seen.values())

    # ── 上下文管理 ───────────────────────────────────────────

    def __enter__(self) -> "BFSSearcher":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ── 工具函数 ──────────────────────────────────────────────────


def _extract_arxiv_id(url: str) -> str:
    """从 URL 中提取 arXiv ID

    支持格式:
      - https://arxiv.org/abs/2301.00001
      - https://arxiv.org/abs/2301.00001v2
      - http://arxiv.org/abs/2301.00001
      - 2301.00001
    """
    if not url:
        return ""
    import re
    m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]+)", url)
    return m.group(1) if m else ""
