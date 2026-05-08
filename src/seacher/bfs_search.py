"""BFS 论文搜索器 — 仿 PaSa 架构，嵌入模型做相关性过滤

核心流程（双阶段 BFS）：
  Stage 1 (search):  用 LLM 生成多个搜索词 → 并行搜索 → 嵌入相似度过滤 → papers_queue
  Stage 2 (expand):  对每篇论文查引用关系 → 嵌入相似度过滤 → 推入下一层 papers_queue

过滤机制：嵌入模型编码 query + abstract → cosine similarity >= 0.50 才纳入

用法:
    from src.seacher.bfs_search import BFSSearcher
    from src.llm import LLM
    from src.embedding import EmbeddingClient

    llm = LLM(api_key="...", model="Qwen/Qwen3-8B")
    embed = EmbeddingClient()
    searcher = BFSSearcher(llm=llm, embed=embed)

    # 单次搜索
    results = searcher.search("LoRA 大模型微调")

    # BFS 引文扩展（depth=1/2）
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
    select_score: float = 0.0  # 嵌入相似度分数
    child: dict[str, list["PaperNode"]] = field(default_factory=dict)  # section →子论文
    source: str = ""     # 来源描述
    cited_by: list["PaperNode"] = field(default_factory=list)  # 该论文的参考文献

    # 用于去重的哈希 key
    @property
    def dedup_key(self) -> str:
        return self.paper_id or self.title.lower()


# ── 核心搜索器 ────────────────────────────────────────────────


class BFSSearcher:
    """BFS 论文搜索引擎（PaSa 风格，嵌入模型替代 LLM 分类器）

    Attributes:
        llm: LLM 实例（用于生成搜索词）
        embed: 嵌入模型客户端（用于计算相似度）
        search_queries_count: 每轮生成多少个搜索词
        search_papers_count: 每个搜索词取多少篇论文
        expand_papers_count: 每层最多扩展多少篇论文
        expand_layers: BFS 最大层数
        similarity_threshold: 相似度过滤阈值
    """

    def __init__(
        self,
        llm: Any,
        embed: Any,
        *,
        search_queries_count: int = DEFAULT_SEARCH_QUERIES,
        search_papers_count: int = DEFAULT_SEARCH_PAPERS,
        expand_papers_count: int = DEFAULT_EXPAND_PAPERS,
        expand_layers: int = DEFAULT_EXPAND_LAYERS,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        threads_num: int = DEFAULT_THREADS,
    ) -> None:
        self.llm = llm
        self.embed = embed
        self.search_queries_count = search_queries_count
        self.search_papers_count = search_papers_count
        self.expand_papers_count = expand_papers_count
        self.expand_layers = expand_layers
        self.similarity_threshold = similarity_threshold
        self.threads_num = threads_num

        logger.debug(
            "BFSSearcher init: embed_model=%s threshold=%.2f layers=%d",
            getattr(embed, "model", "?"), similarity_threshold, expand_layers,
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
        """对单个搜索词执行：retriever 搜索 → 嵌入过滤"""
        from src.retrievers import ArxivRetriever

        papers: list[PaperNode] = []
        try:
            with ArxivRetriever() as retriever:
                results = retriever.search(sq.query, max_results=self.search_papers_count)
        except Exception as e:
            raise RetrievalError(f"检索失败: {sq.query}") from e

        if not results:
            return []

        # 批量编码 abstracts
        titles = [r.title for r in results]
        abstracts = [r.abstract or "" for r in results]
        query_vec = self.embed.encode(sq.query)
        abstract_vecs = self.embed.encode_batch(abstracts)

        for r, abstract_vec in zip(results, abstract_vecs):
            score = self._cosine(query_vec, abstract_vec)
            if score >= self.similarity_threshold:
                node = PaperNode(
                    title=r.title,
                    paper_id=r.url or "",   # arxiv retriever stores arxiv_id in url field
                    abstract=r.abstract or "",
                    depth=0,
                    select_score=score,
                    source=f"Search: {sq.query}",
                )
                papers.append(node)

        return papers

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
        2. arXiv 文本搜索 — 标题 + "related work citations"（降级备用）
        """
        nodes: list[PaperNode] = []
        paper_id = paper.paper_id or ""


        # ── 策略 1：真实引文图（Semantic Scholar）─────────────
        if paper_id:
            # paper_id 形如 "2301.00001"（无前缀），S2 API 需要 "arxiv:2301.00001"
            raw_id = _extract_arxiv_id(paper.paper_id) if paper.paper_id else ""
            s2_id = f"ArXiv:{raw_id}" if raw_id and not raw_id.startswith("ArXiv:") else raw_id
            try:
                from src.retrievers import SemanticScholarRetriever

                with SemanticScholarRetriever() as s2:
                    refs = s2.get_references(s2_id, max_results=10)
                if refs:
                    for r in refs:
                        # 取 arXiv ID 作为 dedup key
                        ref_arxiv = _extract_arxiv_id(r.url)
                        node = PaperNode(
                            title=r.title,
                            paper_id=ref_arxiv or r.url or r.title,
                            abstract=r.abstract or "",
                            depth=paper.depth + 1,
                            select_score=0.0,  # 引文关系不依赖 embedding score
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

        # 批量编码 + 过滤
        abstracts = [r.abstract or "" for r in related]
        paper_vec = self.embed.encode(paper.title)

        for r, abstract_vec in zip(related, self.embed.encode_batch(abstracts)):
            score = self._cosine(paper_vec, abstract_vec)
            if score >= self.similarity_threshold:
                ref_arxiv = _extract_arxiv_id(r.url)
                node = PaperNode(
                    title=r.title,
                    paper_id=ref_arxiv or r.url or r.title,
                    abstract=r.abstract or "",
                    depth=paper.depth + 1,
                    select_score=score,
                    source=f"Ref: [{paper.title[:30]}]",
                )
                nodes.append(node)


        # 控制速率（arXiv API honor system）
        time.sleep(3)
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
