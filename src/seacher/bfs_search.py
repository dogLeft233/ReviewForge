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
DEFAULT_THREADS = 1             # 并行线程数
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
        logger.info("expand_layers config=%s → actual layers=%d (expand_papers_count=%d)",
                    self.expand_layers, layers, self.expand_papers_count)
        if layers > 0:
            logger.info("BFS 扩展层数 depth in [0, %d]，expand_papers_count=%d", layers, self.expand_papers_count)

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
            f"You are an academic search researcher. Generate {self.search_queries_count} search queries for arXiv, "
            f"covering different angles (broad/precise/author/classification/latest trends).\n\n"
            f"Research topic: {question}\n\n"
            f"Output format (one per line): SEARCH|<English search query>|<English reason>\n"
            f"Examples:\n"
            f"SEARCH|ti:LoRA fine-tuning large language models|match LoRA in LLMs|\n"
            f"SEARCH|attention mechanism transformer ASR end-to-end|precise match for ASR architectures|\n"
            f"SEARCH|author:Geoffrey Hinton speech recognition neural networks|author authority|\n"
            f"SEARCH|conversational AI automatic speech recognition 2024|latest trends|\n"
            f"IMPORTANT: All search queries MUST be in English. Do NOT output Chinese queries."
        )

        reply = self.llm.chat(
            external_prompt=prompt,
            system_prompt="You are a researcher. Output ONLY the search lines, nothing else. Each line must start with SEARCH|.",
            temperature=0.3,
            max_tokens=500,
        )

        queries: list[SearchQuery] = []
        seen: set[str] = set()
        dropped: list[str] = []
        for line in reply.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 2 and parts[0] == "SEARCH":
                q = parts[1].strip()
                reason = parts[2].strip() if len(parts) > 2 else ""
                if q in seen:
                    dropped.append(f"跳过（重复）: {line}")
                else:
                    seen.add(q)
                    queries.append(SearchQuery(query=q, reason=reason))
            elif line.startswith("SEARCH|"):
                # 兼容没有明确字段前缀的格式
                rest = line[7:].strip()
                if rest in seen:
                    dropped.append(f"跳过（重复）: {line}")
                else:
                    seen.add(rest)
                    queries.append(SearchQuery(query=rest, reason=""))
            else:
                dropped.append(f"跳过（格式不符）: {line}")


        for d in dropped:
            logger.debug("query 丢弃: %s", d)
        logger.info("Stage 1: 生成 %d/%d 个搜索词（丢弃 %d 个）", len(queries), len(queries), len(dropped))

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
        """对单个搜索词执行：SerpAPIRetriever → arXiv HTML 直接抓取详情

        Stage 1: SerpAPIRetriever (SerpAPI Google 搜索) 提取 arXiv ID
        Stage 2: 直接抓取 https://arxiv.org/abs/{id} HTML 页面（无 rate limit）
        """
        import re as re_module
        import httpx

        try:
            # Stage 1: SerpAPI Google 搜索提取 arXiv ID
            from src.retrievers import SerpAPIRetriever
            with SerpAPIRetriever() as serp:
                papers = serp.search(sq.query, max_results=self.search_papers_count)

            if not papers:
                return []

            # Stage 2: 直接抓取 arXiv abstract page 获取详情（无 rate limit）
            from src.models import PaperCard
            enriched = []
            for p in papers:
                m = re_module.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]+)", p.url)
                if not m:
                    continue
                aid = m.group(1)
                try:
                    resp = httpx.get(f"https://arxiv.org/abs/{aid}", timeout=15, follow_redirects=True)
                    if resp.status_code != 200:
                        continue
                    html = resp.text
                    title_m = re_module.search(r"<title>\[[^\]]+\]\s*(.*?)</title>", html)
                    title = title_m.group(1).strip() if title_m else p.title
                    abstract_m = re_module.search(
                        r'class="abstract mathjax">(.*?)</blockquote>', html, re_module.DOTALL
                    )
                    # Strip the descriptor span "Abstract:" and remaining tags
                    abstract_raw = abstract_m.group(1) if abstract_m else ""
                    abstract_raw = re_module.sub(r"<span[^>]*>Abstract:</span>\s*", "", abstract_raw)
                    abstract = re_module.sub(r"<[^>]+>", "", abstract_raw).strip()
                    if not abstract and p.abstract:
                        abstract = p.abstract
                    author_m = re_module.search(r'class="authors">([^<]+)', html)
                    authors_str = author_m.group(1).strip() if author_m else ""
                    authors = [a.strip() for a in authors_str.split(",")] if authors_str else []
                    enriched.append(PaperCard(
                        title=title,
                        authors=authors,
                        year=0,
                        abstract=abstract[:800],
                        url=f"https://arxiv.org/abs/{aid}",
                        source="serpapi+arxiv",
                        method_category="",
                    ))
                except Exception as e:
                    logger.debug("arXiv HTML fetch failed %s: %s", aid, e)
                    continue

            if not enriched:
                return []

            if self.reranker:
                return self._score_by_reranker_search(sq.query, enriched)
            if self.embed:
                return self._score_by_embedding(sq.query, enriched)
            # Fallback: no reranker/embed → return all with default score
            nodes: list[PaperNode] = []
            for i, r in enumerate(enriched):
                nodes.append(PaperNode(
                    title=r.title,
                    paper_id=r.url or "",
                    abstract=r.abstract or "",
                    depth=0,
                    select_score=1.0 / (i + 1),  # 递减顺序
                    source=f"Search: {sq.query}",
                ))
            return nodes
        except Exception as e:
            raise RetrievalError(f"搜索失败 [{sq.query}]: {e}") from e

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
        0. ar5iv.org — 抓取 HTML 全文，解析 \\cite{} 引用（PaSa 架构，最优先）
        1. Semantic Scholar API — 真实引文关系图（降级）
        2. arXiv 文本搜索 — 标题 + "related work citations"（最后降级）
        """
        import re as re_module
        paper_id = paper.paper_id or ""

        # ── 策略 0：ar5iv HTML 解析引用（PaSa 架构，最优先）────────
        if paper_id:
            raw_id = _extract_arxiv_id(paper.paper_id) if paper.paper_id else ""
            if raw_id:
                try:
                    from src.retrievers import Ar5ivRetriever
                    with Ar5ivRetriever() as ar5iv:
                        html = ar5iv.fetch_full_text(raw_id)

                    if html:
                        # 解析 ar5iv HTML 中的参考文献条目
                        # ar5iv 使用 <li id="bib.bib{N}"> 格式，每个条目包含作者、年份、标题
                        bib_items = re_module.findall(
                            r'<li id="bib\.bib(\d+)" class="ltx_bibitem">(.*?)</li>',
                            html, re_module.DOTALL
                        )

                        ref_titles: list[str] = []
                        for bid, content_block in bib_items[:15]:
                            # 提取标题：第二个 ltx_bibblock（第一个是作者行）
                            blocks = re_module.findall(
                                r'class="ltx_bibblock">(.*?)<', content_block, re_module.DOTALL
                            )
                            if len(blocks) > 1:
                                title = re_module.sub(r'<[^>]+>', '', blocks[1]).strip()
                                if title:
                                    ref_titles.append(title)

                        if ref_titles:
                            nodes = self._fetch_refs_by_titles(ref_titles, paper)
                            if nodes:
                                logger.debug(
                                    "论文 '%s' 借 ar5iv 解析 %d 篇参考文献",
                                    paper.title[:30], len(nodes),
                                )
                                time.sleep(0.5)
                                return nodes
                except Exception as e:
                    logger.debug("ar5iv 解析失败 [%s]: %s，回退 S2", paper.title[:30], e)

        # ── 策略 1：Semantic Scholar 真实引文图 ──────────────────
        if paper_id:
            raw_id = _extract_arxiv_id(paper.paper_id) if paper.paper_id else ""
            s2_id = f"ArXiv:{raw_id}" if raw_id and not raw_id.startswith("ArXiv:") else raw_id
            try:
                from src.retrievers import SemanticScholarRetriever

                with SemanticScholarRetriever() as s2:
                    refs = s2.get_references(s2_id, max_results=10)
                if refs:
                    nodes = []
                    for r in refs:
                        ref_arxiv = _extract_arxiv_id(r.url)
                        node = PaperNode(
                            title=r.title,
                            paper_id=ref_arxiv or r.url or r.title,
                            abstract=r.abstract or "",
                            depth=paper.depth + 1,
                            select_score=0.0,
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

        # ── 策略 2：降级文本搜索 ─────────────────────────────────
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
        elif self.embed:
            nodes = self._score_refs_by_embedding(paper, related)
        else:
            # Fallback: no reranker/embed → return all with default score
            nodes: list[PaperNode] = []
            for r in related:
                ref_arxiv = _extract_arxiv_id(r.url)
                nodes.append(PaperNode(
                    title=r.title,
                    paper_id=ref_arxiv or r.url or r.title,
                    abstract=r.abstract or "",
                    depth=paper.depth + 1,
                    select_score=paper.select_score * 0.9,
                    source=f"Ref: [{paper.title[:30]}]",
                ))

        time.sleep(3)
        return nodes


    def _fetch_refs_by_titles(
        self, ref_titles: list[str], parent: PaperNode
    ) -> list[PaperNode]:
        """根据论文标题列表，用 SerpAPI 搜索获取 arXiv ID，再获取详情"""
        import re as re_module
        import httpx
        from src.retrievers import SerpAPIRetriever

        nodes: list[PaperNode] = []
        for title in ref_titles:
            try:
                with SerpAPIRetriever() as serp:
                    papers = serp.search(f"{title} site:arxiv.org", max_results=1)

                if not papers:
                    continue

                aid = None
                for p in papers:
                    m = re_module.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]+)", p.url)
                    if m:
                        aid = m.group(1)
                        break

                if not aid:
                    continue

                resp = httpx.get(f"https://arxiv.org/abs/{aid}", timeout=15, follow_redirects=True)
                if resp.status_code != 200:
                    continue
                html = resp.text
                title_m = re_module.search(r"<title>\[[^\]]+\]\s*(.*?)</title>", html)
                paper_title = title_m.group(1).strip() if title_m else title
                abstract_m = re_module.search(
                    r'class="abstract mathjax">(.*?)</blockquote>', html, re_module.DOTALL
                )
                abstract_raw = abstract_m.group(1) if abstract_m else ""
                abstract_raw = re_module.sub(r"<span[^>]*>Abstract:</span>\s*", "", abstract_raw)
                abstract = re_module.sub(r"<[^>]+>", "", abstract_raw).strip()
                nodes.append(PaperNode(
                    title=paper_title,
                    paper_id=aid,
                    abstract=abstract[:800],
                    depth=parent.depth + 1,
                    select_score=parent.select_score * 0.9,
                    source=f"ar5iv Ref: [{parent.title[:30]}]",
                ))
                time.sleep(0.5)
            except Exception as e:
                logger.debug("引用扩展失败 [%s]: %s", title[:30], e)
                continue
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
    """从 URL 或纯 ID 字符串中提取 arXiv ID

    支持格式:
      - https://arxiv.org/abs/2301.00001
      - https://arxiv.org/abs/2301.00001v2
      - ArXiv:2501.08008 / arXiv:2411.04358v1（带前缀纯ID）
      - 2301.00001 / 2501.08008（纯数字格式）
    """
    if not url:
        return ""
    import re
    # 去除常见前缀标签（ArXiv: / arXiv: / arxiv:）
    cleaned = re.sub(r"^(arxiv|ArXiv|arXiv):", "", url).strip()
    # 先尝试 URL 提取
    m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]+)", cleaned)
    if m:
        return m.group(1)
    # 纯数字 ID（如 "2501.08008" 或 "2411.04358v1"）
    m = re.match(r"^(\d{4}\.\d+)", cleaned)
    if m:
        return m.group(1)
    return ""
