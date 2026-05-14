"""PaSa 风格 BFS 论文搜索器 — 仿 bytedance/pasa 架构

核心流程（双阶段 BFS）：
  Stage 1 (search):  LLM 生成搜索词 → SerpAPI Google 搜索 → ar5iv 补充摘要 → reranker 过滤
  Stage 2 (expand):  ar5iv 获取 bibliography → SerpAPI 标题搜索找 arXiv ID → reranker 过滤

过滤机制：
  - reranker 模式：reranker 打分，score >= score_threshold 才纳入

特点（相比原版）：
  - 纯 SerpAPI + ar5iv，不依赖 Semantic Scholar API
  - SerpAPI 使用普通 Google 引擎 + site:arxiv.org（PaSa 风格）
  - ar5iv 提供完整 bibliography，支持精准的引用扩展

用法:
    from src.seacher.bfs_search import BFSSearcher
    from src.llm import LLM
    from src.reranker import RerankerClient

    llm = LLM(api_key="...", model="Qwen/Qwen3-8B")
    reranker = RerankerClient()
    searcher = BFSSearcher(llm=llm, reranker=reranker)
    results = searcher.search("LoRA fine-tuning", expand_layers=2)
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from src.explorer.explorer_report import ExplorerReport
from src.retrievers.ar5iv import Ar5ivRetriever
from src.retrievers.serpapi import SerpAPIRetriever

logger = logging.getLogger(__name__)

# ── 超参数默认常量 ────────────────────────────────────────────
DEFAULT_SEARCH_QUERIES = 5       # LLM 生成多少个搜索词
DEFAULT_SEARCH_PAPERS = 20      # 每个搜索词取多少篇论文
DEFAULT_EXPAND_PAPERS = 15      # 每层最多扩展多少篇论文
DEFAULT_EXPAND_LAYERS = 2       # BFS 最大层数（0=只用搜索，1=搜索+一层引文，2=两层）
DEFAULT_THREADS = 4             # 并行线程数
DEFAULT_SCORE_THRESHOLD = 0.3   # reranker score 阈值（0.5 太严格，0.3 更实用）


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
    reason: str = ""     # 搜索理由


@dataclass(slots=True)
class PaperNode:
    """论文节点（BFS 树中的节点）"""
    title: str
    paper_id: str        # arXiv ID（去重 key）
    abstract: str = ""
    depth: int = 0      # BFS 层级（root=0，引用=1，引用-of引用=2...）
    select_score: float = 0.0  # reranker relevance score
    source: str = ""    # 来源描述
    cited_by: list = field(default_factory=list)  # 该论文的参考文献（兼容）

    @property
    def dedup_key(self) -> str:
        return self.paper_id or self.title.lower()


# ── 核心搜索器 ────────────────────────────────────────────────


class BFSSearcher:
    """PaSa 风格 BFS 论文搜索引擎

    用 reranker 替代 PaSa 的 LLM Selector，实现纯 SerpAPI + ar5iv 架构。

    Attributes:
        llm: LLM 实例（仅用于生成搜索词）
        embed: 嵌入模型（reranker 未提供时的 fallback）
        reranker: 重排序客户端（必须，用于替代 LLM Selector）
        search_queries_count: 每轮生成多少个搜索词
        search_papers_count: 每个搜索词取多少篇论文
        expand_papers_count: 每层最多扩展多少篇论文
        expand_layers: BFS 最大层数
        score_threshold: reranker 过滤阈值（默认 0.5）
        rerank_top_n: reranker 返回前 n 条（None=全部）
        threads_num: 并行线程数
    """

    def __init__(
        self,
        llm: Any,
        embed: Any = None,
        reranker: Any = None,
        *,
        search_queries_count: int = DEFAULT_SEARCH_QUERIES,
        search_papers_count: int = DEFAULT_SEARCH_PAPERS,
        expand_papers_count: int = DEFAULT_EXPAND_PAPERS,
        expand_layers: int = DEFAULT_EXPAND_LAYERS,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        rerank_top_n: int | None = None,
        threads_num: int = DEFAULT_THREADS,
    ) -> None:
        self.llm = llm
        self.embed = embed
        self.reranker = reranker
        self.search_queries_count = search_queries_count
        self.search_papers_count = search_papers_count
        self.expand_papers_count = expand_papers_count
        self.expand_layers = expand_layers
        self.score_threshold = score_threshold
        self.rerank_top_n = rerank_top_n
        self.threads_num = threads_num

        logger.debug(
            "BFSSearcher(PaSa): reranker=%s threshold=%.2f layers=%d",
            getattr(reranker, "model", "?") if reranker else None,
            score_threshold, expand_layers,
        )

    # ── 主入口 ──────────────────────────────────────────────

    def search(
        self,
        question: str,
        expand_layers: int | None = None,
    ) -> list[PaperNode]:
        """执行完整 PaSa BFS 搜索流程

        Args:
            question: 用户研究问题
            expand_layers: BFS 最大层数（None=用默认值）

        Returns:
            按 select_score 排序的 PaperNode 列表
        """
        layers = expand_layers if expand_layers is not None else self.expand_layers
        logger.info(
            "PaSa BFS: question='%s' layers=%d expand_papers=%d threshold=%.2f",
            question, layers, self.expand_papers_count, self.score_threshold,
        )

        # ── Stage 1: 搜索阶段 ────────────────────────────────
        t0 = time.time()
        search_queries = self._generate_queries(question)
        logger.info("[Stage 1] 生成了 %d 个搜索词", len(search_queries))
        query_strs = [sq.query for sq in search_queries]

        stage1_papers = self._search_all_pasa(question, query_strs)
        logger.info("[Stage 1] 搜索阶段完成: %d 篇论文（耗时 %.1fs）", len(stage1_papers), time.time() - t0)

        if not stage1_papers:
            logger.warning("[Stage 1] 未找到任何论文，请检查 SerpAPI / ar5iv 连接")
            return []

        # ── Stage 2: BFS 引文扩展 ────────────────────────────
        if layers > 0:
            t1 = time.time()
            all_papers = self._bfs_expand(stage1_papers, layers, question)
            logger.info("[Stage 2] BFS 扩展完成: %d 篇论文（耗时 %.1fs）", len(all_papers), time.time() - t1)
        else:
            all_papers = list(stage1_papers)

        # ── 全局去重 + 排序 ──────────────────────────────────
        papers = self._deduplicate(all_papers)
        papers.sort(key=lambda p: p.select_score, reverse=True)
        logger.info("[Done] 共 %d 篇论文（去重后）", len(papers))
        return papers

    def search_with_explorer_report(
        self,
        question: str,
        explorer_report: ExplorerReport,
        expand_layers: int | None = None,
    ) -> list[PaperNode]:
        """使用 ExplorerReport 领域知识执行 BFS 搜索

        Args:
            question: 用户研究问题
            explorer_report: ExplorerAgent 返回的领域探索报告
            expand_layers: BFS 最大层数（None=用默认值）

        Returns:
            按 select_score 排序的 PaperNode 列表
        """
        layers = expand_layers if expand_layers is not None else self.expand_layers
        logger.info(
            "PaSa BFS with ExplorerReport: question='%s' layers=%d expand_papers=%d threshold=%.2f",
            question, layers, self.expand_papers_count, self.score_threshold,
        )

        # ── Stage 1: 搜索阶段（使用 ExplorerReport 增强查询）──────────
        t0 = time.time()
        search_queries = self._generate_queries(question, explorer_report)
        logger.info("[Stage 1] 生成了 %d 个搜索词（基于 ExplorerReport）", len(search_queries))
        query_strs = [sq.query for sq in search_queries]

        stage1_papers = self._search_all_pasa(question, query_strs)
        logger.info("[Stage 1] 搜索阶段完成: %d 篇论文（耗时 %.1fs）", len(stage1_papers), time.time() - t0)

        if not stage1_papers:
            logger.warning("[Stage 1] 未找到任何论文，请检查 SerpAPI / ar5iv 连接")
            return []

        # ── Stage 2: BFS 引文扩展 ────────────────────────────
        if layers > 0:
            t1 = time.time()
            all_papers = self._bfs_expand(stage1_papers, layers, question)
            logger.info("[Stage 2] BFS 扩展完成: %d 篇论文（耗时 %.1fs）", len(all_papers), time.time() - t1)
        else:
            all_papers = list(stage1_papers)

        # ── 全局去重 + 排序 ──────────────────────────────────
        papers = self._deduplicate(all_papers)
        papers.sort(key=lambda p: p.select_score, reverse=True)
        logger.info("[Done] 共 %d 篇论文（去重后）", len(papers))
        return papers

    # ── Stage 1: 查询生成 ───────────────────────────────────

    def _generate_queries(self, question: str, explorer_report: ExplorerReport | None = None) -> list[SearchQuery]:
        """调用 LLM 生成多个搜索词（PaSa Crawler 的第一步）

        Args:
            question: 用户研究问题
            explorer_report: 可选的 Explorer 报告，提供领域知识以生成更精准的查询词
        """
        # 构建领域知识上下文
        domain_context = ""
        if explorer_report:
            sections = []

            if explorer_report.stage1_overview:
                sections.append(f"## Domain Overview\n{explorer_report.stage1_overview[:500]}")

            if explorer_report.stage1_concepts:
                concepts = ", ".join(explorer_report.stage1_concepts[:10])
                sections.append(f"## Core Concepts\n{concepts}")

            if explorer_report.stage2_classics:
                classic_titles = [f"{c.title} ({c.year})" for c in explorer_report.stage2_classics[:5]]
                sections.append(f"## Classic Works\n" + "; ".join(classic_titles))

            if explorer_report.stage3_benchmarks:
                benchmarks = [b.name for b in explorer_report.stage3_benchmarks[:5]]
                sections.append(f"## Important Benchmarks\n" + ", ".join(benchmarks))

            if explorer_report.stage3_trends:
                trends = ", ".join(explorer_report.stage3_trends[:5])
                sections.append(f"## Research Trends\n{trends}")

            if sections:
                domain_context = "\n\nUse the following domain knowledge to generate more targeted queries:\n\n" + "\n\n".join(sections) + "\n"

        prompt = (
            f"You are an academic research assistant. Generate {self.search_queries_count} diverse English search queries for arXiv.\n\n"
            f"Research topic: {question}\n"
            f"{domain_context}"
            f"Requirements:\n"
            f"- Each query should be 3-8 words maximum\n"
            f"- Cover different angles: broad topic, specific techniques, recent trends, datasets\n"
            f"- You MUST use the domain knowledge above to generate queries that reference actual methods, benchmarks, and techniques mentioned\n"
            f"- Prioritize queries related to: classic methods mentioned, key benchmarks, core techniques in the domain\n"
            f"- Output ONLY English phrases, no operators like author: or ti:\n\n"
            f"Output format (one per line, nothing else):\n"
            f"SEARCH|<English query>\n\n"
            f"Good examples:\n"
            f"SEARCH|<topic> deep learning|\n"
            f"SEARCH|<topic> transformer architecture|\n"
            f"SEARCH|<topic> benchmark dataset|\n"
            f"SEARCH|<topic> survey review|"
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
                    dropped.append(f"重复: {q}")
                else:
                    seen.add(q)
                    queries.append(SearchQuery(query=q, reason=reason))
            elif line.startswith("SEARCH|"):
                rest = line[7:].strip()
                if rest and rest not in seen:
                    seen.add(rest)
                    queries.append(SearchQuery(query=rest, reason=""))

        for d in dropped:
            logger.debug("query 丢弃: %s", d)
        logger.info("[Stage 1] 生成 %d/%d 个搜索词（丢弃 %d 个）", len(queries), len(queries), len(dropped))

        if not queries:
            queries.append(SearchQuery(query=question, reason="原始问题"))
            logger.warning("LLM 未生成有效搜索词，用原始问题代替")

        return queries

    # ── Stage 1: PaSa 风格搜索 ───────────────────────────────

    def _search_all_pasa(self, question: str, query_strs: list[str]) -> list[PaperNode]:
        """PaSa 风格搜索：SerpAPI(google+arxiv) + ar5iv 摘要 + reranker 过滤

        流程：
          1. 对每个搜索词：SerpAPI Google 搜索获取 arXiv ID
          2. 对所有找到的 arXiv ID：ar5iv 抓摘要
          3. 统一 reranker 打分
          4. score >= threshold 才保留
        """
        all_candidates: list[tuple[str, str, str]] = []  # (arxiv_id, title, abstract)

        # Step 1: SerpAPI 搜索所有查询词
        with SerpAPIRetriever() as serp:
            for q in query_strs:
                papers = serp.search_google(q, max_results=self.search_papers_count)
                logger.info("[Stage 1] SerpAPI(q='%s') → %d papers", q[:40], len(papers))
                for p in papers:
                    m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]+)", p.url)
                    if m:
                        all_candidates.append((m.group(1), p.title, p.abstract))

        if not all_candidates:
            logger.warning("[Stage 1] 所有搜索词均未返回 arXiv ID")
            return []

        # 全局去重（按 arXiv ID）
        seen_ids: dict[str, tuple[str, str]] = {}
        for aid, title, abstract in all_candidates:
            if aid not in seen_ids:
                seen_ids[aid] = (title, abstract)

        logger.info("[Stage 1] SerpAPI 共找到 %d 个 arXiv ID（去重后 %d）", len(all_candidates), len(seen_ids))

        # Step 2: ar5iv 批量获取摘要（提升 abstract 质量）
        id_list = list(seen_ids.keys())
        abstracts = self._fetch_ar5iv_abstracts(id_list, seen_ids)

        # Step 3: 构造成 PaperNode 列表（临时节点，无 score）
        nodes: list[PaperNode] = []
        for aid in id_list:
            title, _ = seen_ids[aid]
            abstract = abstracts.get(aid, "")
            nodes.append(PaperNode(
                title=title,
                paper_id=aid,
                abstract=abstract,
                depth=0,
                select_score=0.0,
                source=f"Search: {question[:30]}",
            ))

        # Step 4: reranker 打分 + threshold 过滤
        scored = self._rerank_papers(question, nodes)
        logger.info("[Stage 1] reranker 过滤后: %d/%d 篇论文（threshold=%.2f）", len(scored), len(nodes), self.score_threshold)
        return scored

    def _fetch_ar5iv_abstracts(
        self,
        arxiv_ids: list[str],
        fallback_titles: dict[str, tuple[str, str]],
    ) -> dict[str, str]:
        """批量从 ar5iv 获取论文摘要（多线程）"""
        results: dict[str, str] = {}

        def fetch_one(aid: str) -> tuple[str, str]:
            try:
                with Ar5ivRetriever() as ar5iv:
                    html = ar5iv.fetch_full_text(aid)
                if html:
                    abstract = self._parse_abstract_from_ar5iv(html)
                    return (aid, abstract)
                return (aid, "")
            except Exception as e:
                logger.debug("ar5iv fetch failed %s: %s", aid, e)
                return (aid, "")

        with ThreadPoolExecutor(max_workers=self.threads_num) as pool:
            futures = {pool.submit(fetch_one, aid): aid for aid in arxiv_ids}
            for future in as_completed(futures):
                try:
                    aid, abstract = future.result()
                    if abstract:
                        results[aid] = abstract
                    else:
                        # Fallback: 用 SerpAPI 返回的 snippet
                        _, snippet = fallback_titles.get(aid, ("", ""))
                        results[aid] = snippet
                except Exception as e:
                    logger.debug("ar5iv fetch error: %s", e)

        found = sum(1 for v in results.values() if v)
        logger.info("[Stage 1] ar5iv 摘要获取完成: %d/%d（有摘要）", found, len(arxiv_ids))
        return results

    def _parse_abstract_from_ar5iv(self, html: str) -> str:
        """从 ar5iv HTML 中提取 abstract"""
        abstract_m = re.search(
            r'class="abstract mathjax">(.*?)</blockquote>', html, re.DOTALL
        )
        if not abstract_m:
            return ""
        abstract_raw = abstract_m.group(1)
        abstract_raw = re.sub(r"<span[^>]*>Abstract:</span>\s*", "", abstract_raw)
        abstract = re.sub(r"<[^>]+>", "", abstract_raw).strip()
        return abstract[:1000]

    def _rerank_papers(
        self,
        query: str,
        papers: list[PaperNode],
    ) -> list[PaperNode]:
        """用 reranker 对论文列表打分，score >= threshold 的保留"""
        if not papers:
            return []

        # Translate query to English for cross-language compatibility with English paper abstracts
        en_query = self._translate_question_to_english(query)
        logger.debug("rerank: translated query: '%s' -> '%s'", query, en_query)

        if self.reranker is None:
            # Fallback: 无 reranker 时，全部保留
            for p in papers:
                p.select_score = 1.0
            return papers

        docs = [p.abstract or "" for p in papers]
        logger.info("rerank: query='%s', docs count=%d", en_query, len(docs))
        reranked = self.reranker.rerank(
            query=en_query,
            documents=docs,
            top_n=self.rerank_top_n,
            return_documents=True,
        )

        logger.info("rerank returned %d results: %s", len(reranked), [f"{r.index}:{r.relevance_score:.4f}" for r in reranked])

        score_map = {r.index: r.relevance_score for r in reranked}
        paper_map = {i: p for i, p in enumerate(papers)}

        logger.debug("rerank results: %s", [f"{r.index}:{r.relevance_score:.4f}" for r in reranked])

        passed: list[PaperNode] = []
        for i, p in paper_map.items():
            score = score_map.get(i, 0.0)
            p.select_score = score
            if score >= self.score_threshold:
                passed.append(p)
                logger.debug("  [rerank] score=%.3f >= %.2f ✓ '%s'", score, self.score_threshold, p.title[:50])
            else:
                logger.debug("  [rerank] score=%.3f < %.2f ✗ '%s'", score, self.score_threshold, p.title[:50])

        return passed

    def _translate_question_to_english(self, question: str) -> str:
        """Translate a research topic/question to English for semantic matching with academic paper abstracts."""
        prompt = f"Translate this research topic to English for semantic matching with academic paper abstracts. Output ONLY the English text, nothing else.\nTopic: {question}"
        try:
            reply = self.llm.chat(
                external_prompt=prompt,
                system_prompt="You are a translator. Output ONLY the English translation, nothing else.",
                temperature=0.0,
                max_tokens=200,
            )
            en = reply.strip()
            logger.debug("Translated question: '%s' → '%s'", question, en)
            return en
        except Exception as e:
            logger.warning("Translation failed, falling back to original query: %s", e)
            return question

    # ── Stage 2: BFS 引文扩展 ────────────────────────────────

    def _bfs_expand(
        self,
        initial_papers: list[PaperNode],
        layers: int,
        question: str,
    ) -> list[PaperNode]:
        """PaSa 风格 BFS 引文扩展（纯 SerpAPI + ar5iv）"""
        papers_queue: list[PaperNode] = list(initial_papers)
        all_papers: list[PaperNode] = list(initial_papers)
        visited_ids: set[str] = {p.dedup_key for p in papers_queue if p.paper_id}

        for depth in range(1, layers + 1):
            logger.info("[Stage 2] BFS depth=%d，当前队列 %d 篇", depth, len(papers_queue))
            t_depth = time.time()

            # 取 top N 按 score 排序的论文扩展
            batch = sorted(papers_queue, key=lambda p: p.select_score, reverse=True)[: self.expand_papers_count]
            logger.info("[Stage 2] depth=%d 扩展 top %d 篇（batch size=%d）", depth, self.expand_papers_count, len(batch))

            # 批量获取所有 batch 论文的 bibliography
            batch_bibs = self._fetch_bibliographies_batch(batch)

            # 收集所有待打分的 ref 论文
            ref_candidates: list[tuple[PaperNode, str, str]] = []  # (parent, ref_arxiv_id, ref_title)

            for parent, bibs in zip(batch, batch_bibs):
                for bib_title in bibs:
                    if not bib_title.strip():
                        continue
                    # SerpAPI 搜索 ref 标题 → 获取 arXiv ID
                    ref_aids = self._resolve_ref_by_title(bib_title)
                    for ref_aid in ref_aids:
                        if ref_aid and ref_aid not in visited_ids:
                            ref_candidates.append((parent, ref_aid, bib_title))
                            visited_ids.add(ref_aid)

            logger.info("[Stage 2] depth=%d 共发现 %d 个 ref candidate（去重后）", depth, len(ref_candidates))

            if not ref_candidates:
                logger.info("[Stage 2] depth=%d 无新 ref，停止扩展", depth)
                break

            # 批量抓取 ref 论文的 abstract（多线程）
            ref_arxiv_ids = list({aid for _, aid, _ in ref_candidates})
            abstracts = self._fetch_ar5iv_abstracts(ref_arxiv_ids, {})

            # 构造成 PaperNode
            ref_nodes: list[PaperNode] = []
            for parent, ref_aid, bib_title in ref_candidates:
                abstract = abstracts.get(ref_aid, "")
                if not abstract:
                    # Fallback: 用 bibliography title 作为 abstract
                    abstract = bib_title
                ref_nodes.append(PaperNode(
                    title=bib_title,
                    paper_id=ref_aid,
                    abstract=abstract,
                    depth=depth,
                    select_score=0.0,
                    source=f"ar5iv.Ref[{parent.title[:30]}]",
                ))

            # reranker 打分 + threshold 过滤
            scored_refs = self._rerank_papers(question, ref_nodes)
            logger.info(
                "[Stage 2] depth=%d reranker 过滤后: %d/%d refs（threshold=%.2f，耗时 %.1fs）",
                depth, len(scored_refs), len(ref_nodes), self.score_threshold, time.time() - t_depth,
            )

            if not scored_refs:
                break

            # 下一层队列 = 过滤通过的 refs
            papers_queue = scored_refs
            all_papers.extend(scored_refs)

        return all_papers

    def _fetch_bibliographies_batch(
        self,
        papers: list[PaperNode],
    ) -> list[list[str]]:
        """并行获取一批论文的 bibliography（通过 ar5iv）"""
        results: list[list[str]] = []

        with ThreadPoolExecutor(max_workers=self.threads_num) as pool:
            futures = {pool.submit(self._fetch_bibliography, p): p for p in papers}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.warning("获取 bibliography 失败: %s", e)
                    results.append([])

        return results

    def _fetch_bibliography(self, paper: PaperNode) -> list[str]:
        """获取单篇论文的参考文献标题列表（使用健壮的 BeautifulSoup 解析）

        使用 ar5iv HTML 解析 bibliography，返回每条参考文献的标题。

        Returns:
            list of ref titles（最多取前 15 条）
        """
        paper_id = paper.paper_id
        if not paper_id:
            return []

        raw_id = _extract_arxiv_id(paper_id)
        if not raw_id:
            return []

        try:
            with Ar5ivRetriever() as ar5iv:
                bibs = ar5iv.extract_bibliography(ar5iv.fetch_full_text(raw_id))
            titles = [b["title"] for b in bibs[:15] if b.get("title")]
            logger.debug(
                "paper %r ar5iv bibliography: %d refs (最多取15)",
                paper.title[:30], len(titles),
            )
            time.sleep(0.3)  # 礼貌限速
            return titles
        except Exception as e:
            logger.debug("ar5iv bibliography fetch failed [%s]: %s", paper.title[:30], e)
            return []

    def _resolve_ref_by_title(self, title: str) -> list[str]:
        """用 SerpAPI Google 搜索论文标题，获取 arXiv ID 列表

        PaSa Stage 2 核心：用标题找 arXiv ID，避免依赖 S2 API。
        """
        try:
            with SerpAPIRetriever() as serp:
                papers = serp.search_ref_by_title(title, max_results=3)

            aids = []
            for p in papers:
                m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]+)", p.url)
                if m:
                    aids.append(m.group(1))

            if aids:
                logger.debug("resolve_ref_by_title('%s') → %s", title[:50], aids[:2])
            else:
                logger.debug("resolve_ref_by_title('%s') → 无 arXiv ID", title[:50])
            return aids

        except Exception as e:
            logger.debug("SerpAPI title search failed [%s]: %s", title[:50], e)
            return []

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _deduplicate(papers: list[PaperNode]) -> list[PaperNode]:
        """全局去重（按 dedup_key），保留最高 score 的版本"""
        seen: dict[str, PaperNode] = {}
        for p in papers:
            key = p.dedup_key
            if key not in seen or p.select_score > seen[key].select_score:
                seen[key] = p
        return list(seen.values())

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
      - ArXiv:2501.08008 / arXiv:2411.04358v1
      - 2301.00001 / 2501.08008（纯数字格式）
    """
    if not url:
        return ""
    import re
    cleaned = re.sub(r"^(arxiv|ArXiv|arXiv):", "", url).strip()
    m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]+)", cleaned)
    if m:
        return m.group(1)
    m = re.match(r"^(\d{4}\.\d+)", cleaned)
    if m:
        return m.group(1)
    return ""
