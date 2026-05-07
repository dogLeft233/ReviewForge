#!/usr/bin/env python3
"""
实验：Embedding 相似度 vs LLM selector 的 BFS 过滤效果

目标：验证能否用 API embedding 相似度代替本地 7B LLM 来做 BFS 节点评分。
参考：PaSa (https://pasa-agent.ai/) 的 BFS 策略，将其 selector 替换为 embedding 余弦相似度。

核心假设：
  给定用户查询 Q 和一篇论文 P，
  cos_sim(embed(Q), embed(P.title + P.abstract)) > threshold
  等价于 PaSa 的 P(True | prompt) > 0.5

实验流程：
  Phase 1: 对已知 topic 搜索 arxiv → embedding 过滤 → 人工标注
  Phase 2: 对通过过滤的论文做 BFS 引用展开 → embedding 过滤下一层
  Phase 3: 对比 precision@k / recall / F1

使用方法：
  export SILICONFLOW_API_KEY="sk-xxx"
  python experiments/embedding_bfs_experiment.py --topic "LoRA fine-tuning" --bfs-layers 1

作者：Planner Agent @ ReviewForge
日期：2026-05-07
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# ─── 日志 ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("embedding_bfs_experiment")

# ─── 配置 ─────────────────────────────────────────────────────
SILICONFLOW_API_KEY = (
    os.environ.get("SILICONFLOW_API_KEY")
    or os.environ.get("LLM_API_KEY")
    or ""
)
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"  # 硅基流动支持的 embedding 模型
EMBEDDING_DIM = 1024
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"    # Rerank 模型（作为对比 baseline）
LLM_MODEL = "THUDM/GLM-4-9B-0414"           # 用于生成查询

BASE_URL = "https://api.siliconflow.cn/v1"
VERBOSE = False  # 是否输出详细日志
USER_AGENT = "ReviewForge-Experiment/1.0 (Academic Survey BFS Test)"

# ─── 数据模型 ────────────────────────────────────────────────


@dataclass
class Paper:
    """论文节点（与 PaSa 的 PaperNode 类似）"""
    title: str
    arxiv_id: str = ""
    abstract: str = ""
    sections: dict = field(default_factory=dict)  # section_name → [citation_titles]
    depth: int = 0                    # BFS 深度
    source: str = "Search"            # Search / Expand
    embedding: list[float] = field(default_factory=list)  # 缓存 embedding
    similarity: float = 0.0           # 与 query 的余弦相似度
    relevance_label: Optional[bool] = None  # 人工标注（True=相关）
    citations: list = field(default_factory=list)  # 引用的论文标题列表

    @property
    def text_for_embedding(self) -> str:
        """用于计算 embedding 的文本"""
        return f"{self.title} {self.abstract[:500]}"


@dataclass
class ExperimentResult:
    """实验单轮结果"""
    topic: str
    query: str                          # 用于 embedding 的查询
    total_papers: int
    relevant_papers: int                # 人工标注
    # 不同阈值的过滤效果
    threshold_results: dict = field(default_factory=dict)
    # threshold → {"precision": float, "recall": float, "f1": float, "kept": int, "correct": int}

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Topic: {self.topic}")
        print(f"Embedding Query: {self.query}")
        print(f"Total papers: {self.total_papers}")
        print(f"Relevant papers: {self.relevant_papers}")
        print(f"{'='*60}")
        print(f"{'Threshold':>10} | {'Kept':>5} | {'Correct':>7} | {'Prec':>6} | {'Recall':>6} | {'F1':>6}")
        print(f"{'-'*10}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
        for th, res in sorted(self.threshold_results.items()):
            print(
                f"{float(th):>10.2f} | {res['kept']:>5d} | {res['correct']:>7d} | "
                f"{res['precision']:>6.3f} | {res['recall']:>6.3f} | {res['f1']:>6.3f}"
            )
        print(f"{'='*60}")


# ─── API 工具 ────────────────────────────────────────────────


def siliconflow_headers() -> dict:
    return {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }


def call_embedding(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """调用硅基流动 Embedding API"""
    if not SILICONFLOW_API_KEY:
        raise RuntimeError("SILICONFLOW_API_KEY not set")

    embeddings = []
    # 分批处理（单次最多 32 条）
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        payload = {
            "model": model,
            "input": batch,
            "encoding_format": "float",
        }
        for attempt in range(3):
            try:
                resp = httpx.post(
                    f"{BASE_URL}/embeddings",
                    json=payload,
                    headers=siliconflow_headers(),
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                # 按 input 顺序提取 embedding
                batch_emb = [d["embedding"] for d in data["data"]]
                # 去重：某些 API 返回乱序 data，按 index 排序
                if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])
                    batch_emb = [d["embedding"] for d in sorted_data]
                embeddings.extend(batch_emb)
                break
            except Exception as e:
                logger.warning(f"Embedding API attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(1)

    return embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))


def call_llm(prompt: str, temperature: float = 0.1) -> Optional[str]:
    """调用硅基流动 Chat API"""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 1024,
    }
    try:
        resp = httpx.post(
            f"{BASE_URL}/chat/completions",
            json=payload,
            headers=siliconflow_headers(),
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None


# ─── 搜索工具 ────────────────────────────────────────────────


SEARCH_CACHE = {}  # arxiv_id → Paper


def print_status(msg: str):
    """始终打印的状态信息（不受日志级别影响）"""
    print(f"  [\u2699] {msg}")


def debug_log(msg: str):
    """仅在 verbose 模式下输出"""
    if VERBOSE:
        print(f"  [DEBUG] {msg}")


def search_arxiv(query: str, max_results: int = 15) -> list[Paper]:
    """搜索 arXiv（与 meta.py 一致）"""
    import urllib.parse
    import xml.etree.ElementTree as ET
    import re

    # 清理查询（去掉 arxiv 语法后缀）
    clean = re.sub(r'\s+', ' ', query.replace("all:", "").replace("ti:", "").replace("abs:", ""))
    clean = clean.split("&")[0].strip()

    print_status(f"arXiv search: '{clean[:60]}...' (max={max_results})")

    url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query=all:{urllib.parse.quote(clean)}&"
        f"start=0&max_results={max_results}&sortBy=relevance"
    )
    headers = {"User-Agent": USER_AGENT}

    papers = []
    for attempt in range(5):  # 最多 5 次重试
        try:
            resp = httpx.get(url, headers=headers, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt  # 指数退避: 1, 2, 4, 8, 16 秒
                print_status(f"arXiv rate limited (429), waiting {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                paper_id = entry.find("atom:id", ns).text.strip()
                arxiv_id_match = re.search(r'(\d{4}\.\d+)', paper_id)
                arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else ""
                abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
                papers.append(Paper(
                    title=title,
                    arxiv_id=arxiv_id,
                    abstract=abstract,
                ))
                if VERBOSE:
                    print(f"    \u2514 {title[:70]}...")
            if VERBOSE:
                print(f"  \u2192 {len(papers)} papers found")
            break
        except httpx.TimeoutException:
            wait = 2 ** attempt
            print_status(f"arXiv timeout, retrying in {wait}s (attempt {attempt+1}/5)...")
            time.sleep(wait)
        except Exception as e:
            print_status(f"arXiv error: {e} (attempt {attempt+1}/5)")
            if attempt == 4:
                print_status(f"arXiv search exhausted retries for: {clean[:60]}")
                return []
            time.sleep(2 ** attempt)

    if VERBOSE:
        print(f"  [\u2713] arXiv returned {len(papers)} papers")
    else:
        print_status(f"Found {len(papers)} papers from arXiv")
    return papers


def fetch_arxiv_html(arxiv_id: str) -> Optional[dict]:
    """从 ar5iv 获取论文全文，提取 sections 和引用（同 PaSa 的 get_2nd_section）"""
    import re
    import bs4

    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        resp = httpx.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        html = resp.text
    except Exception as e:
        logger.debug(f"ar5iv fetch failed for {arxiv_id}: {e}")
        return None

    # 解析 HTML（简版，只提取 section 标题和引用 ID）
    try:
        soup = bs4.BeautifulSoup(html, "lxml")
    except Exception:
        return None

    # 提取引用映射
    def _extract_title(ref_text: str) -> Optional[str]:
        """从 bib 条目中提取论文标题"""
        # 优先匹配引号中的标题: "Title Here"
        import re as _re
        m = _re.search(r'"([^"]+)"', ref_text)
        if m:
            return m.group(1)
        # 其次匹配书名号形式: 《Title Here》
        m = _re.search(r'《([^》]+)》', ref_text)
        if m:
            return m.group(1)
        # 再尝试：去掉前导作者部分（逗号分隔，看最后一个逗号前的部分是否含数字）
        m = _re.search(r'\.\s*"([A-Z][A-Za-z0-9 :,-]+)"', ref_text)
        if m:
            return m.group(1)
        # 最后尝试提取: 作者们. "标题" 之外的纯文本标题
        # 匹配 pattern: "Author1, Author2, and Author3. Title." 提取 Title
        segments = ref_text.split(". ")
        if len(segments) >= 2:
            candidate = segments[1].strip()
            # 标题通常较短且不含数字序号
            if 5 < len(candidate) < 120 and candidate[0].isupper():
                return candidate
        return None

    citations = {}
    biblist = soup.find(class_="ltx_biblist")
    if biblist:
        for li in biblist.find_all("li", recursive=False):
            bib_id = li.get("id", "")
            # 优先找 class="ltx_bibblock" 的 span
            metas = [x.text.strip() for x in li.find_all("span", class_="ltx_bibblock")]
            if not metas:
                # 直接取整个 li 的文本
                metas = [li.get_text().strip()]
            ref_text = " ".join(metas)
            title = _extract_title(ref_text)
            if title:
                citations[bib_id] = title

    # 提取 sections 和其中引用的 ID
    sections = {}
    for section_tag in soup.find_all("section"):
        section_title_tag = section_tag.find(["h1", "h2", "h3", "h4"])
        if section_title_tag is None:
            continue
        section_title = section_title_tag.get_text().strip()
        # 跳过参考文献/致谢
        if any(w in section_title.lower() for w in ["reference", "acknowledgment", "appendix"]):
            continue

        # 提取该 section 中的引用 ID
        cite_ids = set()
        for cite_tag in section_tag.find_all("cite"):
            for a in cite_tag.find_all("a", class_="ltx_ref"):
                href = a.get("href", "").strip("#")
                if href:
                    cite_ids.add(href)

        # 解析引用标题
        cited_titles = []
        for cid in cite_ids:
            if cid in citations:
                cited_titles.append(citations[cid])

        if cited_titles:
            sections[section_title] = cited_titles

    return {"sections": sections} if sections else None


def search_cited_paper(title: str) -> Optional[Paper]:
    """根据标题搜索论文（使用 arXiv API 取代 HTML 爬虫）"""
    import urllib.parse
    import xml.etree.ElementTree as ET
    import re

    # 清理标题：去掉引号、特殊字符、截断到合理长度
    clean = title.strip().strip('"').strip("'").strip('"')
    # 如果标题看起来像 bib entry（含作者名逗号），尝试提取后半部分
    m = re.search(r'[Aa]nd\s+.+?\.\s+(.+?)(?:\.|$)', clean)
    if m:
        clean = m.group(1).strip()
    # 如果标题以句号结尾就去掉
    clean = clean.rstrip(".").strip()
    # 截断到 80 字符（arXiv 不接太长 query）
    if len(clean) > 80:
        # 尝试在第一个句号处截断
        dot = clean.find(".", 10, 100)
        if dot > 0:
            clean = clean[:dot]
        else:
            clean = clean[:80]

    if not clean:
        return None

    debug_log(f"Searching cited: '{clean[:60]}...'")

    # 使用 arXiv API 的 ti: 搜索（比 HTML 爬虫更可靠）
    url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query=ti:{urllib.parse.quote(clean)}&"
        f"start=0&max_results=3&sortBy=relevance"
    )
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(3):
        try:
            resp = httpx.get(url, headers=headers, timeout=15)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                entry_title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                # 检查标题是否真的匹配（避免返回不相关结果）
                common = set(clean.lower().split()[:3]) & set(entry_title.lower().split()[:3])
                if len(common) == 0:
                    continue
                paper_id = entry.find("atom:id", ns).text.strip()
                arxiv_id_match = re.search(r'(\d{4}\.\d+)', paper_id)
                arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else ""
                abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
                paper = Paper(title=entry_title, arxiv_id=arxiv_id, abstract=abstract)
                debug_log(f"  Found: {entry_title[:50]} ({arxiv_id})")
                return paper
            return None
        except Exception as e:
            if attempt == 2:
                debug_log(f"  Failed to find cited: {clean[:40]}")
                return None
            time.sleep(2 ** attempt)
    return None


def search_paper_by_arxiv_id(arxiv_id: str) -> Optional[Paper]:
    """根据 arxiv ID 获取论文元信息"""
    if arxiv_id in SEARCH_CACHE:
        return SEARCH_CACHE[arxiv_id]

    import xml.etree.ElementTree as ET
    import re

    # 先试试 arXiv API
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    try:
        resp = httpx.get(url, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
            paper = Paper(
                title=title,
                arxiv_id=arxiv_id,
                abstract=abstract,
            )
            SEARCH_CACHE[arxiv_id] = paper
            return paper
    except Exception:
        pass
    return None


# ─── LLM 查询生成 ────────────────────────────────────────────


def generate_queries(topic: str) -> dict:
    """调用 LLM 生成多源查询（同 MetaSearcher Phase 0）"""
    prompt = f"""You are an expert information retrieval consultant. Generate optimized search queries for topic: "{topic}".

Output JSON format (strict, no markdown):
{{
  "topic_analysis": {{
    "core_concepts": ["concept1", "concept2"],
    "related_fields": ["field1"],
    "era_keywords": ["2024", "latest"]
  }},
  "queries": {{
    "arxiv": [
      {{"query": "natural language query for arxiv search", "variant_type": "primary", "expected_count": 5}},
      {{"query": "alternative query", "variant_type": "title", "expected_count": 3}}
    ]
  }},
  "rerank_query": {{
    "primary": "optimized query for reranking",
    "strategy": "SemanticExpansion"
  }}
}}

IMPORTANT: 
- The query MUST be directly related to "{topic}" — do NOT use example queries from unrelated topics.
- arxiv query should NOT use field prefixes like "all:" — just use plain natural language.
- Output valid JSON only, no markdown fences."""

    content = call_llm(prompt)
    if not content:
        logger.warning("LLM query generation failed, using topic as query")
        return {"arxiv": [{"query": topic, "variant_type": "primary"}],
                "rerank_query": {"primary": topic, "strategy": "Simple"}}

    # 解析 JSON
    import re, json
    text = content.strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        text = parts[1] if len(parts) >= 2 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
        # 提取 queries 字段
        queries = data.get("queries", {})
        rerank_raw = data.get("rerank_query", {})
        rerank_query = rerank_raw.get("primary", topic) if isinstance(rerank_raw, dict) else str(rerank_raw)
        if not queries:
            queries = {"arxiv": [{"query": topic, "variant_type": "primary"}]}
        return {"queries": queries, "rerank_query": rerank_query}
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM output: {content[:100]}")
        return {"queries": {"arxiv": [{"query": topic, "variant_type": "primary"}]},
                "rerank_query": topic}


# ─── BFS 展开 ────────────────────────────────────────────────


def expand_paper(paper: Paper, max_depth: int = 1) -> list[Paper]:
    """对一篇论文做 BFS 引用展开（同 PaSa 的 expand 阶段）"""
    if paper.depth >= max_depth:
        return []

    # 获取 sections + 引用
    html_data = fetch_arxiv_html(paper.arxiv_id)
    if html_data is None:
        logger.debug(f"No HTML for {paper.arxiv_id}, skipping expand")
        return []

    paper.sections = html_data["sections"]

    # 收集所有引用的论文标题
    cited_titles = set()
    for section, titles in paper.sections.items():
        for t in titles:
            cited_titles.add(t)

    logger.info(f"  {paper.title[:40]}... → {len(cited_titles)} unique citations")

    # 搜索引用的论文
    expanded = []
    for title in list(cited_titles)[:10]:  # 每篇论文最多展开 10 篇引用
        cited_paper = search_cited_paper(title)
        if cited_paper:
            cited_paper.depth = paper.depth + 1
            cited_paper.source = "Expand"
            expanded.append(cited_paper)

    return expanded


# ─── 实验核心 ────────────────────────────────────────────────


def run_embedding_filter_experiment(
    papers: list[Paper],
    query_text: str,
) -> ExperimentResult:
    """核心实验：对一组论文测试不同 embedding 阈值下的过滤效果

    Args:
        papers: 论文列表（需包含 relevance_label）
        query_text: 用于 embedding 的查询文本

    Returns:
        ExperimentResult: 各阈值下的 precision/recall/f1
    """
    if not papers:
        return ExperimentResult(topic="", query=query_text, total_papers=0, relevant_papers=0)

    # 1. 计算 query embedding
    logger.info(f"Computing query embedding for: {query_text[:60]}...")
    query_emb = call_embedding([query_text])[0]

    # 2. 计算各论文的 embedding
    texts = [p.text_for_embedding for p in papers]
    logger.info(f"Computing embeddings for {len(texts)} papers...")
    paper_embs = call_embedding(texts)

    # 3. 计算相似度
    for p, emb in zip(papers, paper_embs):
        p.embedding = emb
        p.similarity = cosine_similarity(query_emb, emb)

    # 4. 按相似度排序输出
    sorted_papers = sorted(papers, key=lambda p: p.similarity, reverse=True)
    print(f"\n  Papers sorted by embedding similarity:")
    print(f"  {'Sim':>6} | {'Label':>5} | {'Title'}")
    print(f"  {'-'*6}-+-{'-'*5}-+-{'-'*60}")
    for p in sorted_papers:
        label = "✓REL" if p.relevance_label else "✗IRR" if p.relevance_label is False else "  ?"
        print(f"  {p.similarity:>6.3f} | {label:>5} | {p.title[:60]}")

    # 5. 测试不同阈值
    y_true = np.array([p.relevance_label for p in papers if p.relevance_label is not None])
    if len(y_true) == 0:
        logger.warning("No labeled papers, skipping threshold evaluation")
        return ExperimentResult(
            topic="", query=query_text,
            total_papers=len(papers),
            relevant_papers=0,
        )

    has_label = [p for p in papers if p.relevance_label is not None]
    sim_scores = np.array([p.similarity for p in has_label])

    total_relevant = int(y_true.sum())

    result = ExperimentResult(
        topic="",
        query=query_text,
        total_papers=len(y_true),
        relevant_papers=total_relevant,
    )

    # 从 min 到 max 采样阈值
    min_th = max(0.0, sim_scores.min() - 0.05)
    max_th = min(1.0, sim_scores.max() + 0.05)
    thresholds = np.linspace(min_th, max_th, 20)

    for th in thresholds:
        y_pred = (sim_scores >= th).astype(int)
        if y_pred.sum() == 0:
            prec, rec, f1 = 0.0, 0.0, 0.0
        else:
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
        result.threshold_results[f"{th:.2f}"] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "kept": int(y_pred.sum()),
            "correct": int((y_pred & y_true).sum()),
        }

    # 6. 找最佳 F1 阈值
    best_th = max(result.threshold_results.items(),
                  key=lambda x: x[1]["f1"])
    logger.info(f"  Best threshold: {best_th[0]} (F1={best_th[1]['f1']:.3f}, "
                f"Prec={best_th[1]['precision']:.3f}, Recall={best_th[1]['recall']:.3f})")

    return result


# ─── 主实验 ──────────────────────────────────────────────────


def run_experiment(topic: str, bfs_layers: int = 0):
    """完整实验流程

    Phase 1: LLM 生成查询 → arXiv 搜索 → embedding 过滤标注
    Phase 2: BFS 展开（可选）
    """
    print(f"\n{'#'*60}")
    print(f"# 实验: Embedding BFS Filter")
    print(f"# Topic: {topic}")
    print(f"# BFS Layers: {bfs_layers}")
    print(f"#{'#'*60}\n")

    # Step 0: LLM 生成查询
    print("=" * 50)
    print("Phase 0: LLM Query Generation")
    print("=" * 50)
    gen_result = generate_queries(topic)
    queries = gen_result.get("queries", {})
    rerank_query = gen_result.get("rerank_query", topic)

    print(f"  Rerank query: {rerank_query}")

    # 提取所有查询
    arxiv_queries = queries.get("arxiv", [])
    if not arxiv_queries:
        arxiv_queries = [{"query": topic, "variant_type": "primary"}]

    for i, q in enumerate(arxiv_queries):
        print(f"  Arxiv query [{q.get('variant_type','?')}]: {q['query'][:80]}")

    # Step 1: arXiv 搜索
    print("\n" + "=" * 50)
    print("Phase 1: arXiv Search")
    print("=" * 50)

    all_papers = []
    for q in arxiv_queries[:1]:  # 只用 primary 查询
        papers = search_arxiv(q["query"], max_results=15)
        all_papers.extend(papers)

    print(f"  Total papers found: {len(all_papers)}")

    if not all_papers:
        logger.error("No papers found, aborting")
        return

    # Step 2: 人工标注（展示论文列表让用户标注）
    print("\n" + "=" * 50)
    print("Phase 2: Relevance Annotation")
    print("=" * 50)
    print(f"请对以下 {len(all_papers)} 篇论文标注相关性 (y=相关, n=不相关, s=跳过):")
    print()

    label_map = {}
    for i, p in enumerate(all_papers, 1):
        print(f"  [{i:2d}] {p.title}")
        print(f"       arXiv:{p.arxiv_id}")
        print(f"       {p.abstract[:120]}...")
        inp = input(f"       y/n/s: ").strip().lower()
        if inp == "y":
            p.relevance_label = True
            label_map[i] = True
        elif inp == "n":
            p.relevance_label = False
            label_map[i] = False
        else:
            p.relevance_label = None
            label_map[i] = None
        print()

    labeled = sum(1 for p in all_papers if p.relevance_label is not None)
    relevant = sum(1 for p in all_papers if p.relevance_label is True)
    print(f"  Labeled: {labeled}/{len(all_papers)}, Relevant: {relevant}")

    if labeled == 0:
        logger.error("No labels provided, cannot evaluate")
        return

    # Step 3: 测试不同查询的 embedding 过滤效果
    print("\n" + "=" * 50)
    print("Phase 3: Embedding Similarity Filtering")
    print("=" * 50)
    print()

    queries_to_test = [
        ("Original user query", topic),
        ("LLM rerank query", rerank_query),
        ("Arxiv primary query", arxiv_queries[0]["query"] if arxiv_queries else topic),
    ]

    for query_name, query_text in queries_to_test:
        print(f"\n  --- Query: {query_name} ---")
        result = run_embedding_filter_experiment(all_papers, query_text)
        if result.total_papers > 0:
            result.print_summary()

    # Step 4 (可选): BFS 展开实验
    if bfs_layers > 0:
        print("\n" + "=" * 50)
        print("Phase 4: BFS Citation Expansion")
        print("=" * 50)

        # 取 embedding 相似度最高的论文做 BFS
        # 先用 rerank_query 计算
        rerank_emb = call_embedding([rerank_query])[0]
        for p in all_papers:
            if not p.embedding:
                texts = [p.text_for_embedding]
                p.embedding = call_embedding(texts)[0]
            p.similarity = cosine_similarity(rerank_emb, p.embedding)

        top_papers = sorted(all_papers, key=lambda p: p.similarity, reverse=True)[:5]
        print(f"\n  Top 5 papers by similarity (for BFS expansion):")
        for p in top_papers:
            label = "REL" if p.relevance_label == True else "IRR" if p.relevance_label == False else "?"
            print(f"  [{label}] sim={p.similarity:.3f} | {p.title[:60]}")

        # 对每篇论文做 BFS 展开
        bfs_results = []
        for p in top_papers:
            print(f"\n  Expanding: {p.title[:50]}...")
            expanded = expand_paper(p, max_depth=bfs_layers)
            print(f"    Found {len(expanded)} cited papers")

            if expanded:
                # 计算引用论文与 query 的 embedding 相似度
                exp_texts = [e.text_for_embedding for e in expanded]
                exp_embs = call_embedding(exp_texts)
                for e, emb in zip(expanded, exp_embs):
                    e.embedding = emb
                    e.similarity = cosine_similarity(rerank_emb, emb)

                sorted_exp = sorted(expanded, key=lambda e: e.similarity, reverse=True)
                print(f"    Top cited by similarity:")
                for e in sorted_exp[:5]:
                    print(f"      sim={e.similarity:.3f} | {e.title[:60]}")

                bfs_results.extend(expanded)

        print(f"\n  BFS total expanded papers: {len(bfs_results)}")

    print(f"\n{'#'*60}")
    print(f"# 实验完成")
    print(f"#{'#'*60}")


# ─── 快速模式（用已有数据免标注） ──────────────────────────

def run_quick_test():
    """快速测试：用预定义的 topic + 已知相关的论文验证 embedding 过滤

    使用 "LoRA fine-tuning" 作为测试 topic，因为结果中相关/不相关论文容易区分。
    """
    topic = "LoRA fine-tuning"
    print(f"\n{'#'*60}")
    print(f"# 快速测试: Embedding Filter")
    print(f"# Topic: {topic}")
    print(f"#{'#'*60}\n")

    # 搜索 arxiv
    papers = search_arxiv(topic, max_results=15)
    print(f"Found {len(papers)} papers\n")

    # 自动标注（基于标题关键词——仅供快速测试，不准确）
    relevant_keywords = ["lora", "low-rank adaptation", "parameter-efficient", "adapter", "prefix tuning"]
    for p in papers:
        title_lower = p.title.lower()
        p.relevance_label = any(kw in title_lower for kw in relevant_keywords)

    rel_count = sum(1 for p in papers if p.relevance_label)
    print(f"Auto-labeled: {rel_count} relevant, {len(papers)-rel_count} irrelevant\n")

    # 执行 embedding 过滤
    result = run_embedding_filter_experiment(papers, topic)
    result.print_summary()

    # BFS 快速测试：取 top-3 论文展开
    print(f"\n--- BFS Quick Test (1 layer) ---")
    top3 = sorted(papers, key=lambda p: p.similarity, reverse=True)[:3]
    for p in top3:
        print(f"\nExpanding: {p.title[:50]}...")
        expanded = expand_paper(p, max_depth=1)
        print(f"  Cited papers: {len(expanded)}")
        if expanded:
            exp_texts = [e.text_for_embedding for e in expanded]
            query_emb = call_embedding([topic])[0]
            exp_embs = call_embedding(exp_texts)
            for e, emb in zip(expanded, exp_embs):
                e.similarity = cosine_similarity(query_emb, emb)
            sorted_exp = sorted(expanded, key=lambda e: e.similarity, reverse=True)
            for e in sorted_exp[:5]:
                print(f"  sim={e.similarity:.3f} | {e.title[:60]}")


# ─── 入口 ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding BFS 过滤实验")
    parser.add_argument("--topic", type=str, default="LoRA fine-tuning",
                        help="测试主题 (默认: LoRA fine-tuning)")
    parser.add_argument("--bfs-layers", type=int, default=0,
                        help="BFS 展开层数 (0=不展开)")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：自动标注（基于标题关键词）")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="详细输出模式 (显示每步进度和每篇论文标题)")
    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True
        logger.setLevel(logging.DEBUG)
        print("[Verbose mode ON]")

    if args.quick:
        run_quick_test()
    else:
        run_experiment(topic=args.topic, bfs_layers=args.bfs_layers)
