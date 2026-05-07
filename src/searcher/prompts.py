"""Searcher LLM 提示词

包含：
- QUERY_GENERATION_PROMPT: 生成各源优化查询 + Boost 策略
- RERANK_QUERY_PROMPT: 为 SiliconFlow Rerank 生成优化查询

注意：提示词中的 JSON 示例使用双大括号 {{}} 进行 .format() 转义。
"""

QUERY_GENERATION_PROMPT = """\
You are an expert information retrieval consultant. Your task is to analyze the user's research topic and generate optimized search queries for multiple search sources, along with boost strategy for classic papers.

## USER'S TOPIC
{topic}

## EXPLORER DOMAIN PROFILE (Prior Knowledge)
{domain_profile_text}

## YOUR TASKS
1. Analyze the topic's core concepts, related sub-fields, and canonical terminology
2. Identify potential classic/foundational papers based on the Explorer profile
3. Generate source-specific queries that comply with each source's query syntax
4. Design a boost strategy for classic papers
5. Generate an optimized rerank query for SiliconFlow Rerank API

## SOURCE SYNTAX REFERENCE

### arXiv
- Field prefixes: `all:` (full-text), `ti:` (title only), `abs:` (abstract only), `cat:` (category), `au:` (author)
- Boolean: `AND`, `OR`, `AND NOT`
- Categories: `cs.CV` (Computer Vision), `cs.LG` (Machine Learning), `cs.CL` (NLP), `cs.AI` (AI)
- Sort: `&sortBy=relevance` (default), `&sortBy=submittedDate`, `&sortBy=lastUpdatedDate`
- Example: `all:transformer AND all:attention cat:cs.LG&sortBy=relevance`

### GitHub
- Fields: `in:name`, `in:description`, `in:readme`, `topic:`, `language:`, `stars:>N`
- Boolean: `AND`, `OR`, `AND NOT`
- Sort: `stars` (by stars), `updated` (by update time)
- Example: `transformer AND NLP in:readme language:Python stars:>500`

### HuggingFace
- Model/dataset name search, supports `+` combination
- Sort: `sort=downloads`
- Example: `LLM+text+generation`

### Hacker News (Algolia)
- Numeric filter: `numericFilters=points>N`
- Example: `transformer points:>50`

### Serper (Google)
- Natural language, similar to Google search
- Site限定: `site:arxiv.org`, `site:github.com`
- Language control: `gl` (country), `hl` (language)
- Example: `site:arxiv.org transformer survey 2024`

### Bocha (Chinese-friendly)
- Natural language, Chinese/English mixed
- Good for Chinese themes
- Supports `summary:true` for AI summary
- Example: `Transformer 注意力机制 综述 2024`

### Semantic Scholar
- Natural language semantic search (no boolean operators)
- Filters: `year`, `venue`, `author`
- Returns citation count (citationCount)

## OUTPUT FORMAT (Strict JSON)
```json
{{
  "topic_analysis": {{
    "core_concepts": ["core concept 1", "core concept 2"],
    "related_fields": ["related field 1", "related field 2"],
    "era_keywords": ["2024", "latest"],
    "missing_papers_to_check": ["paper title if any"]
  }},
  "queries": {{
    "arxiv": [
      {{"query": "all:transformer AND all:attention", "variant_type": "primary", "expected_count": 5}},
      {{"query": "ti:transformer", "variant_type": "title", "expected_count": 3}},
      {{"query": "au:Vaswani", "variant_type": "author", "expected_count": 2}}
    ],
    "github": [
      {{"query": "transformer in:readme language:Python stars:>500", "variant_type": "primary", "expected_count": 5}},
      {{"query": "topic:transformer", "variant_type": "topic", "expected_count": 3}}
    ],
    "serper": [
      {{"query": "site:arxiv.org transformer survey 2024", "variant_type": "academic", "expected_count": 5}}
    ],
    "bocha": [
      {{"query": "Transformer 注意力机制 综述 2024", "variant_type": "chinese", "expected_count": 5}}
    ],
    "huggingface": [
      {{"query": "LLM+text+generation", "variant_type": "primary", "expected_count": 5}}
    ],
    "hackernews": [
      {{"query": "transformer points:>50", "variant_type": "primary", "expected_count": 5}}
    ]
  }},
  "boost": {{
    "classical_papers": [
      {{
        "arxivid": "1706.03762",
        "title": "Attention Is All You Need",
        "boost_factor": 1.5,
        "reason": "Transformer开山之作"
      }}
    ],
    "high_citation": {{
      "threshold": 100,
      "boost_factor": 1.2
    }}
  }},
  "rerank_query": {{
    "primary": "transformer architecture attention mechanism OR \\"Attention Is All You Need\\"",
    "strategy": "ClassicPaperInjection",
    "expected_effect": "提升经典论文排名"
  }}
}}
```

## CONSTRAINTS
- Each source's query MUST comply with its syntax
- Use field prefixes (ti:, abs:, au:) for precision over full-text search (all:)
- Generate AT LEAST 2 query variants per source (arxiv, github)
- rerank_query should be ≤ 200 characters
- Classical papers from Explorer profile MUST be included in boost.classical_papers
- Output MUST be valid JSON (no markdown code fences in final output)
"""


RERANK_QUERY_PROMPT = """\
You are an expert semantic reranking consultant. Your task is to analyze the search results and the user's original query, then generate an optimized rerank query that balances relevance and classic paper recall.

## USER'S ORIGINAL QUERY
{original_query}

## SEARCH RESULTS (Top 20, for your reference)
{results_preview}

## KNOWN CLASSIC PAPERS in this field (from Explorer profile)
{classical_papers}

## YOUR ANALYSIS
Before generating the rerank query, analyze:
1. What is the core semantic meaning of the original query?
2. What types of papers are currently ranked at the top?
3. What classic papers might be relevant but missing from the top results?
4. What is the optimal query strategy?

## RERANK QUERY STRATEGY OPTIONS

**Strategy A: Semantic Expansion**
- Expand the original query with related concepts
- Use when: top results are relevant but miss related subfields
- Example: "transformer" → "transformer architecture self-attention attention mechanism"

**Strategy B: Classic Paper Injection**
- Include known classic paper titles/authors in the query
- Use when: classic papers exist but rank low despite relevance
- Example: "transformer" → "transformer OR \\"Attention Is All You Need\\" Vaswani"

**Strategy C: Simplified Core Query**
- Strip verbose terms, keep only the core concept
- Use when: original query is too specific, filtering out foundational works
- Example: "transformer architecture attention mechanism deep learning" → "transformer"

**Strategy D: Temporal Balance**
- Combine core concept with era indicator
- Use when: results are too recent AND too specific
- Example: "transformer" → "transformer neural network foundation"

**Strategy E: Metadata-Enhanced**
- Add author/work indicators if classic paper authors are known
- Example: "transformer" → "transformer OR BERT OR GPT OR \\"Attention Is All You Need\\""

## OUTPUT FORMAT (Strict JSON)
```json
{{
  "analysis": {{
    "original_query_intent": "用户原始查询的真实意图",
    "top_results_analysis": "当前 Top 结果的特点",
    "missing_classics": "可能遗漏的经典论文及原因",
    "recommended_strategy": "A/B/C/D/E 之一",
    "strategy_rationale": "选择该策略的理由"
  }},
  "rerank_query": {{
    "primary": "最终用于 SiliconFlow Rerank 的查询字符串",
    "alternatives": ["备选查询1", "备选查询2"],
    "expected_effect": "这个查询预期达到什么效果"
  }},
  "boost_hint": {{
    "urls_to_boost": ["需要额外加权的URL列表"],
    "boost_factors": {{"url1": 1.5, "url2": 1.3}},
    "rationale": "加权理由说明"
  }}
}}
```

## CONSTRAINTS
- rerank_query should be ≤ 200 characters
- rerank_query must be a valid natural language or boolean expression
- rerank_query should NOT simply repeat the original query
- boost_hint is REQUIRED if any classical papers are identified
- Choose ONLY ONE strategy as primary
- Output MUST be valid JSON (no markdown code fences in final output)
"""


def build_query_generation_prompt(topic: str, domain_profile_text: str) -> str:
    """构建查询生成的提示词"""
    return QUERY_GENERATION_PROMPT.format(
        topic=topic,
        domain_profile_text=domain_profile_text or "(No prior knowledge available — use general search)",
    )


def build_rerank_prompt(
    original_query: str, results_preview: str, classical_papers_text: str
) -> str:
    """构建 Rerank 查询优化的提示词"""
    return RERANK_QUERY_PROMPT.format(
        original_query=original_query,
        results_preview=results_preview or "(No results preview available)",
        classical_papers=classical_papers_text or "(No classic papers identified)",
    )