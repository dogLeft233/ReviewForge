"""LLM 查询生成器（支持缓存）"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import httpx

from src.searcher.models import (
    BoostConfig,
    ClassicalPaper,
    DomainProfile,
    QueryConfig,
    QueryVariant,
    fallback_query_config,
)
from src.searcher.prompts import build_query_generation_prompt, build_rerank_prompt

logger = logging.getLogger(__name__)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "THUDM/GLM-4-9B-0414")
CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", ".llm_cache"))
CACHE_TTL_SECONDS = 3600  # 1 hour


def _call_llm(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = 0.1,
    timeout: float = 60.0,
) -> Optional[str]:
    """调用 LLM，返回原始文本或 None"""
    key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("LLM_API_KEY", "")
    if not key:
        logger.warning("LLMQueryGenerator: no API key, skipping LLM call")
        return None

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 2048,
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=True) as client:
            resp = client.post(f"{LLM_BASE_URL}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return None


def _parse_json(content: str) -> Optional[dict]:
    """提取并解析 JSON（兼容带 ```json 包裹的情况）"""
    text = content.strip()
    # 去掉 markdown code fence：```json ... ```
    if text.startswith("```"):
        # Find the closing fence
        first_fence = text.find("```")
        if first_fence >= 0:
            # Content after opening ```
            rest = text[first_fence + 3:]
            close_fence = rest.find("```")
            if close_fence >= 0:
                text = rest[:close_fence].strip()
            else:
                # No closing fence found, try maxsplit approach
                parts = text.split("```", 2)
                if len(parts) >= 2:
                    text = parts[1].strip()
                else:
                    text = rest.strip()
        else:
            text = text[3:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试找到第一个 { 开始的位置
        idx = text.find("{")
        if idx >= 0:
            try:
                return json.loads(text[idx:])
            except json.JSONDecodeError:
                pass
        return None


def _parse_queries(data: dict) -> dict[str, list[QueryVariant]]:
    queries_raw = data.get("queries", {})
    result: dict[str, list[QueryVariant]] = {}
    for source, variants in queries_raw.items():
        result[source] = []
        for v in variants:
            if isinstance(v, dict):
                result[source].append(QueryVariant(
                    query=v.get("query", ""),
                    variant_type=v.get("variant_type", "primary"),
                    source=source,
                    expected_count=v.get("expected_count", 5),
                ))
            elif isinstance(v, str):
                result[source].append(QueryVariant(query=v, variant_type="primary", source=source))
    return result


def _parse_boost(data: dict) -> BoostConfig:
    boost_raw = data.get("boost", {})
    classical_raw = boost_raw.get("classical_papers", [])
    papers = []
    for p in classical_raw:
        if isinstance(p, dict):
            papers.append(ClassicalPaper(
                title=p.get("title", ""),
                arxivid=p.get("arxivid", ""),
                url=p.get("url", ""),
                boost_factor=p.get("boost_factor", 1.5),
                reason=p.get("reason", ""),
            ))
    hc = boost_raw.get("high_citation", {})
    return BoostConfig(
        classical_papers=papers,
        high_citation_threshold=hc.get("threshold", 100),
        high_citation_boost=hc.get("boost_factor", 1.2),
    )


def _parse_rerank_query(data: dict) -> str:
    rq = data.get("rerank_query", {})
    if isinstance(rq, dict):
        return rq.get("primary", "")
    return str(rq)


def _config_to_cache(cfg: QueryConfig) -> dict:
    return {
        "topic": cfg.topic,
        "queries": {
            source: [
                {
                    "query": v.query,
                    "variant_type": v.variant_type,
                    "expected_count": v.expected_count,
                }
                for v in variants
            ]
            for source, variants in cfg.queries.items()
        },
        "boost": {
            "classical_papers": [
                {
                    "title": p.title,
                    "arxivid": p.arxivid,
                    "url": p.url,
                    "boost_factor": p.boost_factor,
                    "reason": p.reason,
                }
                for p in cfg.boost.classical_papers
            ],
            "high_citation": {
                "threshold": cfg.boost.high_citation_threshold,
                "boost_factor": cfg.boost.high_citation_boost,
            },
        },
        "rerank_query": cfg.rerank_query,
    }


def _cache_to_config(data: dict, topic: str) -> QueryConfig:
    queries = {}
    for source, variants in data.get("queries", {}).items():
        queries[source] = [
            QueryVariant(query=v.get("query", ""), variant_type=v.get("variant_type", "primary"),
                         source=source, expected_count=v.get("expected_count", 5))
            for v in variants
        ]
    boost_data = data.get("boost", {})
    papers = []
    for p in boost_data.get("classical_papers", []):
        papers.append(ClassicalPaper(
            title=p.get("title", ""),
            arxivid=p.get("arxivid", ""),
            url=p.get("url", ""),
            boost_factor=p.get("boost_factor", 1.5),
            reason=p.get("reason", ""),
        ))
    hc = boost_data.get("high_citation", {})
    boost = BoostConfig(
        classical_papers=papers,
        high_citation_threshold=hc.get("threshold", 100),
        high_citation_boost=hc.get("boost_factor", 1.2),
    )
    return QueryConfig(
        topic=topic,
        queries=queries,
        boost=boost,
        rerank_query=data.get("rerank_query", topic),
    )


class LLMQueryGenerator:
    """LLM 查询生成器（支持缓存）"""

    def __init__(self, model: str = LLM_MODEL) -> None:
        self._model = model
        self._cache: dict[str, QueryConfig] = {}

    def _cache_path(self, topic: str) -> Path:
        safe = "".join(c if c.isalnum() else "_" for c in topic)[:50]
        return CACHE_DIR / f"{safe}.json"

    def _read_cache(self, topic: str) -> Optional[QueryConfig]:
        path = self._cache_path(topic)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug("Cache hit for topic='%s'", topic)
            return _cache_to_config(data, topic)
        except Exception as e:
            logger.warning("Cache read failed for '%s': %s", topic, e)
            return None

    def _write_cache(self, cfg: QueryConfig) -> None:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path = self._cache_path(cfg.topic)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_config_to_cache(cfg), f, ensure_ascii=False, indent=2)
            logger.debug("Cache written for topic='%s' at %s", cfg.topic, path)
        except Exception as e:
            logger.warning("Cache write failed for '%s': %s", cfg.topic, e)

    def clear_cache(self, topic: Optional[str] = None) -> None:
        if topic:
            path = self._cache_path(topic)
            if path.exists():
                path.unlink()
                logger.info("Cache cleared for topic='%s'", topic)
        else:
            for p in CACHE_DIR.glob("*.json"):
                p.unlink()
            logger.info("All cache cleared")

    async def generate_queries(
        self, topic: str, domain_profile: DomainProfile
    ) -> QueryConfig:
        """调用 LLM 生成多源优化查询 + Boost 策略

        失败时降级为简单查询。
        """
        # 1. 检查内存缓存
        if topic in self._cache:
            logger.info("Memory cache hit for topic='%s'", topic)
            return self._cache[topic]

        # 2. 检查磁盘缓存
        cached = self._read_cache(topic)
        if cached is not None:
            self._cache[topic] = cached
            return cached

        # 3. 调用 LLM
        logger.info("LLM generating queries for topic='%s' (model=%s)", topic, self._model)

        prompt = build_query_generation_prompt(topic, domain_profile.to_text())
        content = _call_llm(prompt, model=self._model)

        if content is None:
            logger.warning("LLM call failed, using fallback query config")
            return fallback_query_config(topic)

        logger.debug("LLM raw output (first 300 chars): %s", content[:300])

        data = _parse_json(content)
        if data is None:
            logger.warning("Failed to parse LLM JSON output, using fallback")
            return fallback_query_config(topic)

        try:
            queries = _parse_queries(data)
            boost = _parse_boost(data)
            rerank_query = _parse_rerank_query(data)
        except Exception as e:
            logger.warning("Error parsing LLM output fields: %s, using fallback", e)
            return fallback_query_config(topic)

        cfg = QueryConfig(
            topic=topic,
            queries=queries,
            boost=boost,
            rerank_query=rerank_query or topic,
        )

        # 4. 写入缓存
        self._cache[topic] = cfg
        self._write_cache(cfg)

        logger.debug("LLM generated %d sources with queries", len(queries))
        return cfg

    async def generate_rerank_query(
        self,
        original_query: str,
        results_preview: str,
        domain_profile: DomainProfile,
    ) -> str:
        """为 SiliconFlow Rerank 生成优化查询字符串（无缓存，简单逻辑）"""
        classical_text = "\n".join(
            f"- {p.title} ({p.arxivid})" for p in domain_profile.classical_papers
        ) or "(No classic papers)"

        prompt = build_rerank_prompt(original_query, results_preview, classical_text)
        content = _call_llm(prompt, model=self._model)

        if content is None:
            return original_query

        data = _parse_json(content)
        if data is None:
            return original_query

        result = _parse_rerank_query(data)
        return result if result else original_query