"""LLM 查询生成器 — 调用 LLM 生成优化查询"""

import json
import logging
import os
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
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")


def _make_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def _call_llm(prompt: str, temperature: float = 0.1, timeout: float = 30.0) -> Optional[str]:
    """调用 LLM，返回原始文本或 None"""
    key = LLM_API_KEY or os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        logger.warning("LLMQueryGenerator: no API key, skipping LLM call")
        return None

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": _make_messages(prompt),
        "temperature": temperature,
        "max_tokens": 2048,
    }

    try:
        with httpx.Client(timeout=timeout) as client:
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
    # 去掉 markdown code fence
    if text.startswith("```"):
        for line in text.splitlines():
            if line.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 3:
                    text = parts[2].strip()
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_queries(data: dict) -> dict[str, list[QueryVariant]]:
    """从 LLM 输出解析 queries 字段"""
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
    """从 LLM 输出解析 boost 字段"""
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
    """从 LLM 输出解析 rerank_query"""
    rq = data.get("rerank_query", {})
    if isinstance(rq, dict):
        return rq.get("primary", "")
    return str(rq)


class LLMQueryGenerator:
    """LLM 查询生成器"""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or LLM_API_KEY or os.environ.get("SILICONFLOW_API_KEY", "")

    async def generate_queries(
        self, topic: str, domain_profile: DomainProfile
    ) -> QueryConfig:
        """调用 LLM 生成多源优化查询 + Boost 策略

        失败时降级为简单查询。
        """
        logger.info("LLMQueryGenerator: generating queries for topic='%s'", topic)

        prompt = build_query_generation_prompt(topic, domain_profile.to_text())
        content = _call_llm(prompt)

        if content is None:
            logger.warning("LLM call failed, using fallback query config")
            return fallback_query_config(topic)

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

        config = QueryConfig(
            topic=topic,
            queries=queries,
            boost=boost,
            rerank_query=rerank_query or topic,
        )
        logger.debug("LLM generated %d sources with queries", len(queries))
        return config

    async def generate_rerank_query(
        self,
        original_query: str,
        results_preview: str,
        domain_profile: DomainProfile,
    ) -> str:
        """为 SiliconFlow Rerank 生成优化查询字符串

        失败时降级为原始 query。
        """
        classical_text = "\n".join(
            f"- {p.title} ({p.arxivid})" for p in domain_profile.classical_papers
        ) or "(No classic papers)"

        prompt = build_rerank_prompt(original_query, results_preview, classical_text)
        content = _call_llm(prompt)

        if content is None:
            logger.warning("LLM rerank query generation failed, using original query")
            return original_query

        data = _parse_json(content)
        if data is None:
            return original_query

        result = _parse_rerank_query(data)
        return result if result else original_query