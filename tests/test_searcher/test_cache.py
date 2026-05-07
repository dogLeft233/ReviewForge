"""LLM 查询生成器缓存测试"""

import json
import os
import time
from pathlib import Path

import pytest

from src.searcher.llm import LLMQueryGenerator, CACHE_DIR, CACHE_TTL_SECONDS
from src.searcher.models import DomainProfile, QueryConfig


class TestCacheTTL:
    """缓存 TTL 失效测试"""

    def _fake_cache(self, topic: str = "test-topic") -> Path:
        """写入一个模拟缓存文件（使用生成器相同的路径算法）"""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        generator = LLMQueryGenerator()
        path = generator._cache_path(topic)
        data = {
            "topic": topic,
            "queries": {"arxiv": [{"query": topic, "variant_type": "primary", "expected_count": 5}]},
            "boost": {
                "classical_papers": [],
                "high_citation": {"threshold": 100, "boost_factor": 1.2},
            },
            "rerank_query": topic,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def test_fresh_cache_is_readable(self):
        """未过期的缓存可正常读取"""
        cache_file = self._fake_cache("fresh-topic")
        try:
            generator = LLMQueryGenerator()
            cfg = generator._read_cache("fresh-topic")
            assert cfg is not None
            assert cfg.topic == "fresh-topic"
        finally:
            cache_file.unlink(missing_ok=True)

    def test_expired_cache_is_deleted(self):
        """过期的缓存被自动删除并返回 None"""
        cache_file = self._fake_cache("expired-topic")
        # 修改 mtime 使其过期
        old_stamp = time.time() - CACHE_TTL_SECONDS - 60
        os.utime(cache_file, (old_stamp, old_stamp))
        try:
            generator = LLMQueryGenerator()
            cfg = generator._read_cache("expired-topic")
            assert cfg is None, "过期缓存应返回 None"
            assert not cache_file.exists(), "过期缓存文件应被删除"
        finally:
            cache_file.unlink(missing_ok=True)

    def test_cache_within_ttl_kept(self):
        """TTL 内的缓存不被删除"""
        cache_file = self._fake_cache("within-ttl")
        old_stamp = time.time() - CACHE_TTL_SECONDS + 60  # TTL 内（比过期早 60 秒）
        os.utime(cache_file, (old_stamp, old_stamp))
        try:
            generator = LLMQueryGenerator()
            cfg = generator._read_cache("within-ttl")
            assert cfg is not None, "TTL 内缓存应正常读取"
            assert cache_file.exists(), "TTL 内缓存文件应保留"
        finally:
            cache_file.unlink(missing_ok=True)

    def test_edge_case_near_boundary(self):
        """接近但未到 TTL 边界的缓存不被删除"""
        cache_file = self._fake_cache("boundary")
        # age = TTL - 5s（明确在 TTL 内，留出 test 执行缓冲）
        old_stamp = time.time() - CACHE_TTL_SECONDS + 5
        os.utime(cache_file, (old_stamp, old_stamp))
        try:
            generator = LLMQueryGenerator()
            cfg = generator._read_cache("boundary")
            assert cfg is not None, "TTL 内的缓存应正常读取"
            assert cache_file.exists(), "TTL 内缓存文件应保留"
        finally:
            cache_file.unlink(missing_ok=True)
