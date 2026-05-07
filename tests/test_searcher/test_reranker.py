"""Rerank API 测试"""

import pytest
from unittest.mock import patch, MagicMock

from src.searcher.result import SearchResult
from src.searcher.rankers.reranker import rerank


class TestRerank:
    """SiliconFlow Rerank API 测试（mock）"""

    def test_no_results(self):
        """空结果直接返回"""
        result = rerank("test query", [])
        assert result == []

    @patch.dict("os.environ", {"SILICONFLOW_API_KEY": "sk-test"})
    @patch("httpx.Client.post")
    def test_rerank_success(self, mock_post: MagicMock):
        """正常 rerank 流程"""
        mock_post.return_value.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.88},
            ]
        }
        mock_post.return_value.raise_for_status = MagicMock()

        results = [
            SearchResult(title="Paper A", url="https://a.com", source="arxiv", category="papers", domain="a.com"),
            SearchResult(title="Paper B", url="https://b.com", source="arxiv", category="papers", domain="b.com"),
        ]

        reranked = rerank("transformer", results)

        assert len(reranked) == 2
        # B 的 relevance_score 更高，应该排在前面
        assert reranked[0].title == "Paper B"
        assert reranked[0].rank_score == 0.95
        assert reranked[1].title == "Paper A"
        assert reranked[1].rank_score == 0.88

    @patch.dict("os.environ", {"SILICONFLOW_API_KEY": "sk-test"})
    @patch("httpx.Client.post")
    def test_rerank_api_failure_fallback(self, mock_post: MagicMock):
        """API 失败时降级为原始结果"""
        mock_post.side_effect = Exception("API error")

        results = [
            SearchResult(title="Paper A", url="https://a.com", source="arxiv", category="papers", domain="a.com"),
        ]

        reranked = rerank("test", results)
        # 降级为原始结果
        assert reranked == results

    @patch.dict("os.environ", {"SILICONFLOW_API_KEY": ""})
    def test_no_api_key(self):
        """无 API key 时直接返回原始结果"""
        results = [
            SearchResult(title="A", url="https://a.com", source="arxiv", category="papers", domain="a.com"),
        ]
        reranked = rerank("test", results)
        assert reranked == results

    @patch.dict("os.environ", {"SILICONFLOW_API_KEY": "sk-test"})
    @patch("httpx.Client.post")
    def test_top_n(self, mock_post: MagicMock):
        """top_n 参数限制返回数量"""
        mock_post.return_value.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.9},
            ]
        }
        mock_post.return_value.raise_for_status = MagicMock()

        results = [
            SearchResult(title=f"P{i}", url=f"https://p{i}.com", source="arxiv", category="papers", domain=f"p{i}.com")
            for i in range(5)
        ]

        reranked = rerank("test", results, top_n=3)
        # top_n=3，但 mock 只返回 1 条
        assert len(reranked) == 1

    @patch.dict("os.environ", {"SILICONFLOW_API_KEY": "sk-test"})
    @patch("httpx.Client.post")
    def test_explicit_api_key(self, mock_post: MagicMock):
        """优先使用显式传入的 api_key"""
        mock_post.return_value.json.return_value = {"results": []}
        mock_post.return_value.raise_for_status = MagicMock()

        results = [
            SearchResult(title="A", url="https://a.com", source="arxiv", category="papers", domain="a.com"),
        ]

        rerank("test", results, api_key="sk-explicit")
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        headers = call_args.kwargs.get("headers", {})
        assert "sk-explicit" in headers.get("Authorization", "")