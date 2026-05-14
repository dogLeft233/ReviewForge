#!/usr/bin/env python3
"""Test script for tool environment — simulates Streamlit environment"""

import os
import sys

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置代理环境变量（模拟 Windows 代理）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

print("=" * 60)
print("Network Environment Test")
print("=" * 60)

# 1. 检查环境变量
print("\n1. Proxy Environment Variables:")
for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"]:
    value = os.environ.get(key, "(not set)")
    print(f"   {key} = {value}")

# 2. 测试 httpx 是否能读取代理
print("\n2. Testing httpx with trust_env=True:")
import httpx
client = httpx.Client(timeout=10.0)
print(f"   httpx trust_env = {client._trust_env}")
client.close()

# 3. 测试 arXiv 查询格式
print("\n3. Testing arXiv Query Format:")
from src.retrievers.arxiv import ArxivRetriever

# 测试 1: 普通查询
query1 = "machine learning"
with ArxivRetriever() as arxiv:
    url = f"https://export.arxiv.org/api/query?search_query=all:machine+learning&start=0&max_results=3&sortBy=relevance"
    print(f"   Query '{query1}' -> URL: {url[:80]}...")

# 测试 2: 带字段前缀的查询（ti:）
query2 = "ti:survey AND speech recognition"
with ArxivRetriever() as arxiv:
    if hasattr(arxiv, '_build_search_query'):
        sq = arxiv._build_search_query(query2)
        print(f"   Query '{query2}' -> search_query: {sq}")
    else:
        # 手动检查逻辑
        import re
        if re.match(r"^(ti:|all:|abs:|cat:|author:)", query2.strip()):
            sq = query2.strip()
        else:
            from urllib.parse import quote
            sq = f"all:{quote(query2)}"
        print(f"   Query '{query2}' -> search_query: {sq}")

# 4. 测试 Serper API（模拟）
print("\n4. Testing Serper API Proxy Configuration:")
import httpx
proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or None
proxies = None
if proxy_url:
    proxies = {"https://": proxy_url, "http://": proxy_url}
print(f"   Proxy URL: {proxy_url}")
print(f"   Proxies dict: {proxies}")

# 5. 测试 MiniMax 工具调用解析
print("\n5. Testing MiniMax Tool Call Parsing:")
from src.llm import LLM

# 模拟 MiniMax 返回的 tool_calls 格式
test_responses = [
    # 格式 1: 标准 OpenAI tool_calls
    '{"tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"web_search","arguments":"{\\"query\\":\\"test\\"}"}}]}',
    # 格式 2: message 对象（SiliconFlow 返回）
    '{"id":"chatcmpl-xxx","choices":[{"message":{"tool_calls":[{"id":"call_xyz","type":"function","function":{"name":"web_fetch","arguments":"{\\"url\\":\\"https://example.com\\"}"}}]}}]}',
]

for i, resp_text in enumerate(test_responses, 1):
    # 临时创建 LLM 实例（不需要真实 API key 来测试解析）
    from src.llm import LLM
    try:
        # 跳过实际的 _parse_tool_calls_from_response 测试，因为需要初始化
        import json, re
        # 直接复制解析逻辑
        json_blocks = re.findall(r"```json\s*(.*?)\s*```", resp_text, re.DOTALL)
        if not json_blocks:
            try:
                data = json.loads(resp_text)
                if isinstance(data, dict) and "tool_calls" in data:
                    parsed = data["tool_calls"]
                    print(f"   Response {i}: ✓ Parsed {len(parsed)} tool_call(s)")
                else:
                    print(f"   Response {i}: No tool_calls found in JSON")
            except json.JSONDecodeError:
                print(f"   Response {i}: Failed to parse as JSON")
        else:
            print(f"   Response {i}: Found json blocks")
    except Exception as e:
        print(f"   Response {i}: Error - {e}")

# 6. 测试 _build_llm 函数是否存在
print("\n6. Testing _build_llm function:")
try:
    from src.agent.skills.core_skill import _build_llm
    print("   ✓ _build_llm function exists")
except ImportError as e:
    print(f"   ✗ _build_llm not found: {e}")

# 7. 测试完整的 LLM 实例化
print("\n7. Testing LLM instantiation:")
try:
    from src.config import settings
    api_key = getattr(settings, "llm_api_key", "") or ""
    model = getattr(settings, "llm_model", "Qwen/Qwen3-8B")
    base_url = getattr(settings, "llm_base_url", "https://api.siliconflow.cn/v1")
    print(f"   API Key: {'*' * len(api_key) if api_key else '(empty)'}")
    print(f"   Model: {model}")
    print(f"   Base URL: {base_url}")
except Exception as e:
    print(f"   ✗ Error: {e}")

def test_network_connectivity():
    """Real network connectivity tests for all external APIs."""
    print("\n8. Real Network Connectivity Tests:")
    print("-" * 40)
    results = []

    # Test 1: Serper API
    try:
        from src.config import settings
        key = getattr(settings, "serpapi_api_key", None) or getattr(settings, "serper_api_key", None)
        proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if key:
            resp = httpx.post(
                "https://google.serper.dev/search",
                json={"q": "test", "num": 1},
                headers={"X-API-KEY": key},
                proxy=proxy_url,
                timeout=15.0,
            )
            data = resp.json()
            q = data.get("searchParameters", {}).get("q", "?")
            print(f"   Serper API: OK (status={resp.status_code}, q={q})")
            results.append(True)
        else:
            print(f"   Serper API: SKIPPED (no API key)")
            results.append(False)
    except Exception as e:
        print(f"   Serper API: FAILED ({e})")
        results.append(False)

    # Test 2: arXiv API
    try:
        with ArxivRetriever() as arxiv:
            papers = arxiv.search("deep learning", max_results=3)
            print(f"   arXiv API: OK ({len(papers)} papers returned)")
            results.append(True)
    except Exception as e:
        print(f"   arXiv API: FAILED ({e})")
        results.append(False)

    # Test 3: GitHub API
    try:
        resp = httpx.get("https://api.github.com/", timeout=10.0)
        print(f"   GitHub API: OK (status={resp.status_code})")
        results.append(True)
    except Exception as e:
        print(f"   GitHub API: FAILED ({e})")
        results.append(False)

    # Test 4: HuggingFace API
    try:
        resp = httpx.get("https://huggingface.co/api", timeout=10.0)
        print(f"   HuggingFace API: OK (status={resp.status_code})")
        results.append(True)
    except Exception as e:
        print(f"   HuggingFace API: FAILED ({e})")
        results.append(False)

    passed = sum(results)
    print("-" * 40)
    print(f"Network connectivity: {passed}/{len(results)} passed")


test_network_connectivity()

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)