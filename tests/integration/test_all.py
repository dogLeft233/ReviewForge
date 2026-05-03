"""集成测试——实测各检索器是否正常工作

用法:
  uv run python tests/integration/test_all.py
  uv run python tests/integration/test_all.py --skip-hn   # 跳过 HN（Algolia 有时不稳定）
"""

import sys
import time

import httpx


def test_arxiv():
    """测试 arXiv 检索"""
    print("\n1️⃣  arXiv ... ", end="", flush=True)
    url = "http://export.arxiv.org/api/query?search_query=all:big+data+processing&start=0&max_results=3&sortBy=relevance"
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    assert resp.status_code == 200, f"arXiv returned {resp.status_code}"
    assert "xml" in resp.text[:50].lower()
    print(f"✅ ({len(resp.text)} bytes)")


def test_semantic_scholar():
    """测试 Semantic Scholar 检索（注意速率限制 1 req/s）"""
    print("\n2️⃣  Semantic Scholar ... ", end="", flush=True)
    url = "https://api.semanticscholar.org/graph/v1/paper/search?query=big+data&limit=3&fields=title,year,venue,citationCount"
    resp = httpx.get(url, timeout=30)
    if resp.status_code == 429:
        print(f"⏸️ （速率限制，跳过）")
        return
    assert resp.status_code == 200, f"S2 returned {resp.status_code}"
    data = resp.json()
    assert "data" in data, "S2 response missing 'data'"
    assert len(data["data"]) > 0, "S2 returned 0 results"
    print(f"✅ ({len(data['data'])} papers)")
    return data


def test_dblp():
    """测试 DBLP 检索"""
    print("\n3️⃣  DBLP ... ", end="", flush=True)
    url = "https://dblp.org/search/publ/api?q=big+data+processing&h=3&format=json"
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    assert resp.status_code == 200, f"DBLP returned {resp.status_code}"
    data = resp.json()
    print(f"✅")
    return data


def test_github():
    """测试 GitHub 仓库检索"""
    print("\n4️⃣  GitHub ... ", end="", flush=True)
    url = "https://api.github.com/search/repositories?q=big+data+processing+sort:stars&per_page=3"
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    assert resp.status_code == 200, f"GitHub returned {resp.status_code}"
    data = resp.json()
    items = data.get("items", [])
    assert len(items) > 0, "GitHub returned 0 repos"
    print(f"✅ ({items[0]['full_name']}, {items[0]['stargazers_count']} stars)")


def test_papers_with_code():
    """测试 Papers With Code 检索（已废弃，API 迁移至 HuggingFace）"""
    print("\n5️⃣  Papers With Code (deprecated) ... ", end="", flush=True)
    print("⚠️ （API 已迁移至 HuggingFace，返回 HTML 而非 JSON）")


def test_huggingface():
    """测试 HuggingFace 模型检索"""
    print("\n6️⃣  HuggingFace ... ", end="", flush=True)
    url = "https://huggingface.co/api/models?search=big+data&sort=downloads&direction=-1&limit=3"
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    assert resp.status_code == 200, f"HF returned {resp.status_code}"
    data = resp.json()
    assert len(data) > 0, "HF returned 0 models"
    print(f"✅ ({data[0].get('modelId', 'N/A')})")


def test_hackernews():
    """测试 HackerNews Algolia 检索"""
    print("\n7️⃣  Hacker News ... ", end="", flush=True)
    url = "https://hn.algolia.com/api/v1/search?query=big+data&tags=story&hitsPerPage=3&numericFilters=points>10"
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    assert resp.status_code == 200, f"HN returned {resp.status_code}"
    data = resp.json()
    hits = data.get("hits", [])
    print(f"✅ ({len(hits)} stories, top: {hits[0].get('points', 0)} pts)")
    return data


def main() -> int:
    skip_hn = "--skip-hn" in sys.argv

    print("=" * 50)
    print("  ReviewForge 集成测试 — 实时 API 验证")
    print("=" * 50)

    tests = [
        ("arXiv", test_arxiv),
        ("Semantic Scholar", test_semantic_scholar),
        ("DBLP", test_dblp),
        ("GitHub", test_github),
        ("Papers With Code", test_papers_with_code),
        ("HuggingFace", test_huggingface),
    ]

    if not skip_hn:
        tests.append(("Hacker News", test_hackernews))
    else:
        print("\n(skipping Hacker News per --skip-hn)")

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ ({type(e).__name__}: {e})")
            failed += 1

        time.sleep(0.3)  # 简单节流

    print(f"\n{'=' * 50}")
    print(f"  结果: {passed} 通过 / {failed} 失败 / {passed + failed} 总计")
    print(f"{'=' * 50}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
