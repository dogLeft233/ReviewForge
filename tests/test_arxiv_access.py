#!/usr/bin/env python3
"""测试 arXiv 搜索是否可访问"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for p in (str(_PROJECT_ROOT), str(_SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.retrievers import ArxivRetriever


def main():
    print("=" * 60)
    print("arXiv API 连接测试")
    print("=" * 60)

    errors = []

    # 1. 基本搜索
    print("\n[1] search('LoRA large language model', max_results=5)")
    try:
        with ArxivRetriever() as arxiv:
            papers = arxiv.search("LoRA large language model", max_results=5)
        print(f"    → 返回 {len(papers)} 篇")
        for i, p in enumerate(papers, 1):
            print(f"    [{i}] {p.title[:60]}")
            print(f"        authors: {', '.join(p.authors[:2])}{'...' if len(p.authors) > 2 else ''}")
            print(f"        year={p.year}  subject={p.method_category}")
    except Exception as e:
        print(f"    ❌ 异常: {e}")
        errors.append(f"search: {e}")

    # 2. 按分类搜索
    print("\n[2] search_by_category('diffusion model', category='cs.LG', max_results=5)")
    try:
        with ArxivRetriever() as arxiv:
            papers = arxiv.search_by_category(
                "diffusion model", category="cs.LG", max_results=5
            )
        print(f"    → 返回 {len(papers)} 篇")
        for i, p in enumerate(papers, 1):
            print(f"    [{i}] {p.title[:60]}")
    except Exception as e:
        print(f"    ❌ 异常: {e}")
        errors.append(f"search_by_category: {e}")

    # 3. 搜索结果有效性（验证解析）
    print("\n[3] search('xyzabc123nonexistent xyz', max_results=3)")
    try:
        with ArxivRetriever() as arxiv:
            papers = arxiv.search("xyzabc123nonexistent xyz", max_results=3)
        print(f"    → 返回 {len(papers)} 篇（arXiv 对任意字符串都返回结果，这是正常行为）")
        print(f"    → 其中第一篇: {papers[0].title[:50] if papers else 'N/A'}")
    except Exception as e:
        print(f"    ⚠️ 异常（限流中）: {e}")

    # 汇总
    print(f"\n{'=' * 60}")
    if errors:
        print(f"❌ {len(errors)} 项失败:")
        for e in errors:
            print(f"   • {e}")
    else:
        print("✅ 全部通过！arXiv 搜索可正常访问")
    print("=" * 60)


if __name__ == "__main__":
    main()
