"""测试 Ar5ivRetriever 的参考文献解析功能"""

import sys
sys.path.insert(0, ".")

from src.retrievers.ar5iv import Ar5ivRetriever


def test_sample_html():
    """用模拟的 ar5iv HTML 验证解析逻辑"""
    sample_html = """<html>
<body>
<ul class="ltx_biblist">
  <li id="bib.bib1" class="ltx_bibitem">
    <span class="ltx_tag ltx_tag_bibitem">[1]</span>
    <span class="ltx_bibblock"><span class="ltx_bibauthors">John Smith.</span></span>
    <span class="ltx_bibblock"><span class="ltx_bibtitle">A Great Paper About Things.</span></span>
    <span class="ltx_bibblock">CVPR, 2023.</span>
  </li>
  <li id="bib.bib2" class="ltx_bibitem">
    <span class="ltx_tag ltx_tag_bibitem">[2]</span>
    <span class="ltx_bibblock"><span class="ltx_bibauthors">Jane Doe.</span></span>
    <span class="ltx_bibblock"><span class="ltx_bibtitle">Another Significant Work.</span></span>
    <span class="ltx_bibblock">ICML, 2022.</span>
  </li>
  <li id="bib.bib3" class="ltx_bibitem">
    <span class="ltx_tag ltx_tag_bibitem">[3]</span>
    <span class="ltx_bibblock">Alice and Bob, "Yet Another Paper," NeurIPS, 2021.</span>
  </li>
</ul>
<section id="S1">
  <h1>1. Introduction</h1>
  <p>Recent work <cite><a class="ltx_ref" href="#bib.bib1">1</a></cite> shows great results.</p>
  <p>Other work <a class="ltx_ref" href="#bib.bib3">3</a> is also relevant.</p>
</section>
<section id="S2">
  <h2>2. Related Work</h2>
  <p>See <cite><a class="ltx_ref" href="#bib.bib2">2</a></cite> for details.</p>
</section>
</body></html>"""

    retriever = Ar5ivRetriever()

    # Test extract_bibliography
    bibs = retriever.extract_bibliography(sample_html)
    print(f"[Sample HTML] found {len(bibs)} references:")
    for b in bibs:
        print(f"  {b['cite_key']}: title={b['title']!r}, authors={b['authors']!r}, year={b['year']!r}")

    assert len(bibs) == 3, f"Expected 3, got {len(bibs)}"
    assert bibs[0]["cite_key"] == "bib.bib1"
    assert bibs[0]["title"] == "A Great Paper About Things"
    assert bibs[0]["authors"] == "John Smith."
    assert bibs[0]["year"] == "2023"
    assert bibs[1]["title"] == "Another Significant Work"
    assert bibs[2]["title"] == "Yet Another Paper"
    print("Sample HTML bibliography test PASSED")

    # Test extract_section_citations
    sections = retriever.extract_section_citations(sample_html)
    print(f"\n[Sample HTML] found {len(sections)} sections with citations:")
    for sec, keys in sections.items():
        print(f"  {sec!r}: {keys}")

    assert "1. Introduction" in sections
    assert set(sections["1. Introduction"]) == {"bib.bib1", "bib.bib3"}
    assert "2. Related Work" in sections
    assert set(sections["2. Related Work"]) == {"bib.bib2"}
    print("Sample HTML section citations test PASSED")


def test_real_ar5iv():
    """用真实 ar5iv 论文测试"""
    retriever = Ar5ivRetriever()

    # 测试多篇不同风格的论文
    test_papers = [
        ("2307.00235", "FCC / Wi-Fi 6E 相关论文（可能有 ltx_bibblock 纯文本）"),
        ("2103.14030", "Transformer 相关论文（更标准的格式）"),
    ]

    for arxiv_id, desc in test_papers:
        print(f"\n{'='*60}")
        print(f"Testing arXiv {arxiv_id}: {desc}")
        print(f"{'='*60}")

        bibs = retriever.fetch_bibliography(arxiv_id)

        if not bibs:
            print(f"  [WARN] No bibliography found for {arxiv_id}")
            continue

        print(f"  Found {len(bibs)} references:")
        for b in bibs[:10]:
            print(f"    {b['cite_key']}: {b['title'][:80]!r}  (year={b['year']}, authors={b['authors'][:40]!r})")
        if len(bibs) > 10:
            print(f"    ... and {len(bibs) - 10} more")

        # 验证：每个条目必须有 title
        titles_with_content = [b for b in bibs if b["title"]]
        ratio = len(titles_with_content) / len(bibs) if bibs else 0
        print(f"  Title extraction rate: {len(titles_with_content)}/{len(bibs)} ({ratio:.0%})")

        if ratio < 0.3:
            print(f"  [WARN] Title extraction rate too low ({ratio:.0%}), check HTML structure")
        else:
            print(f"  [OK] Bibliography parsing looks good")

        # 测试 section citations（只测一篇）
        if arxiv_id == "2307.00235":
            html = retriever.fetch_full_text(arxiv_id)
            sections = retriever.extract_section_citations(html)
            print(f"\n  Found {len(sections)} sections with citations:")
            for sec, keys in list(sections.items())[:5]:
                print(f"    {sec[:50]!r}: {keys[:5]}")
            if len(sections) > 5:
                print(f"    ... and {len(sections) - 5} more sections")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Sample HTML parsing")
    print("=" * 60)
    test_sample_html()

    print("\n" + "=" * 60)
    print("Test 2: Real ar5iv HTML parsing")
    print("=" * 60)
    test_real_ar5iv()

    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60)
