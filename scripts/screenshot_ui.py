#!/usr/bin/env python3
"""用 Playwright 截取 Streamlit UI 各 Tab 截图，用于论文。

用法:
    # 1. 先启动 Streamlit（另一个终端）
    streamlit run ui/app.py

    # 2. 再跑截图
    python scripts/screenshot_ui.py

    # 指定 URL
    python scripts/screenshot_ui.py --url http://localhost:8501

输出: docs/figures/ui_*.png
"""

import argparse
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tab 名称 → 文件名
TABS = [
    ("Overview", "ui_overview"),
    ("Timeline", "ui_timeline"),
    ("Method Map", "ui_method_map"),
    ("Frontier", "ui_frontier"),
    ("Benchmark", "ui_benchmark"),
    ("Knowledge Graph", "ui_knowledge_graph"),
]


def screenshot_tab(page, tab_name: str, filename: str) -> None:
    """点击指定 Tab 并截图"""
    print(f"  → {tab_name}...")
    tab = page.get_by_role("tab", name=tab_name)
    tab.click()
    # 等待 Tab 内容渲染（Streamlit 动态加载）
    time.sleep(2)
    page.screenshot(
        path=str(OUT_DIR / f"{filename}.png"),
        full_page=True,
    )
    print(f"    ✓ saved to {filename}.png")


def main():
    parser = argparse.ArgumentParser(description="ReviewForge UI 截图")
    parser.add_argument("--url", default="http://localhost:8501", help="Streamlit URL")
    parser.add_argument("--tabs", nargs="+", default=None,
                        help="指定截哪些 Tab（默认全部）")
    args = parser.parse_args()

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        print(f"Opening {args.url} ...")
        page.goto(args.url, wait_until="networkidle", timeout=30000)
        # 等 Streamlit 完全加载
        time.sleep(3)

        tabs_to_capture = args.tabs or [t[0] for t in TABS]
        for tab_name, filename in TABS:
            if tab_name in tabs_to_capture:
                screenshot_tab(page, tab_name, filename)

        browser.close()

    print(f"\n共 {len(tabs_to_capture) if args.tabs else len(TABS)} 张截图 → {OUT_DIR}")


if __name__ == "__main__":
    main()
