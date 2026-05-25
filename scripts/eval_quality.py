#!/usr/bin/env python3
"""引用质量验证 + LLM-as-Judge 内容评估

用法:
    # 仅验证引用（零 LLM 消耗，用免费 arXiv API）
    python scripts/eval_quality.py --citations-only

    # 仅 LLM-as-Judge 内容评分（~3 次 LLM 调用）
    python scripts/eval_quality.py --content-only

    # 全部评估
    python scripts/eval_quality.py

API 调用预算：LLM-as-Judge 3 次（每个 topic 1 次）
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ["PYTHONPATH"] = str(_ROOT / "src")

import httpx
from src.llm import LLM, Message
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger("eval_quality")

RESULTS_DIR = _ROOT / "experiment_results"

# ── 内容评估 prompt（5 维 Likert，参考 SurveyX） ──

JUDGE_SYSTEM = """你是一位资深的学术评审专家。请评估以下由 AI 系统自动生成的文献综述质量。

从以下 5 个维度打分（1-5 分，5 分为最高）：

1. **覆盖度 (Coverage)**：综述是否全面覆盖了该领域的主要方向、经典工作和最新进展？
2. **结构组织 (Organization)**：章节划分是否合理？逻辑是否清晰？分类体系是否恰当？
3. **相关性 (Relevance)**：内容是否紧扣主题？有无无关或偏离的讨论？
4. **综合分析 (Synthesis)**：是否超越简单罗列，进行了跨方法的对比分析和归纳？
5. **批判性分析 (Critical Analysis)**：是否指出了各方法的局限性、开放问题和未来方向？

请以 JSON 格式回复，不要包含其他文字：
{
  "coverage": <1-5>,
  "organization": <1-5>,
  "relevance": <1-5>,
  "synthesis": <1-5>,
  "critical_analysis": <1-5>,
  "overall": <1-5>,
  "comment": "<一句话总体评价，中文>"
}"""


def extract_references_from_survey(step3_path: Path) -> list[dict]:
    """从 step3 JSON 提取参考文献列表"""
    data = json.loads(step3_path.read_text(encoding="utf-8"))
    refs = []

    wr = data.get("writer_report", {})
    if not isinstance(wr, dict):
        return refs

    refs_text = wr.get("references", "")
    if not refs_text:
        return refs

    # 解析 N. Author. Title... [J]/[C]... 格式（可能用普通数字或带圈数字）
    # 带圈数字: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳
    pattern = r'(?:^|\n)\s*(?:\d{1,3}\.|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*(.+?)(?=\n\s*(?:\d{1,3}\.|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*|\n\s*$)'
    matches = re.findall(pattern, refs_text, re.DOTALL)

    for content in matches:
        content = content.strip()
        # 标题通常在作者之后，格式: Author, ... Title[J] 或 Author. Title[C]
        # 跳过作者部分（从第一个句号后开始到 [J] 或 [C] 之前）
        # 更鲁棒的方式：找第一个大写字母开头的长文本段作为标题
        title = ""
        # 尝试匹配 "作者们. 标题[J]" 或 "作者, 标题[C]"
        title_match = re.search(r'(?:[a-z]\s+)?([A-Z][^\.]{10,200}?)(?:\s*\[[JC]\])', content)
        if title_match:
            title = title_match.group(1).strip()
        else:
            # 回退：取第一个句号后的内容
            parts = content.split(". ", 1)
            title = parts[1][:150] if len(parts) > 1 else content[:150]
        refs.append({"raw": content[:300], "title": title})

    logger.info("从 %s 提取到 %d 条参考文献", step3_path.name, len(refs))
    return refs


def verify_citation_via_arxiv(title: str) -> dict:
    """通过 arXiv API 验证引用是否存在（免费，无需 API key）"""
    try:
        # 清理标题用于搜索
        query = re.sub(r'[^\w\s-]', ' ', title[:200])
        query = re.sub(r'\s+', ' ', query).strip()

        url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": f'ti:"{query}"',
            "max_results": 3,
        }
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, params=params)
        if resp.status_code != 200:
            return {"found": False, "error": f"HTTP {resp.status_code}"}

        # 简单检查：响应中是否有匹配的标题
        text = resp.text.lower()
        title_lower = title.lower()[:50]
        found = title_lower in text

        return {"found": found, "title_searched": title[:100]}

    except Exception as e:
        return {"found": False, "error": str(e)[:100]}


def run_citation_eval() -> dict:
    """对所有已完成的 topic 验证引用质量"""
    tmp_root = _ROOT / "tmp"
    results = {}

    for topic_dir in sorted(tmp_root.iterdir()):
        if not topic_dir.is_dir():
            continue
        step3 = topic_dir / "step3_writer_done.json"
        if not step3.exists():
            continue

        topic = topic_dir.name
        logger.info("验证引用: %s", topic)

        refs = extract_references_from_survey(step3)
        if not refs:
            results[topic] = {"total_refs": 0, "message": "无参考文献"}
            continue

        # 只验证前 10 条（避免过多 API 调用）
        sample = refs[:10]
        verified = 0
        for r in sample:
            res = verify_citation_via_arxiv(r["title"])
            if res["found"]:
                verified += 1
            time.sleep(0.5)  # arXiv API 礼貌间隔

        accuracy = verified / len(sample) if sample else 0
        results[topic] = {
            "total_refs": len(refs),
            "sample_checked": len(sample),
            "verified": verified,
            "accuracy": round(accuracy, 3),
        }
        logger.info("  %s: %d/%d 准确 (%.1f%%)", topic, verified, len(sample), accuracy * 100)

    return results


def run_content_eval(llm: LLM, topic: str, step3_path: Path) -> dict:
    """LLM-as-Judge：对一篇综述进行 5 维评分（1 次 LLM 调用）"""
    data = json.loads(step3_path.read_text(encoding="utf-8"))
    wr = data.get("writer_report", {})
    if not isinstance(wr, dict):
        return {"error": "无 writer_report"}

    # 构建评估文本：摘要 + 引言前 500 字 + 正文前 1000 字
    abstract = wr.get("abstract", "")
    introduction = wr.get("introduction", "")[:500]
    body = wr.get("body", "")[:1000]

    eval_text = f"""领域：{topic}

摘要：
{abstract}

引言（节选）：
{introduction}

正文（节选）：
{body}"""

    try:
        import httpx
        from src.config import settings as s

        payload = {
            "model": s.llm_model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": eval_text},
            ],
            "temperature": 0.1,
            "max_tokens": 4000,
        }
        resp = httpx.post(
            f"{s.llm_base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {s.llm_api_key}"},
            timeout=60.0,
        )
        msg = resp.json()["choices"][0]["message"]
        # DeepSeek v4-flash 把输出放在 reasoning_content 而非 content
        reply = msg.get("content", "") or msg.get("reasoning_content", "")

        logger.debug("  raw reply len=%d firstchars=%s", len(reply), reply[:100])

        # 方式1: 直接匹配 JSON 对象
        json_match = re.search(r'\{[^{}]*"coverage"[^{}]*\}', reply, re.DOTALL)
        if not json_match:
            # 方式2: 更宽松的匹配
            json_match = re.search(r'\{[\s\S]*?\}', reply)

        if json_match:
            try:
                scores = json.loads(json_match.group())
                logger.info("  %s 评分: overall=%s", topic, scores.get("overall", "N/A"))
                return scores
            except json.JSONDecodeError as je:
                return {"error": f"JSON解析失败: {je}", "raw_reply": reply[:300]}
        else:
            return {"error": "未找到JSON", "raw_reply": reply[:300]}
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="ReviewForge 质量评估")
    parser.add_argument("--citations-only", action="store_true", help="仅验证引用")
    parser.add_argument("--content-only", action="store_true", help="仅内容评分")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ── 引用验证 ──
    if not args.content_only:
        logger.info("=" * 50)
        logger.info("阶段 1/2: 引用质量验证（免费 arXiv API）")
        logger.info("=" * 50)
        citation_results = run_citation_eval()
        all_results["citations"] = citation_results

        print("\n引用验证结果：")
        for topic, r in citation_results.items():
            if r.get("total_refs", 0) == 0:
                print(f"  {topic}: 无参考文献")
            else:
                print(f"  {topic}: {r['verified']}/{r['sample_checked']} "
                      f"准确 ({r['accuracy']:.1%})")

    # ── LLM-as-Judge 内容评分 ──
    if not args.citations_only:
        logger.info("=" * 50)
        logger.info("阶段 2/2: LLM-as-Judge 内容评分（3 次 LLM 调用）")
        logger.info("=" * 50)

        llm = LLM(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            timeout_seconds=120.0,
            max_tokens=500,
        )

        content_results = {}
        tmp_root = _ROOT / "tmp"
        for topic_dir in sorted(tmp_root.iterdir()):
            if not topic_dir.is_dir():
                continue
            step3 = topic_dir / "step3_writer_done.json"
            if not step3.exists():
                continue

            topic_name = topic_dir.name
            logger.info("评估内容: %s", topic_name)
            scores = run_content_eval(llm, topic_name, step3)
            content_results[topic_name] = scores

        all_results["content_scores"] = content_results

        print("\n内容评分结果：")
        dims = ["coverage", "organization", "relevance", "synthesis", "critical_analysis", "overall"]
        header = f"{'Topic':<30} " + " ".join(f"{d:>12}" for d in dims)
        print(header)
        print("-" * 110)
        for topic, scores in content_results.items():
            if "error" in scores:
                print(f"{topic:<30} ERROR: {scores['error'][:60]}")
                continue
            row = f"{topic:<30} " + " ".join(
                f"{scores.get(d, 'N/A'):>12}" for d in dims
            )
            print(row)

    # ── 保存 ──
    out_path = RESULTS_DIR / "quality_eval.json"
    out_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("评估结果已保存: %s", out_path)


if __name__ == "__main__":
    main()
