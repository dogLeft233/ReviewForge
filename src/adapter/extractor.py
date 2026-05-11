"""从 explorer/writer 产出的 markdown / list 字段抽取结构化要素。

唯一职责：解析。不做 schema 适配，不做 LLM 调用。
所有解析方法都设计为"宁缺毋滥"：拿不准就返回空列表，留给上层 LLM/research 补。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# 内部数据类（仅在 adapter 包内流通；对外仍用 schema.py 的 Pydantic 模型）
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class RawPaper:
    title: str = ""
    year: int = 0
    authors: str = ""
    venue: str = ""
    summary: str = ""
    contribution: str = ""
    impact: str = ""


@dataclass(slots=True)
class RawTimelineEvent:
    year: int
    title: str
    category: str = ""
    description: str = ""
    paper_titles: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RawBenchmark:
    model: str = ""
    dataset: str = ""
    metric: str = ""
    score: float = 0.0
    year: int = 0
    url: str = ""


@dataclass(slots=True)
class RawFrontier:
    name: str
    description: str = ""
    importance: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# 通用工具
# ─────────────────────────────────────────────────────────────────────────────


_YEAR_RE = re.compile(r"(19|20)\d{2}")
_RANGE_RE = re.compile(r"((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2})")
_TITLE_QUOTE_RE = re.compile(r"[《〈]([^》〉]+)[》〉]")
_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")


def _first_year(text: str) -> int:
    m = _YEAR_RE.search(text or "")
    return int(m.group(0)) if m else 0


def _category_from_phase(phase: str) -> str:
    """把"3. **1990-2000：统计模型与混合系统应用**"压缩成"统计模型与混合系统应用"。"""
    s = re.sub(r"^\s*\d+\.\s*", "", phase or "").strip()
    s = _BOLD_RE.sub(lambda m: m.group(1), s).strip()
    # 去掉前面的年份范围
    s = re.sub(r"^\s*\d{4}\s*[-–]?\s*\d{0,4}[\s年代]*\s*[:：]?\s*", "", s).strip()
    s = s.strip("：:").strip()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 论文抽取（首选 markdown 表格，后备 searcher_papers / 时间线 bullet）
# ─────────────────────────────────────────────────────────────────────────────


_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")


def extract_papers_from_table(markdown: str) -> list[RawPaper]:
    """解析形如 `| 论文 | 作者(年, 会议) | 贡献 | 影响 |` 的 markdown 表格。

    返回顺序保留出现顺序；失败行直接跳过。
    """
    papers: list[RawPaper] = []
    if not markdown:
        return papers

    for raw_line in markdown.splitlines():
        m = _TABLE_ROW_RE.match(raw_line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        # 需要 ≥3 列，且第一列含有"《》"或斜体星号才认为是数据行
        if len(cells) < 2:
            continue
        first = cells[0]
        # 跳过表头/分隔行
        if set(first) <= set("-: ") or first.startswith("---"):
            continue
        # 跳过纯文字表头（如"论文 | 作者 | 贡献 | 影响"）
        if first in {"论文", "Paper", "标题"} or "论文" == first.strip():
            continue

        title = ""
        m_quote = _TITLE_QUOTE_RE.search(first)
        if m_quote:
            title = m_quote.group(1).strip()
        else:
            # 退化处理：去掉 markdown 装饰
            title = re.sub(r"^[*_`\s]+|[*_`\s]+$", "", first).strip()
        if not title or len(title) < 4:
            continue

        # 第二列：作者（年份, 会议）
        authors = ""
        venue = ""
        year = 0
        if len(cells) >= 2:
            meta = cells[1]
            year = _first_year(meta)
            # 括号里的内容当作 venue
            m_paren = re.search(r"[（(]([^)）]+)[)）]", meta)
            if m_paren:
                inner = m_paren.group(1)
                venue_parts = [
                    part.strip()
                    for part in re.split(r"[，,]", inner)
                    if part.strip() and not _YEAR_RE.fullmatch(part.strip())
                ]
                venue = "; ".join(venue_parts)
            authors = re.sub(r"[（(].*?[)）]", "", meta).strip()
            authors = re.sub(r"\d{4}年?", "", authors).strip("，, ")

        contribution = cells[2].strip() if len(cells) >= 3 else ""
        impact = cells[3].strip() if len(cells) >= 4 else ""

        papers.append(
            RawPaper(
                title=title,
                year=year,
                authors=authors,
                venue=venue,
                summary=contribution or impact,
                contribution=contribution,
                impact=impact,
            )
        )

    return papers


def extract_papers_from_searcher(searcher_papers: list[Any]) -> list[RawPaper]:
    out: list[RawPaper] = []
    for item in searcher_papers or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or "").strip()
        if not title:
            continue
        out.append(
            RawPaper(
                title=title,
                year=int(item.get("year") or 0),
                authors=str(item.get("authors") or item.get("author") or ""),
                venue=str(item.get("venue") or item.get("source") or ""),
                summary=str(item.get("summary") or item.get("description") or item.get("abstract") or ""),
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 时间线抽取
# ─────────────────────────────────────────────────────────────────────────────


# 匹配像 "3. **1990-2000：统计模型..."、"#### **1990-2000：..."
_PHASE_RE = re.compile(r"^\s*(?:#+\s*)?(?:\d+\.\s*)?\*\*([^*\n]{4,80})\*\*", re.MULTILINE)


def extract_timeline(stage2_timeline_md: str, papers: list[RawPaper]) -> list[RawTimelineEvent]:
    """优先用论文表里的 (year, title) 当时间线事件。

    阶段名（category）从 stage2_timeline 的 phase 标题就近匹配；匹不上时留空。
    不再像旧版那样把每条 bullet 当独立 event，避免出现 "处理长时依赖" 这种碎片。
    """
    events: list[RawTimelineEvent] = []
    if not papers:
        return events

    # 解析所有 phase header 的 (起始年, 结束年, 名称)
    phases: list[tuple[int, int, str]] = []
    for m in _PHASE_RE.finditer(stage2_timeline_md or ""):
        header = m.group(1)
        rng = _RANGE_RE.search(header)
        if rng:
            y0, y1 = int(rng.group(1)), int(rng.group(2))
        else:
            single = _YEAR_RE.search(header)
            if not single:
                continue
            y0 = int(single.group(0))
            y1 = y0 + 50  # "2018 至今" 这种
        name = _category_from_phase(header)
        if name:
            phases.append((y0, y1, name))

    def _category_for(year: int) -> str:
        if not year:
            return ""
        # 取第一个区间
        for y0, y1, name in phases:
            if y0 <= year <= y1:
                return name
        return ""

    seen_titles: set[str] = set()
    for p in papers:
        if not p.title or not p.year:
            continue
        key = p.title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        events.append(
            RawTimelineEvent(
                year=p.year,
                title=p.title,
                category=_category_for(p.year),
                description=p.contribution or p.summary,
                paper_titles=[p.title],
            )
        )

    events.sort(key=lambda e: (e.year, e.title))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 抽取
# ─────────────────────────────────────────────────────────────────────────────


def extract_benchmarks_from_field(stage3_benchmarks: list[Any]) -> list[RawBenchmark]:
    """解析 explorer_report.stage3_benchmarks 列表（每项含 name/url/dataset/metric/leaderboard）。

    leaderboard 通常是 markdown 表 `| Model | Score | Year |` 形态；解析失败则只保留
    benchmark 自身（model 留空，等 LLM/research 补）。
    """
    out: list[RawBenchmark] = []
    for bm in stage3_benchmarks or []:
        if not isinstance(bm, dict):
            continue
        ds = str(bm.get("dataset") or bm.get("name") or "").strip()
        url = str(bm.get("url") or bm.get("leaderboard_url") or "").strip()
        # explorer 经常把 URL 塞进 name，尽量挪到 url，dataset 留空待 research
        if ds.startswith(("http://", "https://")):
            url = url or ds
            ds = ""
        # url 末尾常带半个 markdown 链接，截掉 "](" 后面
        if "](" in url:
            url = url.split("](", 1)[0]
        metric = str(bm.get("metric") or "").strip()
        leaderboard = str(bm.get("leaderboard") or "")

        rows = _parse_leaderboard(leaderboard, default_dataset=ds, default_metric=metric, default_url=url)
        if rows:
            out.extend(rows)
        else:
            out.append(RawBenchmark(model="", dataset=ds, metric=metric, url=url))
    return out


_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _parse_leaderboard(
    md: str,
    *,
    default_dataset: str,
    default_metric: str,
    default_url: str,
) -> list[RawBenchmark]:
    rows: list[RawBenchmark] = []
    if not md:
        return rows

    # 找到所有表格行（| ... |）
    for raw_line in md.splitlines():
        m = _TABLE_ROW_RE.match(raw_line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if len(cells) < 2:
            continue
        first = cells[0]
        if set(first) <= set("-: "):
            continue
        if first.lower() in {"model", "method", "system", "approach", "模型", "方法"}:
            continue

        model = _BOLD_RE.sub(lambda mm: mm.group(1), first).strip("`*_ ").strip()
        if not model or len(model) > 80:
            continue

        # 余下列里找首个数字（百分比优先）
        score_val = 0.0
        score_cell = ""
        for c in cells[1:]:
            mm = _PCT_RE.search(c)
            if mm:
                score_val = float(mm.group(1))
                score_cell = c
                break
        if not score_cell:
            for c in cells[1:]:
                mm = _NUM_RE.search(c)
                if mm:
                    candidate = float(mm.group(1))
                    # 跳过显然是 year 的 4 位整数
                    if 1900 <= candidate <= 2100 and abs(candidate - int(candidate)) < 1e-9:
                        continue
                    score_val = candidate
                    score_cell = c
                    break

        # 末尾找年份
        year = 0
        for c in reversed(cells):
            y = _first_year(c)
            if y:
                year = y
                break

        rows.append(
            RawBenchmark(
                model=model,
                dataset=default_dataset,
                metric=default_metric,
                score=score_val,
                year=year,
                url=default_url,
            )
        )

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 前沿趋势抽取
# ─────────────────────────────────────────────────────────────────────────────


_FRONTIER_BULLET_RE = re.compile(
    r"-\s*\*\*([^*\n]{2,40})\*\*\s*[:：]\s*([^\n]+)"
)

# 这些名字一看就是 markdown 标签而非真正的趋势方向，跳过
_FRONTIER_NAME_BLACKLIST = {
    "url", "链接", "排行榜", "leaderboard", "site", "网站",
    "metric", "指标", "dataset", "数据集", "name", "名称",
    "description", "描述", "当前最佳成绩", "sota",
}


def extract_frontiers(*texts: str, max_count: int = 8) -> list[RawFrontier]:
    seen: set[str] = set()
    out: list[RawFrontier] = []
    joined = "\n".join(t or "" for t in texts)
    for m in _FRONTIER_BULLET_RE.finditer(joined):
        name = m.group(1).strip()
        desc = m.group(2).strip()
        key = name.lower()
        if key in seen or key in _FRONTIER_NAME_BLACKLIST:
            continue
        # 描述如果是 URL 或 markdown 链接就跳过
        if desc.startswith(("http://", "https://", "[http")):
            continue
        seen.add(key)
        out.append(RawFrontier(name=name, description=desc))
        if len(out) >= max_count:
            break
    return out
