# ReviewForge 论文用图 — GPT Image 2 生成提示词

以下三组提示词用于生成论文中的系统架构图。推荐使用 GPT Image 2 (gpt-image-2)
模型，每组提示词独立生成一张图。

风格要求（贯穿全部三张图）：
- 学术论文风格，白色背景，适合黑白打印
- 配色：深蓝 #1a5276 为主色，浅蓝 #d4e6f1 为填充，橙色 #e67e22 为重点标注
- 圆角矩形节点，箭头用灰色 #555，标签用深灰 #333
- 中文字体用黑体/微软雅黑，英文用 Arial/Helvetica
- 所有文字清晰可读，字号分层（标题 > 模块名 > 说明文字）
- 整体尺寸 16:9 或 4:3 横版，适合双栏排版

==============================================================================
FIGURE 1: 系统总体架构图
==============================================================================

Generate a clean academic architecture diagram in horizontal layout (16:9 ratio)
with white background, suitable for a Chinese computer science journal paper.

OVERALL STRUCTURE:
A horizontal pipeline flowing from left to right, divided into 4 main stages
connected by thick arrows. Below the pipeline, show a "中间持久化" (checkpoint)
layer connected with dashed lines. At the bottom right, show the output UI.

STAGE 1 — "阶段一: Explorer 探索器" (left side, light blue rounded box):
- Title at top: "Explorer" in bold English + "领域探索智能体" in Chinese
- Inside, show three sub-stages stacked vertically:
  - "Stage 1: 领域概况" with icon of magnifying glass
  - "Stage 2: 经典工作" (parallel symbol next to Stage 3)
  - "Stage 3: SOTA/前沿" (parallel symbol)
- Below: "Stage 2 ∥ Stage 3 并行执行" small annotation
- Output arrow pointing right with label: "ExplorerReport"
- Show ExplorerReport contents in a small callout: "领域定义 / 核心概念 /
  经典论文 / 时间线 / Benchmark / 趋势"

STAGE 2 — "阶段二: Searcher 检索器" (center-left):
- Title: "Searcher" + "双路并行检索智能体"
- Inside, show TWO parallel paths side by side:
  PATH A (left): "BFSSearcher" with sub-text "BFS 引文图搜索 (深度)"
    - Show tiny loop icon with "L=2 层扩展"
  PATH B (right): "MultiSourceSearcher" with sub-text "多源检索 (广度)"
    - Show small icons for: arXiv, GitHub, HuggingFace, Semantic Scholar
- Below both paths: "ThreadPoolExecutor 并行执行" annotation
- Output arrow with label: "PaperNode[] + 检索摘要文本"

STAGE 3 — "阶段三: Writer 写作器" (center-right):
- Title: "Writer" + "两阶段结构化写作智能体"
- Inside, show two phases:
  - "Phase 0: 规划" → outputs: "候选标题 / 分类体系 / 章节要点"
  - "Phase 1: 各章展开" → shows parallel groups:
    "标题 ∥ 引言" → "正文 (含方法对比表)" →
    "结论 ∥ 参考文献 ∥ 关键词" → "摘要 (最后提炼)"
- Output arrow with label: "WriterReport"

STAGE 4 — "Adapter 适配器" (right side):
- Title: "Adapter" + "保守适配管线"
- Inside, three thin stages:
  - "① 规则提取" → "② LLM 校验" → "③ 可选增强 (OpenAI)"
- Show "needs_research 缺口追踪" as a persistent vertical tracker on the side
- Output arrow pointing to UI

CHECKPOINT LAYER (bottom, dashed border):
- Three JSON file icons connected by dashed vertical lines to each stage:
  - "step1_explorer_done.json"
  - "step2_searcher_done.json"
  - "step3_writer_done.json"
- Annotation: "--resume 断点恢复"

OUTPUT (bottom right):
- "Streamlit UI" box with 7 small tab icons:
  "Overview | Timeline | Method Map | Frontier | Benchmark | Knowledge Graph | Ask Agent"

CONNECTING ELEMENTS:
- Main pipeline arrow flowing left to right across all stages (thick blue arrow)
- Dashed lines from each stage down to its checkpoint JSON
- Small annotation: "ExplorerReport 上下文注入" showing data flow from Stage 1
  feeding into both Stage 2 and Stage 3

COLOR SCHEME:
- Stage boxes: light blue fill (#d4e6f1), dark blue border (#1a5276)
- Arrows: dark gray (#555), 2px stroke
- Checkpoint boxes: light green fill (#d5f5e3), dashed border
- Text: dark gray (#333), bold for titles
- Background: pure white (#ffffff)

TEXT ANNOTATIONS (place around the diagram):
- Top center in bold: "图1 ReviewForge 系统总体架构"
- Bottom right small: "ExplorerReport 贯穿全流程，为下游提供领域知识上下文"

==============================================================================
FIGURE 2: BFS 引文搜索算法流程图
==============================================================================

Generate a detailed algorithm flow diagram showing the dual-stage BFS citation
search process. Vertical layout with two clearly separated stages, white background,
academic style, 4:3 ratio.

TITLE at top: "BFSSearcher: 基于重排序的 BFS 引文图搜索" in bold

====================================================================
STAGE 1 — "Search Phase (搜索阶段)" — large box with light blue fill
====================================================================

Flow from top to bottom within Stage 1:

Step 1: "ExplorerReport" (input box, rounded, dark blue)
  ↓ (arrow labeled "注入领域知识")

Step 2: "LLM 搜索词生成" (box)
  - Inside: "基于 ExplorerReport 生成 N 个精准搜索词"
  - Small annotation: "含 search reasoning"

  ↓ (arrow labeled "N queries")

Step 3: "SerpAPI Google 检索" (box)
  - Inside: "site:arxiv.org 限定搜索"
  - Show small Google logo + arxiv.org text

  ↓ (arrow labeled "提取 arXiv ID")

Step 4: "ar5iv 摘要获取" (box)
  - Inside: "多线程获取 HTML5 全文 → 解析摘要"
  - Small annotation: "ThreadPoolExecutor"

  ↓ (arrow labeled "候选论文列表")

Step 5: "Reranker 批量打分" (highlighted box, orange border)
  - Inside: "Qwen3-Reranker-8B 相关性打分"
  - Show score bar visualization: [0.98 ████] [0.45 ██] [0.12 █]

  ↓ (arrow labeled "score ≥ 0.3")

Step 6: "过滤后论文池" (box, green border)
  - Inside: "按 select_score 降序排列，去重 (按 arXiv ID)"

====================================================================
BRIDGE between stages:
  - Arrow from Stage 1 bottom to Stage 2 top
  - Label: "取 Top-K 论文 (默认 15 篇) 进入 BFS 扩展"
====================================================================

STAGE 2 — "Expand Phase (BFS 扩展阶段)" — large box with light orange fill
====================================================================

Start with a LOOP indicator: "for layer = 1 to L (默认 L=2)" in bold

Step 7: "BFS 当前层论文池" (box)
  ↓

Step 8: "ar5iv 参考文献解析" (highlighted box, orange border — KEY INNOVATION)
  - Inside two sub-methods shown side by side:
    - "方法1 (优先): 语义解析" → <span class="ltx_bibtitle">,
      <span class="ltx_bibauthors">, <span class="ltx_bibyear">
    - "方法2 (回退): 原始文本解析" → 引号匹配提取标题
  - Annotation: "BeautifulSoup HTML 解析, 零外部 API 依赖"

  ↓ (arrow labeled "参考文献标题列表")

Step 9: "SerpAPI 标题反查" (box)
  - Inside: "每篇参考文献标题 → Google 搜索 → 提取 arXiv ID"
  - Small annotation: "search_ref_by_title()"

  ↓ (arrow labeled "新 arXiv ID 列表")

Step 10: "ar5iv 摘要获取 + Reranker 打分" (box)
  - Same as Steps 4-5 but annotated "同搜索阶段流程"

  ↓ (arrow labeled "score ≥ threshold")

Step 11: "合并到论文池，去重 (按 arXiv ID)" (box)

LOOP BACK arrow (curved, from Step 10 back to Step 7):
  - Label: "layer++ 继续扩展, 直至 layer > L 或无可扩展论文"

====================================================================
OUTPUT at bottom:
====================================================================
Final box: "最终论文列表" with annotation:
  "每篇论文含: paper_id (arXiv ID), title, abstract, depth (BFS 层数),
   select_score (Reranker 分数), source (检索来源)"

====================================================================
SIDE ANNOTATIONS:
====================================================================
- Left side, aligned with Stage 2: "ar5iv.org 免费服务 (无 API Key)"
- Right side, aligned with Stage 1-2 transition: "与 PaSa 架构对比:
  本系统用 Reranker 替代 LLM Selector, 降低成本"

==============================================================================
FIGURE 3: 保守适配管线图
==============================================================================

Generate a vertical pipeline diagram showing the three-stage conservative
adaptation process. White background, academic style, 16:9 ratio, with
emphasis on the "needs_research" gap tracking mechanism.

TITLE at top: "Adapter: "宁缺毋滥" 三阶段保守适配管线" in bold

====================================================================
INPUT (top):
====================================================================
Large box: "PipelineResult (原始 Agent 输出)"
  - Inside show text: "嵌套 JSON, 含半结构化 Markdown / 表格 / 列表"
  - Arrow down to Stage 1

====================================================================
STAGE 1 (upper third): "① 规则提取 (script_adapter.py)" — blue box
====================================================================
Center: a large "确定性解析器" block with multiple parsing modules:

Parser modules shown as small boxes:
  - "论文解析" → "Markdown 表格 / classic_works / searcher_papers"
  - "时间线解析" → "阶段标题 / 年份范围 / 论文关联"
  - "Benchmark 解析" → "Leaderboard 表格 / 指标数值"
  - "前沿解析" → "**trend**: 模式匹配 / SOTA 文本"
  - "知识图谱构建" → "启发式关系推理"

On the RIGHT side of Stage 1, show a "needs_research 生成器" (orange box):
  - For each unfilled field → generates ResearchTask{target, reason, hint}
  - Show example: "target: paper:transformer.url, reason: 原始数据未提供链接,
    hint: 为论文找到 arXiv DOI URL"

Output arrow from Stage 1 labeled: "初始 VisualizationData (字段可能为空)"

====================================================================
STAGE 2 (middle third): "② LLM 校验/补全 (llm_refiner.py)" — green box
====================================================================

Show 6 independent refinement modules (each as a small box), all running
in parallel with independent error handling:

  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ 论文精炼 │ │ 时间线   │ │ 方法     │
  │ Papers   │ │ Timeline │ │ Methods  │
  └──────────┘ └──────────┘ └──────────┘
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │Benchmark │ │ 前沿     │ │ 概览     │
  │  精炼    │ │Frontiers │ │Overview  │
  └──────────┘ └──────────┘ └──────────┘

Key constraint callout (bold, red text):
  "LLM 禁止编造: 作者 / 年份 / URL / 分数 等事实字段
   违反时字段留空并回退到脚本输出"

Show that:
  - Satisfied needs_research entries → removed from list
  - Unsatisfied entries → remain in needs_research

Output arrow: "精炼后 VisualizationData"

====================================================================
STAGE 3 (lower third): "③ 可选增强 (openai_responses_enhancer.py)" — purple box
====================================================================

Show 6 focused modules (can be smaller, lighter style since optional):

  "paper_timeline_links" (URL 补链)
  "method_details" (pros / cons)
  "method_links" (论文关联修复)
  "frontier_sources" (web search 找最新文献)
  "frontier_descriptions" (中文描述)
  "benchmark_descriptions" (独特描述)

Show "snapshot.json" as a gate control icon:
  - "字段 = false → 跳过 (不消耗 API)"
  - "字段 = true  → 重跑"

====================================================================
OUTPUT (bottom):
====================================================================
Final box: "VisualizationData" (Pydantic 模型)
Show structure tree:
  "topic → Overview / Timeline[n] / Paper[n] / Method[n] /
   Benchmark[n] / Frontier[n] / KnowledgeGraph{nodes, edges} /
   needs_research[n]"

With annotation: "8 种节点类型 × 11 种边关系 → Streamlit UI (7 个 Tab)"

====================================================================
RIGHT SIDE persistent element: "needs_research 缺口追踪"
====================================================================
A vertical bar on the right side of the entire diagram, spanning all
3 stages, showing:

  Stage 1: "生成缺口任务 (新增)"
  Stage 2: "LLM 补全部分缺口 (移除已满足)"
  Stage 3: "OpenAI 增强补全 (可选)"
  Output: "剩余缺口 → 供人工/后续自动化补全"

Each gap shown as a small card: target + reason + hint

Bottom annotation: "宁缺毋滥: 宁留空也不编造 = 可审计的质量保证"

==============================================================================
COLOR SCHEME for Figure 3:
==============================================================================
- Stage 1: light blue (#d4e6f1)
- Stage 2: light green (#d5f5e3)
- Stage 3: light purple (#e8daef)
- needs_research tracker: light orange (#fdebd0), orange border (#e67e22)
- Constraint text: dark red (#c0392b) for emphasis
- Background: white
- Arrows: dark gray (#555)
