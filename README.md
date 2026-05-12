# ReviewForge

给定一个研究领域，自动完成**综述论文**的全流程：从文献检索到结构化写作。

---

## 工作流程（三步接力）

```
用户输入领域（如 "ASR 自动语音识别"）
        │
        ▼
  ┌─────────────┐
  │  Explorer   │  理解领域，找经典论文，摸清研究脉络
  └─────────────┘
        │  ExplorerReport（含阶段概述 / 经典论文 / 时间线 / SOTA / Benchmarks）
        ▼
  ┌─────────────┐
  │  Searcher   │  基于 Explorer 结果生成高质量搜索词并搜索
  └─────────────┘
        │  SearchResult（含检索结果列表）
        ▼
  ┌─────────────┐
  │  Writer     │  基于所有以上信息撰写综述（标题 / 引言 / 正文 / 结论 / 摘要 / 关键词 / 参考文献）
  └─────────────┘
        ▼
      完成 ✅
```

**每一步的中间结果自动保存**，支持 `--resume` 从任意已完成步骤恢复，不用重头跑。

---

## 目录结构

```
ReviewForge/
├── scripts/
│   ├── run_pipeline.py   ← 入口脚本，一键跑完整流程（最常用）
│   ├── run_explorer.py   ← 只跑 Explorer 阶段（调试用）
│   ├── export_viz.py     ← 把 step3 中间结果转成 UI 可视化数据
│   └── test_explorer.py  ← 单元测试
├── src/
│   ├── core.py           ← 流水线总控（组装三阶段）
│   ├── config.py         ← 配置读取
│   ├── llm.py            ← LLM 调用封装
│   ├── embedding.py      ← Embedding 模型
│   ├── reranker.py       ← 重排序模型
│   ├── explorer/
│   │   ├── agent.py      ← Explorer Agent（三阶段探索逻辑）
│   │   └── explorer_report.py  ← Explorer 输出结构
│   ├── seacher/          ← Searcher（注意原文拼写）
│   │   ├── agent.py      ← Searcher Agent
│   │   └── tools/        ← 搜索工具实现
│   ├── retrievers/       ← 各数据源检索器
│   │   ├── arxiv.py      ← arXiv 检索
│   │   ├── ar5iv.py      ← ar5iv 检索
│   │   ├── github.py     ← GitHub 检索
│   │   ├── huggingface.py ← HuggingFace 检索
│   │   └── semantic_scholar.py ← Semantic Scholar 检索
│   ├── writer/
│   │   ├── agent.py      ← Writer Agent（各章节写作逻辑）
│   │   └── prompts/      ← 各章节 system prompt 模板
│   ├── adapter/          ← Agent 输出 → 前端 VisualizationData 适配层
│   │   ├── schema.py     ← Pydantic 模型（Overview / Timeline / Method / Paper / Benchmark / Frontier / Graph）
│   │   ├── extractor.py  ← markdown / 表格 / leaderboard 解析
│   │   ├── script_adapter.py ← 规则化转换；缺口写入 needs_research
│   │   ├── llm_refiner.py ← LLM 校验/补全；不允许编造事实型字段
│   │   ├── prompts.py    ← refiner 用 prompt 模板
│   │   └── pipeline.py   ← convert / convert_with_llm 入口
│   └── visualizer/       ← UI 兼容层（schema/loader 透传 src.adapter）
├── ui/                   ← Streamlit 前端
│   ├── app.py            ← 主入口（7 个 tab）
│   └── views/            ← 各 tab 渲染（overview / timeline / method_map /
│                            frontier / benchmark / knowledge_graph / ask_agent）
└── tmp/                  ← 中间结果目录（自动创建）
```

---

## 环境配置

### 1. Python 环境

需要 **Python 3.10+**（用了 `match` / 新式类型语法）。

```bash
# 推荐用 venv / conda
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. 安装依赖

```bash
# 后端（agent 流水线）
pip install httpx pydantic

# 前端可视化（Streamlit + 图表）
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
streamlit>=1.32
pandas>=2.0
plotly>=5.18
pyvis>=0.3.2
graphviz>=0.20
pydantic>=2.0
```

> Knowledge Graph tab 用了 `graphviz` Python 包，渲染本身由 Streamlit 内置 `st.graphviz_chart` 完成，无需另装系统级 Graphviz。

### 3. config.json

复制项目根目录下的示例（或新建）：

```jsonc
{
  "bocha_api_key": "",                  // 网页搜索（可空）
  "semantic_scholar_api_key": "",       // S2 检索（可空）
  "github_token": "",                   // GitHub 检索（可空）
  "llm_api_key": "sk-xxx",              // ★ 必填：LLM 推理
  "llm_base_url": "https://api.siliconflow.cn/v1",
  "llm_model": "Qwen/Qwen3-8B",
  "llm_temperature": 0.1,
  "llm_max_tokens": 2000,
  "llm_timeout_seconds": 120.0,
  "llm_max_retries": 2,
  "embedding_model": "BAAI/bge-m3",
  "embedding_base_url": "https://api.siliconflow.cn/v1",
  "reranker_model": "BAAI/bge-reranker-v2-m3",
  "arxiv_email": "you@example.com"      // arXiv API 礼貌要求
}
```

敏感字段也可用环境变量覆盖：`LLM_API_KEY` / `BOCHA_API_KEY` / `S2_API_KEY` / `GITHUB_TOKEN` / `HF_TOKEN` / `RERANKER_API_KEY` / `ARXIV_EMAIL`。

---

## 快速开始

### 完整流程（交互输入）

```bash
cd /mnt/e/Documents/ReviewForge
python scripts/run_pipeline.py
```

### 指定领域（非交互）

```bash
python scripts/run_pipeline.py -t "LoRA fine-tuning"
```

### 从断点恢复（跳过已完成步骤）

```bash
python scripts/run_pipeline.py -t "ASR自动语音识别" --resume
```

### 仅运行 Explorer 阶段

```bash
python scripts/run_explorer.py -t "ASR自动语音识别"
```

### 指定日志级别

```bash
python scripts/run_pipeline.py -t "ASR自动语音识别" --log-level DEBUG --resume
```

### 查看所有已保存的 session

```bash
python scripts/run_pipeline.py --list
```

---

## 各脚本说明

| 脚本 | 作用 | 使用场景 |
|------|------|----------|
| `run_pipeline.py` | **主入口**，完整三阶段流水线 | 日常写综述，用这个就够了 |
| `run_explorer.py` | 只运行 Explorer 阶段 | 调试 Explorer 或只想了解某领域概况 |
| `test_explorer.py` | Explorer 单元测试 | 开发调试 |

---

## 数据流向详解

### ExplorerReport（Explorer → Searcher/Writer）

Explorer 会输出一个报告，包含：

- `stage1_overview` — 领域总览
- `stage1_search_results` — 初步搜索结果
- `stage1_concepts` — 核心关键词/概念
- `stage2_classics` — 经典论文列表（含标题、年份）
- `stage2_timeline` — 研究时间线
- `stage3_state_of_art` — 当前最佳方法（SOTA）
- `stage3_trends` — 研究趋势
- `stage3_benchmarks` — 基准数据集列表

### SearchReport（Searcher 输出）

- 多个高质量搜索词（topic + subtopic 组合），供多源检索使用

### MultiSourceSearchResult（检索阶段输出）

- 并发请求多个数据源，返回去重后的论文列表（含标题、作者、摘要、链接等）

### WriterReport（最终输出）

- `title` — 综述标题
- `plan` — 写作大纲/分类体系
- `introduction` — 引言
- `body` — 正文（各章节内容）
- `conclusion` — 结论
- `abstract` — 摘要
- `keywords` — 关键词列表
- `references` — 参考文献列表

---

## 配置

项目根目录的 `config.json` 包含：

- API Keys（Bocha API / LLM API）
- 模型参数（temperature、max_tokens 等）
- 重排序模型路径
- 检索器并发限制

`src/config.py` 负责读取这些配置。

---

## 中间文件

中间结果保存在 `tmp/{领域名}/` 目录下：

```
tmp/ASR自动语音识别/
├── step1_explorer_done.json    ← Explorer 输出
├── step2_searcher_done.json    ← Searcher 输出（搜索词 + 检索结果）
├── step3_writer_done.json      ← Writer 输出（综述全文）
├── visualization_data.json     ← 前端 UI 数据（由 export_viz.py 生成）
└── snapshot.json               ← OpenAI 增强缓存开关（由 export_viz.py 维护）
```

使用 `--resume` 时会自动找到最新中间文件并从对应步骤恢复。

---

## 导出可视化数据

`scripts/export_viz.py` 把 `step3_writer_done.json` 适配成前端 UI 直接消费的 `visualization_data.json`。

基础管线两阶段：

1. **脚本规则化**（`src/adapter/script_adapter.py`）—— 解析 `stage2_timeline` 论文表、`stage3_benchmarks` 列表等，得到 timeline / methods / papers / benchmarks / frontiers / graph 六大块。所有"拿不准"的字段一律留空，并把缺口登记到 `needs_research`。
2. **LLM 校验/补全**（`src/adapter/llm_refiner.py`）—— 逐段把脚本结果送给 LLM，让它去掉碎片、清洗作者、判断 frontier 重要性、补 method 优缺点。**LLM 禁止编造**作者/年份/URL/分数等事实型字段；任一段失败就静默回退脚本结果。

开启 `--openai-enhance` 时，还会额外调用 OpenAI Responses API 做分模块增强，例如论文链接、method 优缺点、frontier 中文描述、benchmark 独特描述等。该增强会被 `snapshot.json` 控制，避免每次导出都重复消耗 token。

### 用法

```bash
# 默认两阶段（脚本 + LLM）：处理 tmp/ 下所有 step3 完成的领域
python scripts/export_viz.py

# 只跑脚本，跳过 LLM（CI / 没网 / 想快）
python scripts/export_viz.py --no-llm

# 只转某个领域
python scripts/export_viz.py tmp/ASR自动语音识别

# 只转某个领域，并跳过本地 LLM
python scripts/export_viz.py tmp/ASR自动语音识别 --no-llm

# 开启 OpenAI Responses API 增强（受 snapshot.json 控制）
python scripts/export_viz.py tmp/ASR自动语音识别 --no-llm --openai-enhance

# 显式指定 IO
python scripts/export_viz.py \
    --input  tmp/ASR自动语音识别/step3_writer_done.json \
    --output tmp/ASR自动语音识别/visualization_data.json
```

输出会打印每个文件的统计，例如：

```
[OK] step3_writer_done.json -> tmp/ASR自动语音识别/visualization_data.json
     topic=ASR自动语音识别 | timeline=6 methods=10 papers=6 benchmarks=8 frontiers=8
     | nodes=37 edges=36 | needs_research=46
```

`needs_research` 数量 > 0 表示脚本/LLM 都没能可靠填上某些字段（如 paper.url、benchmark.score），列出来等后续 research 阶段补齐。每条记录形如：

```json
{
  "target": "paper:p_attention_is_all_you_need.url",
  "reason": "原始数据未提供链接",
  "hint": "为论文「Attention Is All You Need」找到 arXiv / DOI / 官方页面 URL"
}
```

### snapshot.json 用法

`snapshot.json` 位于每个项目目录下，例如 `tmp/ASR自动语音识别/snapshot.json`。它只在使用 `--openai-enhance` 时生效，用来控制哪些 AI 增强字段需要重跑。

核心规则：

- 没有 `snapshot.json`：认为所有增强字段都需要跑，`export_viz.py` 会调用 Responses API。
- 有 `snapshot.json`：只重跑值为 `true` 的字段；值为 `false` 的字段会跳过，不消耗 token。
- 某个模块成功返回合理 patch 后，脚本会把它负责的字段写成 `false`。
- 某个模块调用失败或返回不合理时，它负责的字段保持/写成 `true`，下次仍会尝试。
- 如果 `snapshot.json` 和 `visualization_data.json` 都存在，脚本会以现有 `visualization_data.json` 为基底，只更新需要重跑的字段，避免已增强内容丢失。

示例：

```json
{
  "version": 1,
  "description": "True means the field should be regenerated by OpenAI Responses API on the next export.",
  "fields": {
    "papers.url": false,
    "timeline.related_papers": false,
    "methods.pros": false,
    "methods.cons": false,
    "methods.papers": false,
    "frontiers.items": true,
    "frontiers.resources": true,
    "frontiers.description": false,
    "benchmarks.description": false
  }
}
```

想单独重跑 Frontier 文献检索，就把相关字段改成 `true`：

```json
"frontiers.items": true,
"frontiers.resources": true
```

想单独重跑 Benchmark 描述，就改：

```json
"benchmarks.description": true
```

然后再次运行：

```powershell
python scripts/export_viz.py tmp/ASR自动语音识别 --no-llm --openai-enhance
```

---

## 启动可视化 UI

可视化前端是 Streamlit 应用，入口在 `ui/app.py`。

```bash
# 从项目根目录启动
streamlit run ui/app.py
```

默认浏览器会弹开 `http://localhost:8501`。

### 界面与操作

侧栏：

- **上传 `visualization_data.json`** —— 不上传则用内置的 ASR demo（自动读 `tmp/ASR自动语音识别/visualization_data.json`）
- **方法类别筛选** —— 影响 Timeline / Method Map 的展示
- **年份范围** —— 联动 Timeline / Benchmark

主区 7 个 tab：

| Tab | 内容 |
|---|---|
| **Overview** | 领域定义、核心问题、关键概念、代表论文卡片列表 |
| **Timeline** | Plotly 散点时间线（按年份 × 类别） |
| **Method Map** | Graphviz 树（topic → category → method） + 方法详情表 |
| **Frontier** | 前沿趋势卡片（含 importance 标注） |
| **Benchmark** | 全表 + 分数随年份变化散点 |
| **Knowledge Graph** | PyVis 交互式图（topic / method / paper / dataset / benchmark / trend 多类节点） |
| **Ask Agent** | 基于本地 JSON 的关键词路由问答 |

### 典型流程

```bash
# 1. 跑完三阶段管线
python scripts/run_pipeline.py -t "ASR自动语音识别"

# 2. 把 step3 结果转成 UI 数据
python scripts/export_viz.py tmp/ASR自动语音识别

# 3. 启动前端
streamlit run ui/app.py
# 侧栏上传刚生成的 tmp/ASR自动语音识别/visualization_data.json
```
## OpenAI Responses API 增强（可选）

项目根目录提供 `adapter_enhance.yaml`。默认 `enabled: false`，不会产生 OpenAI API 调用。

如需启用专业性评估、web search 补链和结构化增强：

```yaml
enabled: true
openai:
  api_key: "你的 OpenAI API key"
  model: "gpt-4o"
web_search:
  enabled: true
  tool_type: "web_search_preview"
```

导出时显式开启：

```powershell
python scripts/export_viz.py tmp/ASR自动语音识别 --no-llm --openai-enhance
```

增强层会按模块调用 OpenAI Responses API，而不是一次性发送一个大请求。当前模块包括：

- `paper_timeline_links`：补论文 URL、Timeline 相关论文链接，并把文献链接收集到 Links 页。
- `method_details`：补 Method Map 中简明的 pros / cons。
- `method_links`：修复 method 到代表论文/资源的关联。
- `frontier_sources`：检索更多 2024-2026 前沿论文和开源项目。
- `frontier_descriptions`：把 Frontier 卡片 description 改成中文短描述。
- `benchmark_descriptions`：为每个 benchmark 生成独特中文说明。

增强层默认只增强表述、分类、关联和来源链接，不会凭空改作者、年份、benchmark 分数等事实字段。每个模块是否重跑由 `tmp/{领域名}/snapshot.json` 控制；字段为 `false` 时会跳过对应模块，不消耗 token。
