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
│   ├── run_explorer.py  ← 只跑 Explorer 阶段（调试用）
│   └── test_explorer.py ← 单元测试
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
│   ├── retrievers/        ← 各数据源检索器
│   │   ├── arxiv.py      ← arXiv 检索
│   │   ├── ar5iv.py      ← ar5iv 检索
│   │   ├── github.py    ← GitHub 检索
│   │   ├── huggingface.py ← HuggingFace 检索
│   │   └── semantic_scholar.py ← Semantic Scholar 检索
│   └── writer/
│       ├── agent.py      ← Writer Agent（各章节写作逻辑）
│       └── prompts/      ← 各章节 system prompt 模板
└── tmp/                  ← 中间结果目录（自动创建）
```

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
├── step1_explorer_done.json   ← Explorer 输出
├── step2_searcher_done.json    ← Searcher 输出（搜索词 + 检索结果）
└── step3_writer_done.json     ← Writer 输出（综述全文）
```

使用 `--resume` 时会自动找到最新中间文件并从对应步骤恢复。
