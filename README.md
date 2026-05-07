# ReviewForge

智能论文检索与结构化增强平台。

输入主题 → 自动规划多源检索 → 策展分析 → 生成综述细纲，覆盖论文 + 代码 + Benchmark + 数据集 + 讨论。

## 功能

- **多源论文检索**: arXiv / Semantic Scholar / DBLP / Papers With Code
- **开源资源发现**: GitHub / HuggingFace / Papers With Code SOTA
- **社区讨论检索**: Hacker News (Algolia API)
- **策展分析**: 自动分类、洞察提炼、质量评估
- **LLM 生成细纲**: 基于检索结果生成逻辑严密的综述章节框架
- **全量检索**: 一次查询，覆盖论文+代码+benchmark+数据集+讨论

## 快速开始

```bash
# 安装依赖
uv sync

# 运行全量检索（CLI）
uv run python -m src.manager

# 运行测试
uv run pytest tests/
```

## TUI 交互界面

ReviewForge 提供交互式 TUI，集成了检索规划、并发执行、策展分析、细纲生成完整流程。

### 运行方式

在项目根目录（`E:\Documents\ReviewForge`）下：

```powershell
# 进入项目目录
cd E:\Documents\ReviewForge

# 激活虚拟环境并运行 TUI（Windows PowerShell）
& .venv\Scripts\python -m src.tui
```

### 命令行参数

```powershell
# 直接指定主题，单次运行后退出（非交互模式）
.venv\Scripts\python -m src.tui --topic "diffusion model for time series"

# 开启 debug 日志
.venv\Scripts\python -m src.tui --debug

# 交互模式（不传参数）
.venv\Scripts\python -m src.tui
```

### 使用 API_KEY

TUI 的 LLM 规划 / 细纲生成功能需要配置 `LLM_API_KEY`。支持 SiliconFlow（默认）、OpenAI 或任意 OpenAI 兼容端点。

**SiliconFlow（推荐，免翻墙）**

```powershell
$env:LLM_API_KEY = "your-siliconflow-api-key"
.venv\Scripts\python -m src.tui
```

**或创建 `.env` 文件（项目根目录）**

```env
LLM_API_KEY=your-siliconflow-api-key
```

> TUI 工作流：规划检索 → 执行检索 → 策展分析 → 生成细纲 → 补搜（如需要）→ 展示结果

## 检索器配置

| 检索器 | 配置项 | 是否需要 API Key | 是否免费 |
|--------|--------|----------------|---------|
| arXiv | 无 | 否 | ✅ 完全免费 |
| Semantic Scholar | `S2_API_KEY` 环境变量 | 是（必须注册）| ✅ 免费 tier |
| DBLP | 无 | 否 | ✅ 完全免费 |
| GitHub | `GITHUB_TOKEN` 环境变量 | 是 | ✅ 免费（60→5000 req/hr）|
| Papers With Code | 无 | 否 | ✅ 完全免费 |
| HuggingFace | 无 | 否 | ✅ 完全免费 |
| Hacker News | 无 | 否 | ✅ 完全免费 |

> 策展分析前需至少配置 `S2_API_KEY`（Semantic Scholar）和 `LLM_API_KEY`（SiliconFlow）。

## LLM 配置

通过环境变量或 `.env` 文件配置：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LLM_API_KEY` | **必需** | LLM API Key |
| `LLM_BASE_URL` | `https://api.siliconflow.cn/v1` | API 端点 |
| `LLM_MODEL` | `Qwen/Qwen3-8B` | 模型名称 |
| `LLM_TEMPERATURE` | `0.1` | 生成温度 |
| `LLM_MAX_TOKENS` | `2000` | 最大 token 数 |
| `S2_API_KEY` | 必需（策展分析）| Semantic Scholar API Key |
| `GITHUB_TOKEN` | 可选 | GitHub API Token |

## 项目结构

```
ReviewForge/
├── src/
│   ├── __init__.py              # 包入口
│   ├── models.py                # 数据模型（PaperCard, ResourceCard, CurationReport）
│   ├── config.py                # 配置管理（Settings）
│   ├── manager.py               # All-in-One 检索管理器
│   ├── executor.py             # 检索执行器（RetrieverManager）
│   ├── planner/                # LLM 规划模块
│   │   ├── planner.py          # 检索规划器
│   │   ├── llm.py              # LLM 客户端（OpenAI 兼容）
│   │   ├── searcher.py         # 查询生成器
│   │   ├── fetcher.py          # 结果抓取器
│   │   ├── prompts.py          # 提示词模板
│   │   └── schemas.py          # Pydantic 输入/输出 schema
│   ├── retrievers/             # 各数据源检索器
│   │   ├── base.py             # 基类
│   │   ├── arxiv.py            # arXiv
│   │   ├── semantic_scholar.py # Semantic Scholar
│   │   ├── dblp.py             # DBLP
│   │   ├── github.py           # GitHub
│   │   ├── papers_with_code.py # Papers With Code
│   │   ├── huggingface.py      # HuggingFace
│   │   ├── hackernews.py       # Hacker News
│   │   └── serper.py          # Google Serper
│   ├── curation/
│   │   └── curator.py         # 策展管理
│   └── tui/                    # 交互式 TUI
│       ├── __main__.py
│       └── tui.py
├── tests/
│   ├── conftest.py            # 共享 fixtures
│   ├── test_retrievers.py     # 综合测试
│   └── unit/
│       └── test_imports.py
├── .env                      # 环境变量（手动创建）
├── pyproject.toml
└── uv.lock
```

## 开发

```bash
# 安装开发依赖
uv sync --group dev

# 代码检查
ruff check src/ --fix
black src/
mypy src/

# 运行测试
pytest tests/
```
