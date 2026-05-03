# ReviewForge

智能论文检索与结构化增强平台。

## 功能

- **多源论文检索**: arXiv / Semantic Scholar / DBLP / Papers With Code
- **开源资源发现**: GitHub / HuggingFace / Papers With Code SOTA
- **社区讨论检索**: Hacker News (Algolia API)
- **策展分析**: 自动分类、洞察提炼、质量评估
- **全量检索**: 一次查询，覆盖论文+代码+benchmark+数据集+讨论

## 快速开始

```bash
# 安装依赖
uv sync

# 运行测试
uv run pytest tests/

# 运行全量检索
uv run python -m src.manager
```

## 检索器配置

| 检索器 | 配置项 | 是否需要API Key | 是否免费 |
|--------|--------|----------------|---------|
| arXiv | 无 | 否 | ✅ 完全免费 |
| Semantic Scholar | `S2_API_KEY` 环境变量 | 是（必须注册）| ✅ 免费tier |
| DBLP | 无 | 否 | ✅ 完全免费 |
| GitHub | `GITHUB_TOKEN` 环境变量 | 是 | ✅ 免费（60→5000 req/hr）|
| Papers With Code | 无 | 否 | ✅ 完全免费 |
| HuggingFace | 无 | 否 | ✅ 完全免费 |
| Hacker News | 无 | 否 | ✅ 完全免费 |

## 项目结构

```
ReviewForge/
├── src/
│   ├── __init__.py           # 包入口
│   ├── models.py             # 数据模型（PaperCard, ResourceCard, CurationReport）
│   ├── exceptions.py         # 异常层次
│   ├── config.py             # 配置管理
│   ├── manager.py            # All-in-One 检索管理器
│   ├── retrievers/           # 检索器
│   │   ├── base.py           # 基类
│   │   ├── arxiv.py          # arXiv
│   │   ├── semantic_scholar.py  # Semantic Scholar
│   │   ├── dblp.py           # DBLP
│   │   ├── github.py         # GitHub
│   │   ├── papers_with_code.py # Papers With Code
│   │   ├── huggingface.py    # HuggingFace
│   │   └── hackernews.py     # Hacker News
│   └── curation/
│       └── curator.py        # 策展管理
├── tests/
│   ├── conftest.py           # 共享 fixtures
│   ├── test_retrievers.py    # 综合测试
│   └── unit/
│       └── test_imports.py
└── pyproject.toml
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
