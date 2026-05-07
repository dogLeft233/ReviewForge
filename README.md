# MetaSearcher

多源联网论文检索与智能排序模块。

输入主题 → 并行搜索多个来源 → 分类融合 → 语义重排序 → 输出结构化结果。

---

## 工作流程

```
用户输入（如 "transformer architecture"）
           │
           ▼
    ┌──────────────────┐
    │  多源并行检索      │
    │  papers:  arXiv, Serper
    │  resources: GitHub, HuggingFace
    │  news: Bocha, Hacker News
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │  分类 RRF 融合    │
    │  同类结果合并排序   │
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │  URL 去重         │
    │  归一化 + 标题去重  │
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │       │
    │  语义重排序        │
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │  返回 SearchContext │
    │  papers/          │
    │  resources/       │
    │  news/            │
    └──────────────────┘
```

---

## 数据源

| 类别 | 来源 | 说明 |
|------|------|------|
| **papers** | arXiv | 免费，无需 API Key |
| | Serper | Google 学术搜索，需 API Key |
| **resources** | GitHub | 代码仓库，按 stars 排序 |
| | HuggingFace | 模型 / 数据集 |
| **news** | Bocha | 中文友好，支持 AI 摘要 |
| | Hacker News | 社区讨论热点 |

---

## 快速使用

```python
from src.searcher import MetaSearcher, SearchContext

searcher = MetaSearcher()
result: SearchContext = await searcher.search("transformer architecture survey")

# 分类结果
for paper in result.papers:
    print(paper.title, paper.url, paper.rank_score)

for resource in result.resources:
    print(resource.title, resource.url, resource.rank_score)
```

---

## 配置

环境变量或 `.env` 文件：

| 变量 | 必需 | 说明 |
|------|------|------|
| `SILICONFLOW_API_KEY` | 是 | SiliconFlow API Key（用于 Rerank） |
| `SERPER_API_KEY` | 否 | Serper API Key |
| `BOCHA_API_KEY` | 否 | Bocha API Key |
| `GITHUB_TOKEN` | 否 | GitHub Token（提升速率限制） |

---

## 核心概念

- **RRF (Reciprocal Rank Fusion)**: 多源结果融合算法，无需训练，对多源排名融合效果稳定
- **SiliconFlow Rerank**: BAAI/bge-reranker-v2-m3 模型，语义重排序，支持中文
- **SearchContext**: 返回数据结构，包含 `papers`、`resources`、`news` 三个分类的结果列表
- **SearchResult**: 单条搜索结果，包含 `title`、`url`、`abstract`、`rank_score`、`source`
