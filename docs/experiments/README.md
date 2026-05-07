# Embedding BFS 过滤实验

## 背景

参考 PaSa (https://pasa-agent.ai/) 的 BFS 搜索策略：
- Search 阶段：LLM 生成查询 → 搜索 arxiv
- Expand 阶段：对论文做引用展开 → 用 **7B LLM selector** 逐篇打分（1-token True/False 概率）
- Score > 0.5 → 进入下一层 BFS

本实验验证能否用 **API Embedding 相似度** 代替本地 7B selector：

```
cos_sim(embed(Q), embed(P.title + P.abstract)) > threshold
  → P 通过过滤，进入 BFS 展开
```

## 实验脚本

```bash
export SILICONFLOW_API_KEY="sk-xxx"

# 完整实验（需要人工标注）
python experiments/embedding_bfs_experiment.py --topic "ASR for low-resource languages" --bfs-layers 1

# 快速模式（关键词自动标注，仅用于粗略验证）
python experiments/embedding_bfs_experiment.py --topic "LoRA fine-tuning" --quick
```

## 实验流程

```
Phase 0: LLM Query Generation
  └─ 调用硅基流动 Chat API → 生成多源查询 + rerank_query

Phase 1: arXiv Search
  └─ 用 LLM 生成的 primary query 搜索 arxiv → 获得 15 篇论文

Phase 2: Relevance Annotation（人工）
  └─ 逐篇标注 y=相关 / n=不相关 / s=跳过

Phase 3: Embedding Similarity Filtering
  ├─ 测试 3 种 query: 原始用户查询 / LLM rerank query / arxiv primary query
  ├─ 计算 query embedding + 每篇论文 (title+abstract) embedding
  ├─ 余弦相似度排序
  └─ 在 20 个均匀间隔阈值下计算 precision/recall/F1

Phase 4 (可选): BFS Citation Expansion
  ├─ 取 top-k 论文
  ├─ 从 ar5iv 解析全文 → 提取 sections + 引用
  ├─ 对每篇引用论文搜索元信息
  └─ 计算 embedding 相似度排序
```

## 评估指标

| 指标 | 含义 |
|------|------|
| Precision@threshold | 过滤后结果中相关论文占比 |
| Recall@threshold | 所有相关论文中被保留的比例 |
| F1@threshold | 综合最优阈值 |
| BFS Gain | BFS 展开后新增相关论文数 |

## 测试 Topics（建议）

| Topic | 预期特性 | 测试目的 |
|-------|---------|---------|
| "LoRA fine-tuning" | 结果相关/不相关混合 | 基础过滤效果 |
| "ASR" | LLM 查询完全错位 | embedding 能否纠正 |
| "3D Gaussian Splatting" | 较新领域 | 冷门主题效果 |

## 预期实验结果

1. **查询类型对比**：LLM rerank_query > 原始用户查询 > arxiv primary query（embedding 相似度）
2. **最佳阈值**：0.3-0.5 之间（取决于 embedding 模型）
3. **BFS 扩展**：embedding 过滤后 BFS 展开的相关论文比例应 > 随机展开
4. **与 PaSa 对比**：embedding 方案 precision 可能略低于 7B selector，但 recall 相当，成本大幅降低
