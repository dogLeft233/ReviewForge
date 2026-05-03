"""Planner 提示词模板——独立文件便于维护

所有提示词集中管理，支持按版本号迭代。
"""

# ════════════════════════════════════════════════════════════════════
# PLAN_RETRIEVAL v1 — 检索规划
# ════════════════════════════════════════════════════════════════════

PLAN_RETRIEVAL_V1 = """你是一个学术综述检索规划专家。

## 任务
给定一个综述主题，制定检索策略。你需要：
1. 分析主题的核心概念和子方向
2. 生成多个有针对性的搜索查询（中英文）
3. 为每个查询指定目标检索源
4. 区分"经典工作"和"前沿工作"的搜索策略
5. 指定需要的资源类型

## 可用检索源
- arxiv：学术论文预印本
- semantic_scholar：学术论文（含引用数据）
- dblp：计算机科学文献（含分类标签）
- github：开源代码项目
- huggingface：模型/数据集
- hackernews：社区讨论/技术热点

## 输出要求
输出严格的 JSON 对象（不要添加任何解释文字），格式如下：
{{
    "topic_analysis": {{
        "core_concepts": ["核心概念列表"],
        "sub_directions": ["子方向列表"],
        "focus_level": "broad|focused|specific"
    }},
    "queries": [
        {{
            "query": "搜索关键词",
            "language": "en|zh",
            "target_sources": ["arxiv", "semantic_scholar"],
            "rationale": "为什么这样搜索",
            "priority": 1
        }}
    ],
    "classic_search_strategy": "如何搜索经典工作",
    "frontier_search_strategy": "如何搜索前沿工作",
    "resource_types_needed": ["github_repo", "dataset", "benchmark", "model", "discussion"],
    "estimated_coverage": ["预计可以覆盖的论文方向"]
}}

## 主题
{topic}
"""

# ════════════════════════════════════════════════════════════════════
# EVALUATE_COVERAGE v1 — 覆盖度评估
# ════════════════════════════════════════════════════════════════════

EVALUATE_COVERAGE_V1 = """你是一个学术综述覆盖度评估专家。

## 任务
评估已检索到的论文和资源是否足以撰写一份高质量的综述报告，识别缺失的关键方向和补充建议。

## 输入
- 主题：{topic}
- 检索到的论文数：{total_papers}
- 经典论文数（高引用）：{classic_count}
- 前沿论文数（2023+）：{frontier_count}
- GitHub 项目数：{github_count}
- Benchmark 数：{benchmark_count}
- 数据集数：{dataset_count}
- 已涉及的方法分类：{categories}
- 当前洞察：{insights}

## 输出要求
输出严格的 JSON 对象（不要添加任何解释文字），格式如下：
{{
    "overall_assessment": "adequate|insufficient|critical_gaps",
    "strengths": ["已有覆盖较好的方向"],
    "gaps": ["缺失的关键方向"],
    "missing_classics": ["可能的经典工作缺失"],
    "missing_frontiers": ["可能的前沿方向缺失"],
    "supplementary_queries": [
        {{
            "query": "补搜关键词",
            "target_sources": ["arxiv", "github"],
            "rationale": "为什么需要补搜"
        }}
    ],
    "resource_gaps": ["缺失的资源类型"],
    "confidence": 0.8
}}
"""

# ════════════════════════════════════════════════════════════════════
# SYNTHESIZE_INSIGHTS v1 — 洞察提取
# ════════════════════════════════════════════════════════════════════

SYNTHESIZE_INSIGHTS_V1 = """你是一个学术综述洞察提取专家。

## 任务
分析已检索到的论文集合，提取对综述写作有价值的洞察：
1. 方法演进脉络——哪些工作构建在前人基础上
2. 研究热点——哪些方向受到的关注最多
3. 未解决的开放问题
4. 论文之间的关联

## 输入
主题：{topic}

论文列表（标题 | 年份 | 引用 | 方法分类 | 会议）：
{papers_summary}

资源列表（名称 | 类型 | 描述）：
{resources_summary}

## 输出要求
输出严格的 JSON 对象（不要添加任何解释文字），格式如下：
{{
    "evolution_paths": [
        {{
            "path_name": "演进路线名称",
            "papers": ["涉及的论文标题"],
            "description": "演进描述"
        }}
    ],
    "hot_topics": ["热点方向列表"],
    "open_problems": ["开放问题列表"],
    "cross_references": [
        {{
            "paper_a": "论文A",
            "paper_b": "论文B",
            "relationship": "builds_on|compares_to|complements|contradicts"
        }}
    ],
    "writing_recommendations": [
        {{
            "section": "论文章节名称",
            "suggested_papers": ["建议引用的论文"],
            "key_message": "该章节的核心论点"
        }}
    ]
}}
"""

# ════════════════════════════════════════════════════════════════════
# 提示词注册表——按版本号索引
# ════════════════════════════════════════════════════════════════════

PROMPT_REGISTRY = {
    "plan_retrieval": {
        "v1": PLAN_RETRIEVAL_V1,
    },
    "evaluate_coverage": {
        "v1": EVALUATE_COVERAGE_V1,
    },
    "synthesize_insights": {
        "v1": SYNTHESIZE_INSIGHTS_V1,
    },
}


def get_prompt(name: str, version: str = "v1", **kwargs: str) -> str:
    """获取指定名称和版本的提示词，并格式化参数"""
    prompt_group = PROMPT_REGISTRY.get(name)
    if prompt_group is None:
        raise KeyError(f"Unknown prompt: {name!r}, available: {list(PROMPT_REGISTRY)}")
    template = prompt_group.get(version)
    if template is None:
        raise KeyError(
            f"Unknown version {version!r} for prompt {name!r}, "
            f"available: {list(prompt_group)}"
        )
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(
            f"Missing format parameter {e} for prompt {name!r} v{version}, "
            f"got keys: {list(kwargs)}"
        ) from e
