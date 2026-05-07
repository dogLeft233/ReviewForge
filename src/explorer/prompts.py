"""领域探索提示词模板"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExplorationPrompt:
    """领域探索分析提示词"""

    system: str = """你是一个专业的领域知识分析助手。
给定一段领域探索结果（包含网络搜索摘要、Hacker News 讨论等），
你需要输出一份该领域的初步探索总结。

输出格式要求（JSON）：
- exploration_summary: 200-400 字的领域初步探索总结，包含：该领域是什么、核心研究问题、主要技术路线、当前活跃方向
- key_terms: 5-10 个核心领域术语（含中英文），每个附简短定义
- core_concepts: 3-5 个核心概念的简洁描述（每条 15-30 字）
- related_topics: 2-4 个相关主题或子领域
- active_work: 该领域当前最活跃的研究/工程方向（1-2 句话）

注意：
- 只输出 JSON，不要有任何额外的解释文字
- exploration_summary 应该是连贯段落，而不是列表
- 术语定义要精准，避免模糊描述"""

    user: str = """## 领域探索原始材料

### 网络搜索摘要（来自博查搜索）
{bocha_content}

### Hacker News 社区讨论
{hn_content}

### 已发现的术语
{terms_content}

## 请分析并输出 JSON

请从以上材料中提取领域知识，输出以下 JSON 结构（只输出 JSON，不要其他文字）：
{{
  "exploration_summary": "领域初步探索总结：这是一篇关于xxx的领域...",
  "key_terms": [
    {{"term": "术语名称", "definition": "简洁定义（10-20字）", "source": "来源"}}
  ],
  "core_concepts": ["概念描述1", "概念描述2", ...],
  "related_topics": ["相关主题1", "相关主题2", ...],
  "active_work": "当前活跃方向的简洁描述"
}}"""


@dataclass(frozen=True)
class TermExtractionPrompt:
    """术语提取提示词"""

    system: str = """你是一个专业的技术术语提取助手。
给定一段领域文本，你需要从中提取该领域的核心技术术语。

输出要求：
- 只提取真正有意义的术语（不是通用词如 "model"、"system"、"data"）
- 每个术语附上简明定义
- 返回 JSON 数组格式"""

    user: str = """从以下文本中提取核心技术术语：

{text}

要求：
- 只提取该领域特有的术语（如模型名称、技术方法、工具名称）
- 过滤掉通用词（如 "model"、"system"、"data"、"method"、"result"）
- 每个术语需要附上 10-20 字的中文定义
- 术语保留英文原名（如果已是常用英文术语）

输出 JSON 数组：
[
  {{"term": "术语名", "definition": "简洁定义"}},
  ...
]

只输出 JSON，不要其他文字。"""


EXPLORATION_PROMPT = ExplorationPrompt()
TERM_EXTRACTION_PROMPT = TermExtractionPrompt()
