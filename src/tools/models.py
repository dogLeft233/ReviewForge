"""Web 工具数据模型"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class WebSearchResult:
    """单条搜索引擎结果"""

    title: str
    url: str
    description: str = ""
    site_name: str = ""
    published_date: str = ""
    raw: dict = field(default_factory=dict)


@dataclass(slots=True)
class WebFetchResult:
    """页面抓取结果"""

    url: str
    title: str = ""
    content: str = ""  # 提取的正文文本
    raw_content: str = ""  # 原始 HTML（调试用）
    content_type: str = ""
    status_code: int = 0
    fetched_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @property
    def is_html(self) -> bool:
        return "text/html" in self.content_type

    @property
    def is_pdf(self) -> bool:
        return "application/pdf" in self.content_type
