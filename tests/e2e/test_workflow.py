"""端到端测试——模拟完整检索流程"""

from src.manager import RetrieverManager
from src.curation.curator import Curator


def test_end_to_end():
    """模拟一次完整检索流程（含策展）"""
    query = "distributed computing survey"

    with RetrieverManager() as mgr:
        report = mgr.search_all(query, max_results=10)

    curator = Curator()
    report = curator.curate(report)

    assert report.topic == query
    assert report.total_papers >= 0
    assert report.classic_count >= 0
    assert report.frontier_count >= 0

    # 策展后的报告应该有分类
    assert isinstance(report.categories, dict)

    # 评估
    grade = curator.evaluate(report)
    assert grade in ("优秀", "良好", "需补充")

    print(f"端到端测试通过:")
    print(f"  论文: {report.total_papers} | 经典: {report.classic_count} | 前沿: {report.frontier_count}")
    print(f"  资源: GitHub={report.github_count} | Benchmark={report.benchmark_count} | 数据集={report.dataset_count}")
    print(f"  评级: {grade}")
    print(f"  洞察: {report.insights[:2]}")
