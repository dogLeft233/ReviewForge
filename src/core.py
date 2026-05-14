"""ReviewForge Core — 探索→搜索→写作 编排层

流程：
    用户输入领域
        → ExplorerAgent（三阶段探索）
        → SearcherAgent（基于 Explorer 结果生成搜索词）
        → MultiSourceSearcher（执行多源文献检索）
        → WriterAgent（撰写综述）

中间结果自动保存到 tmp/{topic_slug}/，
每个阶段可直接从磁盘恢复，跳过已完成的步骤。

用法:
    # 完整流程
    from src.core import ReviewForge, CoreConfig
    from src.config import settings

    rf = ReviewForge.from_settings()
    result = rf.run("LoRA large language model fine-tuning")

    # 从第 2 步恢复（跳过 Explorer）
    result = ReviewForge.resume_from("tmp/.../step2_searcher_done.json")

    # 单独使用某一步
    explorer_report = ReviewForge.run_explorer("LoRA", llm)
    search_result = ReviewForge.run_searcher("LoRA", explorer_report, llm)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.explorer import ExplorerAgent
from src.explorer.explorer_report import ExplorerReport
from src.llm import LLM, Message
from src.seacher.bfs_search import BFSSearcher
from src.seacher import MultiSourceSearcher, SearcherAgent
from src.writer import WriterAgent, WriterConfig, WriterReport
from src.adapter.openai_responses_enhancer import enhance_with_openai_responses

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CoreConfig:
    """Core 编排配置"""

    search_max_turns: int = 2
    """MultiSourceSearcher 最多搜索轮数"""
    search_max_results: int = 50
    """每个数据源最大结果数"""
    writer_max_tokens: int = 4096
    """Writer 每章最大 token 数"""
    temperature: float = 0.2
    verbose: bool = False

    # ── BFSSearcher 参数 ───────────────────────────────────────
    bfs_expand_layers: int = 1
    """BFS 最大层数（0=只用搜索，1=搜索+一层引文，2=两层）"""
    bfs_expand_papers_count: int = 20
    """BFS 每层最多扩展多少篇论文"""
    bfs_search_queries_count: int = 10
    """BFSSearcher 每轮生成多少个搜索词"""
    bfs_search_papers_count: int = 3
    """BFSSearcher 每个搜索词取多少篇论文"""
    bfs_similarity_threshold: float = 0.1
    """相似度过滤阈值"""
    bfs_rerank_top_n: int = 100
    """重排序后保留前 n 条（None=全部）"""
    bfs_enabled: bool = True
    """启用 BFSSearcher 作为主检索（False 则退化为仅 MultiSourceSearcher）"""


# ─────────────────────────────────────────────────────────────────────────────
# 结果结构
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """完整流程的各阶段结果"""

    topic: str = ""
    step: str = ""
    """当前完成的步骤: explorer_done | searcher_done | writer_done | complete"""

    # ── 各阶段原始结果 ─────────────────────────────────────
    searcher_papers: list = field(default_factory=list)
    """BFSSearcher 返回的 PaperNode 列表"""

    explorer_report: ExplorerReport | None = None
    searcher_messages: list[dict[str, str]] = field(default_factory=list)
    searcher_result: str = ""
    writer_report: WriterReport | None = None

    # ── 元数据 ─────────────────────────────────────────────
    tmp_dir: str = ""
    explorer_elapsed_seconds: float = 0.0
    searcher_elapsed_seconds: float = 0.0
    writer_elapsed_seconds: float = 0.0
    created_at: str = ""
    error: str = ""

    def save(self, path: Path) -> None:
        """持久化到 JSON 文件"""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _serialize_pipeline_result(self)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("PipelineResult 已保存: %s", path)

    @classmethod
    def load(cls, path: Path) -> "PipelineResult":
        """从 JSON 文件恢复"""
        data = json.loads(path.read_text(encoding="utf-8"))
        return _deserialize_pipeline_result(data)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────


def _slugify(topic: str) -> str:
    """将主题转换为安全的目录名"""
    import re
    slug = re.sub(r"[\s\-]+", "_", topic.strip())
    slug = re.sub(r"[^\w\u4e00-\u9fa5_]", "", slug)
    slug = slug[:40]
    return slug or "untitled"


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat()


def _serialize_pipeline_result(r: PipelineResult) -> dict[str, Any]:
    """PipelineResult → JSON 友好的字典"""
    return {
        "topic": r.topic,
        "step": r.step,
        "tmp_dir": r.tmp_dir,
        "explorer_elapsed_seconds": r.explorer_elapsed_seconds,
        "searcher_elapsed_seconds": r.searcher_elapsed_seconds,
        "writer_elapsed_seconds": r.writer_elapsed_seconds,
        "created_at": r.created_at,
        "error": r.error,
        # ExplorerReport
        "explorer_report": (
            _serialize_explorer_report(r.explorer_report)
            if r.explorer_report else None
        ),
        # Searcher messages（可能是 Message 对象或 dict）
        "searcher_messages": _to_dict(r.searcher_messages),
        "searcher_result": r.searcher_result,
        "searcher_papers": [
            {
                "paper_id": getattr(p, "paper_id", ""),
                "title": getattr(p, "title", ""),
                "abstract": getattr(p, "abstract", ""),
                "depth": getattr(p, "depth", 0),
                "source": getattr(p, "source", ""),
                "select_score": getattr(p, "select_score", 0.0),
            }
            for p in r.searcher_papers
        ] if r.searcher_papers else [],
        # WriterReport
        "writer_report": (
            _serialize_writer_report(r.writer_report)
            if r.writer_report else None
        ),
    }


def _deserialize_pipeline_result(data: dict[str, Any]) -> PipelineResult:
    """dict → PipelineResult"""
    er_data = data.get("explorer_report")
    wr_data = data.get("writer_report")
    return PipelineResult(
        topic=data["topic"],
        step=data["step"],
        tmp_dir=data.get("tmp_dir", ""),
        explorer_elapsed_seconds=data.get("explorer_elapsed_seconds", 0.0),
        searcher_elapsed_seconds=data.get("searcher_elapsed_seconds", 0.0),
        writer_elapsed_seconds=data.get("writer_elapsed_seconds", 0.0),
        created_at=data.get("created_at", ""),
        error=data.get("error", ""),
        explorer_report=(
            _deserialize_explorer_report(er_data) if er_data else None
        ),
        searcher_messages=data.get("searcher_messages", []),
        searcher_result=data.get("searcher_result", ""),
        searcher_papers=[
            type("PaperNode", (), {
                "paper_id": p.get("paper_id", ""),
                "title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "depth": p.get("depth", 0),
                "source": p.get("source", ""),
                "select_score": p.get("select_score", 0.0),
            })()
            for p in data.get("searcher_papers", [])
        ],
        writer_report=(
            _deserialize_writer_report(wr_data) if wr_data else None
        ),
    )


def _to_dict(obj: Any) -> Any:
    """通用的 Python 对象转 dict（支持 dataclass slots、普通 dataclass、dict、list）"""
    from dataclasses import is_dataclass

    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if is_dataclass(obj):
        if hasattr(obj, "__slots__"):
            return {k: _to_dict(getattr(obj, k)) for k in obj.__slots__ if hasattr(obj, k)}
        return {k: _to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _serialize_explorer_report(er) -> dict[str, Any]:
    """ExplorerReport → dict（处理循环引用和复杂类型）"""
    return _to_dict(er)


def _deserialize_explorer_report(data: dict[str, Any]) -> ExplorerReport:
    """dict → ExplorerReport"""
    from src.explorer.explorer_report import BenchmarkEntry, ClassicWork

    classics = [
        ClassicWork(title=c["title"], year=c.get("year", 0), authors=c.get("authors", ""),
                    venue=c.get("venue", ""), key_idea=c.get("key_idea", ""), impact=c.get("impact", ""))
        for c in data.get("stage2_classics", [])
        if isinstance(c, dict) and "title" in c
    ]
    benchmarks = [
        BenchmarkEntry(name=b.get("name", ""), url=b.get("url", ""),
                       description=b.get("description", ""),
                       metric=b.get("metric", ""), dataset=b.get("dataset", ""))
        for b in data.get("stage3_benchmarks", [])
        if isinstance(b, dict)
    ]
    er = ExplorerReport(topic=data.get("topic", ""))
    for k, v in data.items():
        if hasattr(er, k) and k not in ("stage2_classics", "stage3_benchmarks"):
            setattr(er, k, v)
    er.stage2_classics = classics
    er.stage3_benchmarks = benchmarks
    return er


def _serialize_writer_report(wr) -> dict[str, Any]:
    """WriterReport → dict"""
    return _to_dict(wr)


def _deserialize_writer_report(data: dict[str, Any]) -> WriterReport:
    """dict → WriterReport"""
    if not data:
        return WriterReport()
    wr = WriterReport()
    for k, v in data.items():
        if hasattr(wr, k):
            setattr(wr, k, v)
    return wr


# ─────────────────────────────────────────────────────────────────────────────
# Core 主体
# ─────────────────────────────────────────────────────────────────────────────


class ReviewForge:
    """ReviewForge 核心编排器

    三阶段流水线：Explorer → Searcher → Writer
    中间结果自动保存到 tmp/{topic_slug}/，支持从任意步骤恢复。
    """

    def __init__(
        self,
        llm: LLM,
        config: CoreConfig | None = None,
        tmp_root: Path | str | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or CoreConfig()
        self.tmp_root = Path(tmp_root) if tmp_root else Path(__file__).parent.parent / "tmp"
        self.tmp_root.mkdir(parents=True, exist_ok=True)

        # 各阶段 Agent 实例（懒加载）
        self._explorer: ExplorerAgent | None = None
        self._searcher: SearcherAgent | None = None
        self._multi_searcher: MultiSourceSearcher | None = None
        self._bfs_searcher: BFSSearcher | None = None
        self._writer: WriterAgent | None = None

    # ── 工厂方法 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, config: CoreConfig | None = None) -> "ReviewForge":
        """从 src.config.settings 构建实例"""
        from src.config import settings

        llm = LLM(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            max_concurrency=settings.llm_max_concurrency,
        )
        return cls(llm=llm, config=config)

    # ── Agent 访问器（懒加载）───────────────────────────────────────────────

    @property
    def explorer(self) -> ExplorerAgent:
        if self._explorer is None:
            self._explorer = ExplorerAgent(llm=self.llm, verbose=self.config.verbose)
        return self._explorer

    @property
    def searcher(self) -> SearcherAgent:
        if self._searcher is None:
            self._searcher = SearcherAgent(llm=self.llm, verbose=self.config.verbose)
        return self._searcher

    @property
    def multi_searcher(self) -> MultiSourceSearcher:
        if self._multi_searcher is None:
            self._multi_searcher = MultiSourceSearcher(llm=self.llm)
        return self._multi_searcher

    @property
    def bfs_searcher(self) -> BFSSearcher:
        if self._bfs_searcher is None:
            from src.config import settings

            embed = self._load_embed_client()
            reranker = self._load_reranker_client()
            self._bfs_searcher = BFSSearcher(
                llm=self.llm,
                embed=embed,
                reranker=reranker,
                search_queries_count=self.config.bfs_search_queries_count,
                search_papers_count=self.config.bfs_search_papers_count,
                expand_papers_count=self.config.bfs_expand_papers_count,
                expand_layers=self.config.bfs_expand_layers,
                score_threshold=self.config.bfs_similarity_threshold,
                rerank_top_n=self.config.bfs_rerank_top_n,
            )
        return self._bfs_searcher

    def _load_embed_client(self):
        """"懒加载 embedding 客户端"""
        try:
            from src.embedding import EmbeddingClient
            from src.config import settings

            return EmbeddingClient()
        except Exception as e:
            logger.warning("Embedding 客户端初始化失败: %s", e)
            return None

    def _load_reranker_client(self):
        """懒加载 reranker 客户端"""
        try:
            from src.reranker import RerankerClient
            from src.config import settings
            return RerankerClient()
        except Exception as e:
            logger.warning("Reranker 客户端初始化失败: %s", e)
            return None

    @property
    def writer(self) -> WriterAgent:
        if self._writer is None:
            writer_config = WriterConfig(
                max_tokens=self.config.writer_max_tokens,
                temperature=self.config.temperature,
            )
            self._writer = WriterAgent(llm=self.llm, config=writer_config)
        return self._writer

    # ── 主流程 ─────────────────────────────────────────────────────────────

    def run(self, topic: str) -> PipelineResult:
        """执行完整流水线：Explorer → Searcher → Writer"""
        slug = _slugify(topic)
        tmp_dir = self.tmp_root / slug
        tmp_dir.mkdir(parents=True, exist_ok=True)

        result = PipelineResult(topic=topic, tmp_dir=str(tmp_dir), created_at=_now())

        # Step 1: Explorer
        try:
            logger.info("[Step 1/3] Explorer 开始探索: %s", topic)
            t0 = time.time()
            result.explorer_report = self.explorer.run(topic)
            result.explorer_elapsed_seconds = time.time() - t0
            result.step = "explorer_done"
            self._save_result(result, tmp_dir, "step1_explorer_done.json")
            logger.info(
                "[Step 1/3] Explorer 完成，耗时 %.1fs，共 %d 次查询",
                result.explorer_elapsed_seconds,
                result.explorer_report.total_queries,
            )
        except Exception as e:
            logger.error("Explorer 步骤失败: %s", e, exc_info=True)
            result.error = f"explorer: {e}"
            self._save_result(result, tmp_dir, "step1_explorer_done.json")
            raise

        # Step 2: Searcher (BFSSearcher + MultiSourceSearcher 并行)
        try:
            logger.info("[Step 2/3] Searcher 开始检索: %s", topic)
            t0 = time.time()

            # BFSSearcher: 使用 explorer_report 生成更精准的查询
            # MultiSourceSearcher: 补充搜索（去掉论文搜索）
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2) as pool:
                bfs_future = pool.submit(
                    self.bfs_searcher.search_with_explorer_report,
                    topic,
                    result.explorer_report,
                )
                multi_future = pool.submit(
                    self.multi_searcher.run,
                    topic,
                )

                papers = bfs_future.result()
                search_text, messages = multi_future.result()

            search_text = self._format_papers_for_writer(papers)
            logger.info("[Step 2] BFSSearcher 获取 %d 篇论文", len(papers))

            result.searcher_messages = messages
            result.searcher_result = search_text
            result.searcher_papers = papers
            result.searcher_elapsed_seconds = time.time() - t0
            result.step = "searcher_done"
            self._save_result(result, tmp_dir, "step2_searcher_done.json")
            logger.info(
                "[Step 2/3] Searcher 完成，耗时 %.1fs",
                result.searcher_elapsed_seconds,
            )
        except Exception as e:
            logger.error("Searcher 步骤失败: %s", e, exc_info=True)
            result.error = f"searcher: {e}"
            self._save_result(result, tmp_dir, "step2_searcher_done.json")
            raise

        # Step 3: Writer
        try:
            logger.info("[Step 3/3] Writer 开始写作: %s", topic)
            t0 = time.time()
            result.writer_report = self.writer.write(
                topic=topic,
                explorer_report=result.explorer_report,
            )
            result.writer_elapsed_seconds = time.time() - t0
            result.step = "complete"
            self._save_result(result, tmp_dir, "step3_writer_done.json")
            logger.info(
                "[Step 3/3] Writer 完成，耗时 %.1fs",
                result.writer_elapsed_seconds,
            )
            self._write_visualization_json(result, tmp_dir)

            # 可选：OpenAI Responses API 质量增强
            self._run_adapter_enhance(result, tmp_dir)
        except Exception as e:
            logger.error("Writer 步骤失败: %s", e, exc_info=True)
            result.error = f"writer: {e}"
            self._save_result(result, tmp_dir, "step3_writer_done.json")
            raise

        return result

    # ── 步骤级 API（独立调用）─────────────────────────────────────────────

    @staticmethod
    def run_explorer(topic: str, llm: LLM, verbose: bool = False) -> ExplorerReport:
        """单独运行 Explorer（不经过 Core 实例）"""
        agent = ExplorerAgent(llm=llm, verbose=verbose)
        return agent.run(topic)

    @staticmethod
    def run_searcher(
        topic: str,
        explorer_report: ExplorerReport,
        llm: LLM,
        max_turns: int = 2,
        max_results: int = 50,
    ) -> tuple[str, list[dict[str, str]]]:
        """单独运行 Searcher（Explorer 结果作为输入）

        Returns:
            (search_text, messages) — messages 可传入下次调用保持上下文
        """
        searcher = SearcherAgent(llm=llm)
        multi = MultiSourceSearcher(llm=llm)
        keywords_text = searcher.run_with_explorer_report(topic, explorer_report)
        search_text, messages = multi.run(topic, messages=[], max_results=max_results)
        return search_text, messages

    @staticmethod
    def run_writer(
        topic: str,
        explorer_report: ExplorerReport,
        llm: LLM,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> WriterReport:
        """单独运行 Writer（Explorer 结果作为输入）"""
        config = WriterConfig(max_tokens=max_tokens, temperature=temperature)
        writer = WriterAgent(llm=llm, config=config)
        return writer.write(topic=topic, explorer_report=explorer_report)

    # ── 恢复机制 ──────────────────────────────────────────────────────────

    @classmethod
    def resume_from(cls, path: str | Path) -> PipelineResult:
        """从中间结果文件恢复，跳过已完成的步骤"""
        path = Path(path)
        result = PipelineResult.load(path)

        if result.step == "complete":
            logger.info("流程已完成，直接返回结果（step=%s）", result.step)
            return result

        logger.info("从 step=%s 恢复流程", result.step)
        return result  # caller 可根据 result.step 判断当前状态

    def resume(self, topic: str) -> PipelineResult:
        """在已有 tmp 目录的情况下恢复流程"""
        slug = _slugify(topic)
        tmp_dir = self.tmp_root / slug

        # 找最新的中间文件
        step_files = sorted(tmp_dir.glob("step*.json"))
        if not step_files:
            # 没有中间文件，从头开始
            logger.info("tmp/%s 无中间文件，从头开始", slug)
            return self.run(topic)

        latest = step_files[-1]
        logger.info("从 %s 恢复（step 文件: %s）", slug, latest.name)
        result = PipelineResult.load(latest)

        if result.step == "explorer_done":
            return self._resume_from_explorer_done(result, tmp_dir)
        if result.step == "searcher_done":
            return self._resume_from_searcher_done(result, tmp_dir)

        logger.info("step=%s 无对应恢复逻辑，从头开始", result.step)
        return self.run(topic)

    def _resume_from_explorer_done(
        self, result: PipelineResult, tmp_dir: Path
    ) -> PipelineResult:
        """从 Explorer 完成状态恢复"""
        # Step 2: Searcher
        try:
            t0 = time.time()
            messages: list[dict[str, str]] = []
            keywords_text = self.searcher.run_with_explorer_report(
                result.topic, result.explorer_report
            )
            search_text, messages = self.multi_searcher.run(
                result.topic,
                messages=messages,
            )
            result.searcher_messages = messages
            result.searcher_result = search_text
            result.searcher_elapsed_seconds = time.time() - t0
            result.step = "searcher_done"
            self._save_result(result, tmp_dir, "step2_searcher_done.json")
        except Exception as e:
            result.error = f"searcher: {e}"
            self._save_result(result, tmp_dir, "step2_searcher_done.json")
            raise

        # Step 3: Writer
        try:
            t0 = time.time()
            result.writer_report = self.writer.write(
                topic=result.topic,
                explorer_report=result.explorer_report,
            )
            result.writer_elapsed_seconds = time.time() - t0
            result.step = "complete"
            self._save_result(result, tmp_dir, "step3_writer_done.json")
            self._write_visualization_json(result, tmp_dir)

            # 可选：OpenAI Responses API 质量增强
            self._run_adapter_enhance(result, tmp_dir)
        except Exception as e:
            result.error = f"writer: {e}"
            self._save_result(result, tmp_dir, "step3_writer_done.json")
            raise

        return result

    def _resume_from_searcher_done(
        self, result: PipelineResult, tmp_dir: Path
    ) -> PipelineResult:
        """从 Searcher 完成状态恢复"""
        try:
            t0 = time.time()
            result.writer_report = self.writer.write(
                topic=result.topic,
                explorer_report=result.explorer_report,
            )
            result.writer_elapsed_seconds = time.time() - t0
            result.step = "complete"
            self._save_result(result, tmp_dir, "step3_writer_done.json")
            self._write_visualization_json(result, tmp_dir)

            # 可选：OpenAI Responses API 质量增强
            self._run_adapter_enhance(result, tmp_dir)
        except Exception as e:
            result.error = f"writer: {e}"
            self._save_result(result, tmp_dir, "step3_writer_done.json")
            raise

        return result

    def _write_visualization_json(
        self, result: PipelineResult, tmp_dir: Path
    ) -> None:
        """将 PipelineResult 转换为 VisualizationData 并写入 visualization_data.json。"""
        try:
            from src.adapter.script_adapter import writer_json_to_visualization

            from dataclasses import asdict
            viz_data = writer_json_to_visualization(asdict(result))
            viz_path = tmp_dir / "visualization_data.json"
            with open(viz_path, "w", encoding="utf-8") as f:
                json.dump(viz_data.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
            logger.info("visualization_data.json 已写入: %s", viz_path)
        except Exception as exc:
            logger.warning("写入 visualization_data.json 失败（非致命）: %s", exc)

    def _format_papers_for_writer(self, papers: list) -> str:
        """将 PaperNode 列表格式化为 Writer 可读的文本"""
        lines = []
        for i, p in enumerate(papers, 1):
            title = p.title or "Untitled"
            paper_id = p.paper_id or ""
            score = f", score={p.select_score:.3f}" if p.select_score else ""
            source = f", source={p.source}" if p.source else ""
            depth = f", depth={p.depth}" if p.depth else ""
            abstract = p.abstract or ""
            lines.append(
                f"### [{i}] {title}\n"
                f"**ID:** {paper_id}{depth}{source}{score}\n\n"
                f"{abstract}\n"
            )
        return "\n\n".join(lines)

    def _save_result(self, result: PipelineResult, tmp_dir: Path, filename: str) -> None:
        path = tmp_dir / filename
        result.save(path)

    def _run_adapter_enhance(
        self, result: PipelineResult, tmp_dir: Path
    ) -> None:
        """可选的 OpenAI Responses API 质量增强阶段"""
        try:
            from src.config import settings

            cfg = settings.adapter_enhance
            if not cfg:
                logger.info("[adapter-enhance] not configured")
                return
            enabled = cfg.get("enabled", False) if isinstance(cfg, dict) else getattr(cfg, "enabled", False)
            if not enabled:
                logger.info("[adapter-enhance] disabled in config")
                return
            api_key = cfg.get("api_key", "") if isinstance(cfg, dict) else getattr(cfg, "api_key", "")
            if not (api_key or "").strip():
                logger.info("[adapter-enhance] no api_key in config, skipping")
                return

            # 读取刚生成的 visualization_data.json
            viz_path = tmp_dir / "visualization_data.json"
            if not viz_path.exists():
                logger.warning("[adapter-enhance] visualization_data.json not found, skipping")
                return

            with open(viz_path, encoding="utf-8") as f:
                viz_json = json.load(f)

            # 构建 raw 字典（供 enhancer 使用）
            raw: dict[str, Any] = {"topic": result.topic}
            if result.explorer_report:
                raw["explorer_report"] = result.explorer_report

            # 将 settings 中的 adapter_enhance 写入临时 yaml 文件供 enhancer 读取
            import tempfile, yaml as yaml_lib

            adapter_cfg = cfg if isinstance(cfg, dict) else {
                "enabled": getattr(cfg, "enabled", False),
                "openai": {
                    "api_key": getattr(cfg, "api_key", ""),
                    "base_url": getattr(cfg, "base_url", "https://api.openai.com/v1"),
                    "model": getattr(cfg, "model", "gpt-4o"),
                    "timeout_seconds": getattr(cfg, "timeout_seconds", 120),
                    "max_output_tokens": getattr(cfg, "max_output_tokens", 5000),
                    "temperature": getattr(cfg, "temperature", 0.1),
                },
                "web_search": {
                    "enabled": getattr(cfg, "web_search_enabled", True),
                    "tool_type": getattr(cfg, "web_search_tool_type", "web_search_preview"),
                },
                "enhancement": {
                    "max_raw_chars": getattr(cfg, "max_raw_chars", 12000),
                    "max_viz_chars": getattr(cfg, "max_viz_chars", 20000),
                    "allow_fact_field_updates": getattr(cfg, "allow_fact_field_updates", False),
                },
                "debug": {
                    "enabled": getattr(cfg, "debug_enabled", False),
                },
            }

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
                yaml_lib.safe_dump(adapter_cfg, tmp, allow_unicode=True)
                tmp_path = tmp.name

            # 调用增强器
            from src.visualizer.schema import VisualizationData

            viz_data = VisualizationData(**viz_json)
            enhanced = enhance_with_openai_responses(
                viz_data,
                raw,
                config_path=tmp_path,
                force=True,
            )

            # 删除临时配置文件
            import os as os_mod
            os_mod.unlink(tmp_path)

            # 覆盖 visualization_data.json
            with open(viz_path, "w", encoding="utf-8") as f:
                json.dump(enhanced.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
            logger.info("[adapter-enhance] done, enhanced visualization_data.json")

            # 同时更新 step3_writer_done.json 中的 viz 数据
            step3_path = tmp_dir / "step3_writer_done.json"
            if step3_path.exists():
                with open(step3_path, encoding="utf-8") as f:
                    step3_data = json.load(f)
                step3_data["visualization_data"] = enhanced.model_dump(mode="json")
                with open(step3_path, "w", encoding="utf-8") as f:
                    json.dump(step3_data, f, ensure_ascii=False, indent=2)

        except Exception as exc:
            logger.warning("[adapter-enhance] 增强失败（非致命）: %s", exc)

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有已保存的 session（按主题）"""
        sessions = []
        if not self.tmp_root.exists():
            return sessions
        for d in sorted(self.tmp_root.iterdir()):
            if d.is_dir():
                step_files = sorted(d.glob("step*.json"))
                current_step = "none"
                if step_files:
                    last = PipelineResult.load(step_files[-1])
                    current_step = last.step
                sessions.append({
                    "topic": d.name,
                    "path": str(d),
                    "current_step": current_step,
                    "last_modified": d.stat().st_mtime,
                })
        return sessions
