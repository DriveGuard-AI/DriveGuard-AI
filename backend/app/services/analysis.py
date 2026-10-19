"""离线分析任务服务。

职责：
- 校验素材来源（路径必须落在媒体根目录内，帧数与磁盘占用受限，见 NFR-003）；
- 通过 `InferenceEngine.run_analysis()` 切换推理源并开启新会话；
- 把执行进度与终态持久化到 `analysis_jobs`，供页面轮询。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from ..config import AppConfig
from ..database import Database
from ..errors import (
    AnalysisJobConflictError,
    AnalysisJobNotFoundError,
    InvalidAnalysisSourceError,
)
from ..inference.port import AnalysisSource
from ..repositories.analysis_jobs import AnalysisJobRepository
from ..repositories.sessions import SessionRepository
from ..schemas import AnalysisJobCreate, AnalysisJobOut
from ..timeutil import to_iso_z, utcnow
from .monitoring import InferenceEngine

logger = logging.getLogger(__name__)

SUPPORTED_SOURCE_KINDS = ("frames",)
JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
PROGRESS_INTERVAL_SECONDS = 1.0


def job_to_schema(job) -> AnalysisJobOut:
    return AnalysisJobOut(
        job_id=job.job_id,
        session_id=job.session_id,
        source_kind=job.source_kind,
        source_path=job.source_path,
        source_sha256=job.source_sha256,
        status=job.status,
        frames_total=job.frames_total,
        frames_done=job.frames_done,
        timeline_ms=job.timeline_ms,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        ended_at=job.ended_at,
        updated_at=job.updated_at,
    )


class AnalysisService:
    def __init__(self, db: Database, config: AppConfig, engine: InferenceEngine) -> None:
        self._db = db
        self._config = config
        self._engine = engine
        self._watcher: asyncio.Task | None = None
        self._watching_job: str | None = None

    # ------------------------------------------------------------ 素材校验

    def _resolve_source(self, request: AnalysisJobCreate) -> tuple[str, Path, int]:
        if request.source_kind not in SUPPORTED_SOURCE_KINDS:
            raise InvalidAnalysisSourceError(
                f"暂不支持的来源类型 {request.source_kind}", field="source_kind"
            )
        media_root = self._config.media_dir.resolve()
        raw = Path(request.source_path)
        candidate = raw if raw.is_absolute() else media_root / raw
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise InvalidAnalysisSourceError(f"素材路径不存在: {error}") from None
        # 路径必须落在媒体根目录内，否则该接口等于允许读取服务器任意目录。
        if not resolved.is_relative_to(media_root):
            raise InvalidAnalysisSourceError("素材路径必须位于媒体根目录之下")
        if not resolved.is_dir():
            raise InvalidAnalysisSourceError("素材路径必须是目录（JPEG 序列）")
        frames = sorted(p for p in resolved.iterdir() if p.suffix.lower() in JPEG_SUFFIXES)
        if not frames:
            raise InvalidAnalysisSourceError("目录中没有 JPEG 帧")
        if len(frames) > self._config.analysis_max_frames:
            raise InvalidAnalysisSourceError(
                f"帧数 {len(frames)} 超过上限 {self._config.analysis_max_frames}"
            )
        used_mb = self._media_usage_mb(media_root)
        if used_mb > self._config.analysis_max_disk_mb:
            raise InvalidAnalysisSourceError(
                f"媒体目录已占用 {used_mb:.0f} MB，超过上限 {self._config.analysis_max_disk_mb} MB",
                field="source_path",
            )
        return request.source_kind, resolved, len(frames)

    def _media_usage_mb(self, media_root: Path) -> float:
        total = 0
        for path in media_root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total / (1024 * 1024)

    # ------------------------------------------------------------ 任务生命周期

    async def create(self, request: AnalysisJobCreate) -> AnalysisJobOut:
        kind, resolved, frames_total = self._resolve_source(request)
        session_id = await self._engine.run_analysis(
            AnalysisSource(kind=kind, frame_sequence=str(resolved))
        )
        try:
            with self._db.session() as session:
                job = AnalysisJobRepository(session).create(
                    job_id=str(uuid.uuid4()),
                    session_id=session_id,
                    source_kind=kind,
                    source_path=str(resolved),
                    frames_total=frames_total,
                    now=utcnow(),
                )
                schema = job_to_schema(job)
        except Exception:
            # 落库失败时不能把推理留在运行状态，否则后续任务无法切换素材。
            await self._engine.stop()
            raise
        self._start_watcher(schema.job_id, session_id)
        return schema

    def get(self, job_id: str) -> AnalysisJobOut:
        with self._db.session() as session:
            job = AnalysisJobRepository(session).get(job_id)
            if job is None:
                raise AnalysisJobNotFoundError(job_id)
            return job_to_schema(job)

    def list(self, *, status: str | None, page: int, page_size: int):
        with self._db.session() as session:
            jobs, total = AnalysisJobRepository(session).list(
                status=status, page=page, page_size=page_size
            )
            return [job_to_schema(job) for job in jobs], total

    async def cancel(self, job_id: str) -> AnalysisJobOut:
        with self._db.session() as session:
            job = AnalysisJobRepository(session).get(job_id)
            if job is None:
                raise AnalysisJobNotFoundError(job_id)
            if job.status in TERMINAL_STATUSES:
                raise AnalysisJobConflictError(f"任务已结束（{job.status}），无法取消")
            session_id = job.session_id
        # 停止推理会让会话结束；任务记录里保留已完成的帧数。
        await self._engine.stop()
        with self._db.session() as session:
            repository = AnalysisJobRepository(session)
            job = repository.get(job_id)
            if job is None:
                raise AnalysisJobNotFoundError(job_id)
            repository.finish(job, status="cancelled", now=utcnow())
            SessionRepository(session).end(session_id, to_iso_z(utcnow()))
            return job_to_schema(job)

    # ------------------------------------------------------------ 进度跟踪

    def _start_watcher(self, job_id: str, session_id: str) -> None:
        if self._watcher is not None and not self._watcher.done():
            self._watcher.cancel()
        self._watching_job = job_id
        self._watcher = asyncio.create_task(
            self._watch(job_id, session_id), name="driveguard-analysis-watch"
        )

    async def _watch(self, job_id: str, session_id: str) -> None:
        """把推理侧的进度与终态写回任务记录。

        不依赖客户端轮询，因此列表页看到的状态也是准的。
        """
        try:
            while True:
                await asyncio.sleep(PROGRESS_INTERVAL_SECONDS)
                if self._is_terminal(job_id):
                    # 已经被取消（或别处写入终态），跟踪任务没有继续的必要。
                    return
                status = self._engine.inference_status() or {}
                if self._engine.session_id != session_id:
                    # 会话已被别的任务顶掉，按取消处理，避免永久 running。
                    self._finish(job_id, status="cancelled", error="会话已被其他任务占用")
                    return
                frames_done = int(status.get("frames_sent") or 0)
                timeline = self._timeline_ms()
                state = str(status.get("state") or "")
                if state == "completed":
                    self._finish(
                        job_id, status="completed", frames_done=frames_done, timeline_ms=timeline
                    )
                    # 素材跑完就停掉循环并结束会话：离线任务不该继续占用
                    # 摄像头/模型，也该让本会话的 ended_at 落下来。
                    await self._stop_engine_if_current(session_id)
                    return
                if state == "failed":
                    self._finish(
                        job_id,
                        status="failed",
                        error=str(status.get("last_error") or "推理失败"),
                        frames_done=frames_done,
                        timeline_ms=timeline,
                    )
                    await self._stop_engine_if_current(session_id)
                    return
                self._progress(job_id, frames_done=frames_done, timeline_ms=timeline)
        except asyncio.CancelledError:  # pragma: no cover - 正常关闭路径
            raise

    def _is_terminal(self, job_id: str) -> bool:
        try:
            with self._db.session() as session:
                job = AnalysisJobRepository(session).get(job_id)
                return job is None or job.status in TERMINAL_STATUSES
        except Exception:  # noqa: BLE001 - 读不到就继续跟踪
            return False

    def _timeline_ms(self) -> int:
        latest = self._engine.latest()
        if latest is None or latest.source_timestamp_ms is None:
            return 0
        return int(latest.source_timestamp_ms)

    def _progress(self, job_id: str, *, frames_done: int, timeline_ms: int) -> None:
        try:
            with self._db.session() as session:
                repository = AnalysisJobRepository(session)
                job = repository.get(job_id)
                if job is None or job.status != "running":
                    return
                repository.progress(
                    job, frames_done=frames_done, timeline_ms=timeline_ms, now=utcnow()
                )
        except Exception:  # noqa: BLE001 - 进度写失败不应终止跟踪
            logger.warning("分析任务进度写入失败: %s", job_id)

    def _finish(
        self,
        job_id: str,
        *,
        status: str,
        error: str | None = None,
        frames_done: int | None = None,
        timeline_ms: int | None = None,
    ) -> None:
        try:
            with self._db.session() as session:
                repository = AnalysisJobRepository(session)
                job = repository.get(job_id)
                if job is None or job.status != "running":
                    return
                repository.finish(
                    job,
                    status=status,
                    now=utcnow(),
                    error=error,
                    frames_done=frames_done,
                    timeline_ms=timeline_ms,
                )
        except Exception:  # noqa: BLE001 - 终态写失败仅记录
            logger.warning("分析任务终态写入失败: %s", job_id)

    async def _stop_engine_if_current(self, session_id: str) -> None:
        """只停自己那一轮。追踪到终态与用户新建任务之间可能刚好错开。"""

        if self._engine.session_id != session_id:
            return
        try:
            await self._engine.stop()
        except Exception:  # noqa: BLE001 - 停不下来也不能让任务卡在 running
            logger.warning("分析任务结束后停止引擎失败: %s", session_id, exc_info=True)

    async def shutdown(self) -> None:
        if self._watcher is not None and not self._watcher.done():
            self._watcher.cancel()
            try:
                await self._watcher
            except asyncio.CancelledError:
                pass
        self._watcher = None
