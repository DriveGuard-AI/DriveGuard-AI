"""离线分析任务服务。

职责：
- 校验素材来源（路径必须落在媒体根目录内，帧数与磁盘占用受限，见 NFR-003）；
- 通过 `InferenceEngine.run_analysis()` 切换推理源并开启新会话；
- 把执行进度与终态持久化到 `analysis_jobs`，供页面轮询。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import time
import uuid
from pathlib import Path

from ..config import AppConfig
from ..database import Database
from ..errors import (
    AnalysisJobConflictError,
    AnalysisJobNotFoundError,
    InvalidAnalysisSourceError,
    PipelineUnavailableError,
)
from ..inference.port import AnalysisSource
from ..repositories.analysis_jobs import AnalysisJobRepository
from ..repositories.audit import AuditRepository
from ..repositories.sessions import SessionRepository
from ..schemas import AnalysisJobCreate, AnalysisJobOut
from ..timeutil import to_iso_z, utcnow
from .frame_extraction import (
    STEP_MS_CHOICES,
    VIDEO_SUFFIXES,
    DecoderUnavailableError,
    ExtractionError,
    extract_frames,
    validate_options,
)
from .monitoring import InferenceEngine

logger = logging.getLogger(__name__)

SUPPORTED_SOURCE_KINDS = ("frames",)
JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
PROGRESS_INTERVAL_SECONDS = 1.0
PURGE_INTERVAL_SECONDS = 3600.0
UPLOAD_CHUNK_BYTES = 1024 * 1024


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
        self._purge_task: asyncio.Task | None = None

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
        self._check_disk_budget(media_root, field="source_path")
        return request.source_kind, resolved, len(frames)

    def _check_disk_budget(self, media_root: Path, *, field: str) -> None:
        used_mb = self._media_usage_mb(media_root)
        if used_mb > self._config.analysis_max_disk_mb:
            raise InvalidAnalysisSourceError(
                f"媒体目录已占用 {used_mb:.0f} MB，超过上限 {self._config.analysis_max_disk_mb} MB",
                field=field,
            )

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
        return await self._launch(
            source_kind=kind,
            source_path=str(resolved),
            frames_total=frames_total,
        )

    async def create_from_upload(
        self,
        stream,
        *,
        filename: str | None,
        step_ms: int,
        size: str,
        quality: int,
    ) -> AnalysisJobOut:
        """上传一段视频，抽帧后立即开跑。

        抽帧在请求内完成（放线程里），因此申请不到的解码依赖、坏文件、超限视频
        都会在**建任务之前**报错，不会留下一个永远 queued 的空任务。
        """

        suffix = Path(filename or "").suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise InvalidAnalysisSourceError(
                f"只接受视频文件 {sorted(VIDEO_SUFFIXES)}，收到 {filename!r}", field="file"
            )
        try:
            validate_options(step_ms=step_ms, size=size, quality=quality)
        except ExtractionError as error:
            raise InvalidAnalysisSourceError(str(error)) from None

        media_root = self._config.media_dir.resolve()
        media_root.mkdir(parents=True, exist_ok=True)
        self._check_disk_budget(media_root, field="file")

        job_id = str(uuid.uuid4())
        job_dir = media_root / "analysis" / job_id
        source = job_dir / f"source{suffix}"
        frames_dir = job_dir / "frames"
        try:
            job_dir.mkdir(parents=True, exist_ok=False)
            digest, size_bytes = await asyncio.to_thread(self._spool, stream, source)
            frames_total = await asyncio.to_thread(
                extract_frames,
                source,
                frames_dir,
                step_ms=step_ms,
                size=size,
                quality=quality,
                max_frames=self._config.analysis_max_frames,
            )
            self._write_manifest(
                job_dir,
                {
                    "job_id": job_id,
                    "source_filename": filename,
                    "source_bytes": size_bytes,
                    "source_sha256": digest,
                    "step_ms": step_ms,
                    "size": size,
                    "quality": quality,
                    "frames_total": frames_total,
                    "created_at": to_iso_z(utcnow()),
                },
            )
            # NFR-003：默认不留原始视频；抽帧成功后立即删除。
            if not self._config.analysis_retain_source:
                source.unlink(missing_ok=True)
            return await self._launch(
                job_id=job_id,
                source_kind="frames",
                source_path=str(frames_dir),
                frames_total=frames_total,
                source_sha256=digest,
            )
        except DecoderUnavailableError as error:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise PipelineUnavailableError(str(error)) from None
        except ExtractionError as error:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise InvalidAnalysisSourceError(str(error)) from None
        except Exception:
            # 校验失败、抽帧半途失败、落库失败都不该留下半个任务目录。
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

    def _spool(self, stream, target: Path) -> tuple[str, int]:
        """把上传流写成文件并计算摘要，同时执行大小上限。"""

        limit = self._config.analysis_max_upload_mb * 1024 * 1024
        digest = hashlib.sha256()
        written = 0
        with target.open("wb") as sink:
            while True:
                chunk = stream.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise InvalidAnalysisSourceError(
                        f"文件超过 {self._config.analysis_max_upload_mb} MB 上限", field="file"
                    )
                digest.update(chunk)
                sink.write(chunk)
        if written == 0:
            raise InvalidAnalysisSourceError("上传文件为空", field="file")
        return digest.hexdigest(), written

    def _write_manifest(self, job_dir: Path, payload: dict) -> None:
        (job_dir / "job.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def _launch(
        self,
        *,
        source_kind: str,
        source_path: str,
        frames_total: int,
        source_sha256: str | None = None,
        job_id: str | None = None,
    ) -> AnalysisJobOut:
        """切换推理源并登记任务记录（目录来源与上传来源共用）。

        上传来源已经在磁盘上建好了 `analysis/<job_id>/`，此时任务号必须由调用方传入，
        否则目录名与 `job_id` 对不上，按任务号找不到产物。
        """

        session_id = await self._engine.run_analysis(
            AnalysisSource(kind=source_kind, frame_sequence=source_path)
        )
        try:
            with self._db.session() as session:
                job = AnalysisJobRepository(session).create(
                    job_id=job_id or str(uuid.uuid4()),
                    session_id=session_id,
                    source_kind=source_kind,
                    source_path=source_path,
                    frames_total=frames_total,
                    source_sha256=source_sha256,
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

    # ------------------------------------------------------------ 产物保留期

    def start(self) -> None:
        """启动保留期清理循环（幂等）。"""

        if self._purge_task is None or self._purge_task.done():
            self._purge_task = asyncio.create_task(
                self._purge_loop(), name="driveguard-analysis-purge"
            )

    async def _purge_loop(self) -> None:
        while True:
            try:
                removed = await asyncio.to_thread(self._purge_once)
                if removed:
                    logger.info("已清理 %d 个过期分析目录", removed)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 清理失败不得拖垮服务
                logger.warning("分析产物清理失败", exc_info=True)
            await asyncio.sleep(PURGE_INTERVAL_SECONDS)

    def _purge_once(self) -> int:
        """按保留期删除过期任务目录。

        以目录 mtime 为准：抽帧产物、`job.json` 都在任务开头一次写完，
        mtime 就是"这份产物最后一次被使用"的近似值。
        """

        root = self._config.media_dir.resolve() / "analysis"
        if not root.is_dir():
            return 0
        cutoff = time.time() - self._config.analysis_retain_frames_hours * 3600
        removed = 0
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            try:
                if path.stat().st_mtime > cutoff:
                    continue
                shutil.rmtree(path)
            except OSError:
                logger.warning("删除过期分析目录失败: %s", path, exc_info=True)
                continue
            removed += 1
            self._audit_purge(path.name)
        return removed

    def _audit_purge(self, job_id: str) -> None:
        try:
            with self._db.session() as session:
                AuditRepository(session).add(
                    action="analysis.purge",
                    resource_type="analysis_job",
                    resource_id=job_id,
                    result="success",
                    occurred_at=to_iso_z(utcnow()),
                    details={
                        "reason": "retention",
                        "retain_hours": self._config.analysis_retain_frames_hours,
                    },
                )
        except Exception:  # noqa: BLE001 - 审计失败不影响清理
            logger.warning("清理审计写入失败: %s", job_id)

    async def shutdown(self) -> None:
        for task in (self._watcher, self._purge_task):
            if task is None or task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._watcher = None
        self._purge_task = None
