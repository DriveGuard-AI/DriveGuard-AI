"""真实推理端口：把 live 服务的四任务输出映射成 `RiskSnapshot`。

三个硬约束决定了这里的结构：

1. `InferenceEngine` 在事件循环里**同步**调用 `next_snapshot`，所以本类必须非阻塞：
   后台线程负责泵帧并缓存最新结果，`next_snapshot` 只读缓存。
2. **不可用证据不得输出 SAFE**（业务规则 1）：没有可用任务时返回 WARNING +
   `quality="unknown"`，并带 `EVIDENCE_UNAVAILABLE` 原因码，而不是把缺失当成低风险。
3. live 服务遇到推理错误会关闭会话，所以出错后要重开会话并从下一帧继续；
   连续失败超过上限则标记失败，不再空转。

映射范围：只映射四任务（fatigue / phone_use / drinking / distraction）。
`gaze` 与逐帧特征（EAR/MAR/PERCLOS/头姿）目前不在 live 输出里，`FeatureSet` 保持空值，
不用占位数字填充；`detections` 同样留空。
"""

from __future__ import annotations

from datetime import datetime
import logging
import threading
import time
from typing import Any

from ..schemas import FeatureSet, RiskSnapshot
from .frame_source import FrameSource, FrameSourceError
from .live_client import LiveCamera, LiveClient, LiveError
from .port import InferencePort

logger = logging.getLogger(__name__)

TASK_NAMES = ("fatigue", "phone_use", "drinking", "distraction")
STALE_SECONDS = 3.0


def summarize_tasks(tasks: dict[str, dict] | None) -> dict[str, Any]:
    """把 live 的 tasks 汇总成风险等级、主风险、置信度与原因码。"""

    tasks = tasks or {}
    available = {
        name: value for name, value in tasks.items()
        if isinstance(value, dict) and value.get("available") is True
    }
    scores = {
        name: round(float(value["score"]), 4)
        for name, value in available.items()
        if isinstance(value.get("score"), (int, float))
    }
    detected = {
        name: value for name, value in available.items()
        if value.get("detected") is True
    }

    if not available:
        # 没有任何可用证据：不能输出 SAFE。
        return {
            "risk_level": "WARNING",
            "primary_risk": None,
            "confidence": 0.0,
            "risk_scores": {},
            "reason_codes": ["EVIDENCE_UNAVAILABLE"],
            "quality": "unknown",
        }

    if detected:
        primary = max(detected, key=lambda name: scores.get(name, 0.0))
        reasons = [f"{name.upper()}_DETECTED" for name in sorted(detected)]
        return {
            "risk_level": "DANGER" if len(detected) >= 2 else "WARNING",
            "primary_risk": primary,
            "confidence": scores.get(primary, 0.0),
            "risk_scores": scores,
            "reason_codes": reasons,
            "quality": "good" if len(available) == len(TASK_NAMES) else "degraded",
        }

    return {
        "risk_level": "SAFE",
        "primary_risk": None,
        "confidence": 0.0,
        "risk_scores": scores,
        "reason_codes": ["NO_BEHAVIOUR_DETECTED"],
        "quality": "good" if len(available) == len(TASK_NAMES) else "degraded",
    }


class LiveInference(InferencePort):
    """由 live 服务驱动的实心推理端口。"""

    mode = "real"

    def __init__(
        self,
        client: LiveClient,
        camera: LiveCamera,
        source: FrameSource,
        *,
        preview_tap=None,
        stale_seconds: float = STALE_SECONDS,
        retry_seconds: float = 2.0,
        max_failures: int = 5,
        startup_wait_seconds: float = 90.0,
    ) -> None:
        self._client = client
        self._camera = camera
        self._source = source
        # 帧源装饰器（预览用）；与 `_source` 是同一对象或包着同一底层源。
        # 必须是可选的：注入的测试源没有预览概念。
        self.preview_tap = preview_tap
        self._stale_seconds = float(stale_seconds)
        self._retry_seconds = float(retry_seconds)
        self._max_failures = int(max_failures)
        self._startup_wait_seconds = float(startup_wait_seconds)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._session_id: str | None = None
        self._result: dict | None = None
        self._received_at: float | None = None
        self._state = "idle"
        self._last_error: str | None = None
        self._frames_sent = 0
        self._results_seen = 0

    # ------------------------------------------------------------ 生命周期

    def start(self) -> None:
        if self._thread is not None:
            return
        self._source.start()
        self._state = "streaming"
        self._thread = threading.Thread(target=self._run, name="driveguard-live-pump", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._thread = None
        session_id = self._session_id
        if session_id:
            self._client.close_session(session_id)
            self._session_id = None
        self._source.close()

    # ------------------------------------------------------------ 端口实现

    @property
    def degraded(self) -> bool:
        with self._lock:
            if self._last_error or self._result is None:
                return True
            tasks = self._result.get("tasks") or {}
            if len(tasks) < len(TASK_NAMES):
                return True
            return any(not (task or {}).get("available") for task in tasks.values())

    def status(self) -> dict[str, Any]:
        with self._lock:
            age = None if self._received_at is None else round(time.monotonic() - self._received_at, 3)
            state = self._state
            result = self._result
            payload = {
                "mode": self.mode,
                "degraded": self.degraded_locked(),
                "state": state,
                "frames_sent": self._frames_sent,
                "results_seen": self._results_seen,
                "last_error": self._last_error,
                "result_age_seconds": age,
                "session_active": self._session_id is not None,
                "live_url": self._client.base_url,
                "camera": {
                    "view": self._camera.view,
                    "image_size": list(self._camera.image_size),
                    "driver_roi": list(self._camera.driver_roi),
                },
            }
        tasks = (result or {}).get("tasks") or {}
        payload["tasks"] = {
            name: {
                "available": bool((tasks.get(name) or {}).get("available")),
                "detected": (tasks.get(name) or {}).get("detected"),
                "score": (tasks.get(name) or {}).get("score"),
            }
            for name in TASK_NAMES
        }
        # 帧源状态（摄像头/序列）与预览对齐信息；查询失败不影响主状态。
        reporter = getattr(self.preview_tap, "status", None)
        if callable(reporter):
            try:
                payload["source"] = reporter()
            except Exception:  # noqa: BLE001
                payload["source"] = None
        return payload

    def next_snapshot(self, *, session_id: str, sequence: int, now: datetime) -> RiskSnapshot:
        with self._lock:
            result = self._result
            received_at = self._received_at
            state = self._state
            last_error = self._last_error

        if result is None:
            reason = "LIVE_SOURCE_FAILED" if state == "failed" else "EVIDENCE_PENDING"
            return self._snapshot(
                session_id=session_id, sequence=sequence, now=now,
                risk_level="WARNING", primary_risk=None, confidence=0.0,
                quality="unknown", risk_scores={}, reason_codes=[reason],
            )

        age = None if received_at is None else time.monotonic() - received_at
        if age is None or age > self._stale_seconds:
            return self._snapshot(
                session_id=session_id, sequence=sequence, now=now,
                risk_level="WARNING", primary_risk=None, confidence=0.0,
                quality="unknown", risk_scores={}, reason_codes=["RESULT_STALE"],
            )

        summary = summarize_tasks(result.get("tasks"))
        if last_error and state != "completed":
            summary["reason_codes"] = [*summary["reason_codes"], "LIVE_SOURCE_RETRY"]
        timeline = result.get("source_timestamp_ms")
        return self._snapshot(
            session_id=session_id, sequence=sequence, now=now,
            risk_level=summary["risk_level"], primary_risk=summary["primary_risk"],
            confidence=summary["confidence"], quality=summary["quality"],
            risk_scores=summary["risk_scores"], reason_codes=summary["reason_codes"],
            source_timestamp_ms=timeline if isinstance(timeline, int) else None,
        )

    # ------------------------------------------------------------ 内部实现

    def degraded_locked(self) -> bool:
        """`status()` 已持锁时使用的无锁版本。"""

        if self._last_error or self._result is None:
            return True
        tasks = self._result.get("tasks") or {}
        if len(tasks) < len(TASK_NAMES):
            return True
        return any(not (task or {}).get("available") for task in tasks.values())

    def _snapshot(
        self,
        *,
        session_id: str,
        sequence: int,
        now: datetime,
        risk_level: str,
        primary_risk: str | None,
        confidence: float,
        quality: str,
        risk_scores: dict[str, float],
        reason_codes: list[str],
        source_timestamp_ms: int | None = None,
    ) -> RiskSnapshot:
        return RiskSnapshot(
            schema_version="1.0",
            session_id=session_id,
            sequence=sequence,
            captured_at=now,
            risk_level=risk_level,  # type: ignore[arg-type]
            primary_risk=primary_risk,
            confidence=round(float(confidence), 4),
            quality=quality,  # type: ignore[arg-type]
            risk_scores={key: round(float(value), 4) for key, value in risk_scores.items()},
            features=FeatureSet(),
            detections=[],
            reason_codes=list(reason_codes),
            inference_mode="real",
            source_timestamp_ms=source_timestamp_ms,
        )

    def _record_error(self, error: BaseException) -> None:
        message = f"{type(error).__name__}: {error}"[:400]
        with self._lock:
            self._last_error = message
        logger.warning("live 推理出错: %s", message)

    def _store(self, reply: dict) -> None:
        result = reply.get("result")
        if not isinstance(result, dict):
            return
        with self._lock:
            self._result = result
            self._received_at = time.monotonic()
            self._results_seen += 1

    def _release_stale_session(self) -> None:
        """释放同一服务上残留的会话（只应在启动、尚未开会话时调用）。"""

        try:
            status = self._client.status()
        except LiveError:
            return
        session_id = status.get("session_id") if isinstance(status, dict) else None
        if isinstance(session_id, str) and session_id:
            logger.warning("释放残留的 live 会话: %s", session_id)
            self._client.close_session(session_id)

    def _run(self) -> None:
        # 上一个进程被强杀时会话会残留，服务端要 60 秒才自然过期；
        # 应用是 live 服务的唯一客户端，启动时先释放残留会话。
        self._release_stale_session()
        failures = 0
        # 首次连接允许等满服务端的过期窗口，避免刚启动就放弃。
        startup_deadline = time.monotonic() + self._startup_wait_seconds
        while not self._stop.is_set():
            try:
                session_id = self._client.open_session(self._camera)
            except LiveError as error:
                failures += 1
                self._record_error(error)
                if self._frames_sent == 0 and time.monotonic() < startup_deadline:
                    self._stop.wait(self._retry_seconds)
                    continue
                if failures > self._max_failures:
                    with self._lock:
                        self._state = "failed"
                    return
                self._stop.wait(self._retry_seconds)
                continue

            with self._lock:
                self._session_id = session_id
                self._state = "streaming"
            completed = False
            # 每个会话的时间戳必须从 0 开始；重开会话后重新取原点。
            origin: int | None = None
            try:
                while not self._stop.is_set():
                    frame = self._source.next_frame()
                    if frame is None:
                        completed = True
                        break
                    timestamp_ms, jpeg = frame
                    if origin is None:
                        origin = timestamp_ms
                    reply = self._client.send_frame(session_id, timestamp_ms - origin, jpeg)
                    with self._lock:
                        self._frames_sent += 1
                        self._last_error = None
                        failures = 0
                    self._store(reply)
            except (LiveError, FrameSourceError) as error:
                # live 服务报错与帧源故障都要重试。帧源故障尤其不能当成
                # "素材读完"（next_frame 返回 None 才是），否则摄像头发抖
                # 会被当成正常收尾，页面直接停在最后一帧。
                failures += 1
                self._record_error(error)
            finally:
                with self._lock:
                    self._session_id = None
                self._client.close_session(session_id)

            if completed:
                with self._lock:
                    self._state = "completed"
                return
            if failures > self._max_failures:
                with self._lock:
                    self._state = "failed"
                return
            self._stop.wait(self._retry_seconds)


__all__ = ["LiveInference", "STALE_SECONDS", "TASK_NAMES", "summarize_tasks"]
