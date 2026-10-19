"""系统状态服务（FR-010 / API-002）。"""

from __future__ import annotations

import json

from ..config import AppConfig
from ..database import Database
from ..repositories.events import EventRepository
from ..repositories.models_registry import ModelRegistryRepository
from ..repositories.settings import SettingsRepository
from ..schemas import (
    CameraStatus,
    DatabaseStatus,
    ModelInfo,
    ModelStatus,
    PipelineStatus,
    SystemStatus,
    WebSocketStatus,
)
from ..timeutil import parse_iso, to_iso_z, utcnow
from .monitoring import InferenceEngine


def _load_metrics(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}



class SystemService:
    def __init__(self, db: Database, config: AppConfig, engine: InferenceEngine) -> None:
        self._db = db
        self._config = config
        self._engine = engine

    def list_models(self, *, model_type: str | None = None, status: str | None = None) -> list[ModelInfo]:
        """API-008：只暴露工件逻辑标识，不返回宿主机绝对路径。"""

        with self._db.session() as session:
            records = ModelRegistryRepository(session).list(model_type=model_type, status=status)
        return [
            ModelInfo(
                model_version_id=record.model_version_id,
                model_name=record.model_name,
                model_type=record.model_type,
                semantic_version=record.semantic_version,
                runtime=record.runtime,
                feature_schema_version=record.feature_schema_version,
                metrics=_load_metrics(record.metrics_json),
                status=record.status,
                activated_at=parse_iso(record.activated_at) if record.activated_at else None,
            )
            for record in records
        ]

    def status(self) -> SystemStatus:
        engine_status = self._engine.status()
        detail = self._engine.inference_status()
        live = (self._config.inference_mode or "fake").lower() == "live"

        with self._db.session() as session:
            SettingsRepository(session).ensure_defaults(now_iso=to_iso_z(utcnow()))
            bundle_version = SettingsRepository(session).get_value("model.bundle_version")
            active_models = ModelRegistryRepository(session).active_names()
            event_count = EventRepository(session).count_events()

        return SystemStatus(
            service_version=self._config.service_version,
            camera=CameraStatus(
                # live 模式下输入是 JPEG 序列（由外部工具抽帧），不是摄像头
                available=bool(live and detail.get("state") == "streaming"),
                device_id="frame-sequence" if live else f"cam-{self._config.inference_scenario}",
                source="jpeg-sequence" if live else "scripted-stub",
            ),
            pipeline=PipelineStatus(
                state=engine_status["state"],
                processed_fps=engine_status["processed_fps"],
                latency_p50_ms=engine_status["latency_p50_ms"],
                latency_p95_ms=engine_status["latency_p95_ms"],
                dropped_frame_rate=engine_status["dropped_frame_rate"],
                inference_mode=engine_status["inference_mode"],  # type: ignore[arg-type]
                degraded=engine_status["degraded"],
            ),
            database=DatabaseStatus(
                writable=self._db.is_writable(),
                driver="sqlite",
            ),
            websocket=WebSocketStatus(
                clients=self._engine.client_count,
                max_hz=self._config.ws_max_hz,
            ),
            model=ModelStatus(
                # 只有 live 模式且已收到真实结果时才算“已加载模型”
                loaded=bool(live and (detail.get("results_seen") or 0) > 0),
                bundle_version=str(bundle_version) if bundle_version is not None else "unknown",
                active_models=active_models,
            ),
        )


__all__ = ["SystemService"]
