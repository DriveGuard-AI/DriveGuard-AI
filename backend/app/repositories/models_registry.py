"""模型版本注册表仓储。

对应 `docs/api/API_DESIGN.md` API-008：对外只暴露工件逻辑标识，不返回宿主机绝对路径。
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import ModelVersion

# 无模型阶段：注册占位模型，明确标注为 stub，避免与真实模型混淆
STUB_MODELS: tuple[dict[str, object], ...] = (
    {
        "model_name": "object-detector-stub",
        "model_type": "detector",
        "semantic_version": "0.0.0-stub",
        "runtime": "onnxruntime",
        "artifact_path": "models/stub-detector.onnx",
        "metrics_json": {"note": "not-trained", "precision": None, "recall": None},
        "feature_schema_version": "detector/1.0",
    },
    {
        "model_name": "face-landmarker-stub",
        "model_type": "landmarker",
        "semantic_version": "0.0.0-stub",
        "runtime": "onnxruntime",
        "artifact_path": "models/stub-landmarker.onnx",
        "metrics_json": {"note": "not-trained"},
        "feature_schema_version": "landmarker/1.0",
    },
    {
        "model_name": "temporal-tcn-stub",
        "model_type": "temporal",
        "semantic_version": "0.0.0-stub",
        "runtime": "onnxruntime",
        "artifact_path": "models/stub-tcn.onnx",
        "metrics_json": {"note": "not-trained", "recall": None},
        "feature_schema_version": "features/1.0",
    },
    {
        "model_name": "risk-fusion-stub",
        "model_type": "fusion",
        "semantic_version": "0.0.0-stub",
        "runtime": "pytorch",
        "artifact_path": "models/stub-fusion.pt",
        "metrics_json": {"note": "rule-based-placeholder"},
        "feature_schema_version": "features/1.0",
    },
)


class ModelRegistryRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def ensure_stub_models(self, *, now_iso: str) -> None:
        for entry in STUB_MODELS:
            exists = self._session.execute(
                select(ModelVersion).where(
                    ModelVersion.model_name == entry["model_name"],
                    ModelVersion.semantic_version == entry["semantic_version"],
                )
            ).scalar_one_or_none()
            if exists is not None:
                continue
            self._session.add(
                ModelVersion(
                    model_version_id=str(uuid.uuid4()),
                    model_name=str(entry["model_name"]),
                    model_type=str(entry["model_type"]),
                    semantic_version=str(entry["semantic_version"]),
                    runtime=str(entry["runtime"]),
                    artifact_path=str(entry["artifact_path"]),
                    sha256="0" * 64,
                    feature_schema_version=entry.get("feature_schema_version"),  # type: ignore[arg-type]
                    metrics_json=json.dumps(entry["metrics_json"]),
                    status="active",
                    created_at=now_iso,
                    activated_at=now_iso,
                )
            )
        self._session.flush()

    def list(self, *, model_type: str | None = None, status: str | None = None) -> list[ModelVersion]:
        stmt = select(ModelVersion)
        if model_type:
            stmt = stmt.where(ModelVersion.model_type == model_type)
        if status:
            stmt = stmt.where(ModelVersion.status == status)
        stmt = stmt.order_by(ModelVersion.model_type, ModelVersion.model_name)
        return list(self._session.execute(stmt).scalars())

    def active_names(self) -> list[str]:
        stmt = select(ModelVersion.model_name).where(ModelVersion.status == "active")
        return [row for row in self._session.execute(stmt).scalars()]

    def temporal_model_id(self) -> str | None:
        stmt = (
            select(ModelVersion.model_version_id)
            .where(ModelVersion.model_type == "temporal", ModelVersion.status == "active")
            .limit(1)
        )
        return self._session.execute(stmt).scalar_one_or_none()


__all__ = ["ModelRegistryRepository", "STUB_MODELS"]
