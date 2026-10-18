"""配置服务：读取（secret 不回显）与乐观锁更新。"""

from __future__ import annotations

from typing import Any

from ..database import Database
from ..errors import ConfigVersionConflictError, ValidationError
from ..repositories.audit import AuditRepository
from ..repositories.settings import SettingsRepository, decode
from ..schemas import SettingItem, SettingsResponse, SettingsUpdateResponse
from ..settings_registry import GLOBAL_REVISION_KEY, get_spec
from ..timeutil import parse_iso, to_iso_z, utcnow


class SettingsService:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ---------------------------------------------------------------- 读取

    def get_settings(self) -> SettingsResponse:
        with self._db.session() as session:
            repo = SettingsRepository(session)
            repo.ensure_defaults(now_iso=to_iso_z(utcnow()))
            version = repo.get_global_version()
            items = [
                _to_item(record)
                for record in repo.list_all()
                if record.setting_key != GLOBAL_REVISION_KEY
            ]
        return SettingsResponse(version=version, items=items)

    def read_runtime_value(self, key: str) -> Any:
        with self._db.session() as session:
            return SettingsRepository(session).get_value(key)

    # ---------------------------------------------------------------- 更新

    def update_settings(
        self,
        *,
        version: int,
        changes: dict[str, Any],
        actor: str,
        request_id: str | None,
    ) -> SettingsUpdateResponse:
        if not changes:
            raise ValidationError([{"field": "changes", "reason": "at least one change is required"}])

        errors: list[dict[str, Any]] = []
        for key, value in changes.items():
            spec = get_spec(key)
            if spec is None:
                errors.append({"field": key, "reason": "unknown setting key"})
                continue
            reason = spec.validate(value)
            if reason:
                errors.append({"field": key, "reason": reason})

        # 跨字段一致性：warning 阈值必须低于 danger 阈值
        merged: dict[str, Any] = {}
        with self._db.session() as session:
            repo = SettingsRepository(session)
            for key in ("risk.warning_threshold", "risk.danger_threshold"):
                merged[key] = changes.get(key, repo.get_value(key))
        warning = merged.get("risk.warning_threshold")
        danger = merged.get("risk.danger_threshold")
        if isinstance(warning, (int, float)) and isinstance(danger, (int, float)) and warning >= danger:
            errors.append(
                {
                    "field": "risk.warning_threshold",
                    "reason": "must be lower than risk.danger_threshold",
                }
            )

        if errors:
            raise ValidationError(errors)

        now = utcnow()
        now_iso = to_iso_z(now)
        with self._db.session() as session:
            repo = SettingsRepository(session)
            current_version = repo.get_global_version()
            if current_version != version:
                raise ConfigVersionConflictError(version, current_version)

            applied: dict[str, Any] = {}
            restart_required: list[str] = []
            for key, value in changes.items():
                spec = get_spec(key)
                assert spec is not None  # 已在上方校验
                repo.set_value(key, value, value_type=spec.value_type, actor=actor, now_iso=now_iso)
                applied[key] = "***" if spec.is_secret else value
                if spec.requires_restart:
                    restart_required.append(key)

            new_version = current_version + 1
            repo.set_global_version(new_version, actor=actor, now_iso=now_iso)
            AuditRepository(session).add(
                action="settings.update",
                resource_type="system_settings",
                resource_id=",".join(changes.keys()),
                result="success",
                occurred_at=now_iso,
                actor_id=actor,
                request_id=request_id,
                details={"changed_keys": sorted(changes.keys()), "new_version": new_version},
            )

        return SettingsUpdateResponse(
            version=new_version,
            applied=applied,
            restart_required=restart_required,
            rejected={},
        )


def _to_item(record) -> SettingItem:
    is_secret = bool(record.is_secret)
    value = None if is_secret else decode(record.value_type, record.value_json)
    is_configured = bool(decode(record.value_type, record.value_json)) if is_secret else True
    return SettingItem(
        key=record.setting_key,
        value_type=record.value_type,
        value=value,
        is_secret=is_secret,
        is_configured=is_configured,
        requires_restart=bool(record.requires_restart),
        updated_at=parse_iso(record.updated_at),
        applied=True,
    )


__all__ = ["SettingsService"]
