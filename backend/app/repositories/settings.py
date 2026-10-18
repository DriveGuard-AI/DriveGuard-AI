"""系统配置仓储：默认值播种与乐观锁更新。"""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import SystemSetting
from ..settings_registry import GLOBAL_REVISION_KEY, SETTING_SPECS
from ..timeutil import to_iso_z


def encode(value_type: str, value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def decode(value_type: str, raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw if value_type == "string" else None


class SettingsRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    # ---------------------------------------------------------------- 播种

    def ensure_defaults(self, *, now_iso: str) -> None:
        for spec in SETTING_SPECS:
            if self._session.get(SystemSetting, spec.key) is None:
                self._session.add(
                    SystemSetting(
                        setting_key=spec.key,
                        value_json=encode(spec.value_type, spec.default),
                        value_type=spec.value_type,
                        version=1,
                        requires_restart=1 if spec.requires_restart else 0,
                        is_secret=1 if spec.is_secret else 0,
                        updated_at=now_iso,
                        updated_by="system",
                    )
                )
        if self._session.get(SystemSetting, GLOBAL_REVISION_KEY) is None:
            self._session.add(
                SystemSetting(
                    setting_key=GLOBAL_REVISION_KEY,
                    value_json=encode("integer", 1),
                    value_type="integer",
                    version=1,
                    requires_restart=0,
                    is_secret=0,
                    updated_at=now_iso,
                    updated_by="system",
                )
            )
        self._session.flush()

    # ---------------------------------------------------------------- 读取

    def list_all(self) -> list[SystemSetting]:
        stmt = select(SystemSetting).order_by(SystemSetting.setting_key)
        return list(self._session.execute(stmt).scalars())

    def get(self, key: str) -> SystemSetting | None:
        return self._session.get(SystemSetting, key)

    def get_value(self, key: str) -> object | None:
        record = self.get(key)
        if record is None:
            return None
        return decode(record.value_type, record.value_json)

    def get_int(self, key: str, default: int) -> int:
        record = self.get(key)
        if record is None:
            return default
        value = decode(record.value_type, record.value_json)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default

    def get_float(self, key: str, default: float) -> float:
        record = self.get(key)
        if record is None:
            return default
        value = decode(record.value_type, record.value_json)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default

    def get_bool(self, key: str, default: bool) -> bool:
        record = self.get(key)
        if record is None:
            return default
        value = decode(record.value_type, record.value_json)
        return bool(value) if isinstance(value, bool) else default

    def get_global_version(self) -> int:
        record = self.get(GLOBAL_REVISION_KEY)
        if record is None:
            return 1
        value = decode(record.value_type, record.value_json)
        return int(value) if isinstance(value, int) else 1

    # ---------------------------------------------------------------- 写入

    def set_value(self, key: str, value: object, *, value_type: str, actor: str, now_iso: str) -> int:
        record = self.get(key)
        if record is None:
            record = SystemSetting(
                setting_key=key,
                value_json=encode(value_type, value),
                value_type=value_type,
                version=1,
                requires_restart=0,
                is_secret=0,
                updated_at=now_iso,
                updated_by=actor,
            )
            self._session.add(record)
        else:
            record.value_json = encode(value_type, value)
            record.version = record.version + 1
            record.updated_at = now_iso
            record.updated_by = actor
        self._session.flush()
        return record.version

    def set_global_version(self, version: int, *, actor: str, now_iso: str) -> None:
        record = self.get(GLOBAL_REVISION_KEY)
        if record is None:
            self._session.add(
                SystemSetting(
                    setting_key=GLOBAL_REVISION_KEY,
                    value_json=encode("integer", version),
                    value_type="integer",
                    version=version,
                    requires_restart=0,
                    is_secret=0,
                    updated_at=now_iso,
                    updated_by=actor,
                )
            )
        else:
            record.value_json = encode("integer", version)
            record.version = version
            record.updated_at = now_iso
            record.updated_by = actor
        self._session.flush()


__all__ = ["SettingsRepository", "decode", "encode"]
