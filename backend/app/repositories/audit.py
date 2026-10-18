"""审计日志仓储。"""

from __future__ import annotations

import json
import uuid

from sqlalchemy.orm import Session

from ..models import AuditLog


class AuditRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self,
        *,
        action: str,
        resource_type: str,
        result: str,
        occurred_at: str,
        actor_id: str | None = None,
        resource_id: str | None = None,
        request_id: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        self._session.add(
            AuditLog(
                audit_id=str(uuid.uuid4()),
                occurred_at=occurred_at,
                actor_id=actor_id or "system",
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                result=result,
                request_id=request_id,
                details_json=json.dumps(_redact(details or {}), ensure_ascii=False),
            )
        )
        self._session.flush()


def _redact(details: dict[str, object]) -> dict[str, object]:
    """去敏：不写入疑似凭证的值。"""

    redacted: dict[str, object] = {}
    for key, value in details.items():
        lowered = key.lower()
        if any(marker in lowered for marker in ("token", "secret", "password", "key")):
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


__all__ = ["AuditRepository"]
