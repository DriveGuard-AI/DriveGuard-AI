"""API-003 / API-004 事件查询。"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query

from ...deps import AppContext, get_context, require_permission
from ...errors import InvalidTimeRangeError, ValidationError
from ...models import EVENT_STATUSES, EVENT_TYPES, RISK_LEVELS
from ...repositories.events import EventFilters, MAX_PAGE_SIZE, ensure_utc
from ...schemas import EventListResponse, EventOut
from ...security import PERMISSION_MONITORING_READ

router = APIRouter(tags=["events"])


def _check_enum(value: str | None, allowed: tuple[str, ...], field: str) -> None:
    if value is not None and value not in allowed:
        raise ValidationError([{"field": field, "reason": f"must be one of {list(allowed)}"}])


def _check_uuid(value: str | None, field: str) -> None:
    if value is None:
        return
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError):
        raise ValidationError([{"field": field, "reason": "must be a UUID"}]) from None


@router.get(
    "/events",
    response_model=EventListResponse,
    dependencies=[Depends(require_permission(PERMISSION_MONITORING_READ))],
)
def list_events(
    context: AppContext = Depends(get_context),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=MAX_PAGE_SIZE),
    event_type: str | None = Query(None),
    risk_level: str | None = Query(None),
    status: str | None = Query(None),
    started_from: datetime | None = Query(None),
    started_to: datetime | None = Query(None),
    session_id: str | None = Query(None),
) -> EventListResponse:
    _check_enum(event_type, EVENT_TYPES, "event_type")
    _check_enum(risk_level, RISK_LEVELS, "risk_level")
    _check_enum(status, EVENT_STATUSES, "status")
    _check_uuid(session_id, "session_id")

    if started_from and started_to and ensure_utc(started_to) < ensure_utc(started_from):
        raise InvalidTimeRangeError()

    filters = EventFilters(
        event_type=event_type,
        risk_level=risk_level,
        status=status,
        started_from=ensure_utc(started_from) if started_from else None,
        started_to=ensure_utc(started_to) if started_to else None,
        session_id=session_id,
    )
    return context.events.list_events(filters, page=page, page_size=page_size)


@router.get(
    "/events/{event_id}",
    response_model=EventOut,
    dependencies=[Depends(require_permission(PERMISSION_MONITORING_READ))],
)
def get_event(event_id: str, context: AppContext = Depends(get_context)) -> EventOut:
    _check_uuid(event_id, "event_id")
    return context.events.get_event(event_id)


__all__ = ["router"]
