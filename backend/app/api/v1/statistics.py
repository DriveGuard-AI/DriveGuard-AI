"""API-005 统计摘要。"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query

from ...deps import AppContext, get_context, require_permission
from ...repositories.events import ensure_utc
from ...schemas import StatisticsSummary
from ...security import PERMISSION_MONITORING_READ

router = APIRouter(tags=["statistics"])


@router.get(
    "/statistics/summary",
    response_model=StatisticsSummary,
    dependencies=[Depends(require_permission(PERMISSION_MONITORING_READ))],
)
def statistics_summary(
    context: AppContext = Depends(get_context),
    started_from: datetime = Query(...),
    started_to: datetime = Query(...),
) -> StatisticsSummary:
    return context.statistics.summary(
        started_from=ensure_utc(started_from),
        started_to=ensure_utc(started_to),
    )


__all__ = ["router"]
