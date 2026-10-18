"""统计服务。

`docs/requirements/REQUIREMENTS.md` FR-008：
- 无数据时返回空统计结果，不得以服务异常代替
- 时间跨度过大返回 `422 TIME_RANGE_TOO_LARGE`

已丢弃（discarded）的候选不构成危险事件，因此不计入统计。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from ..database import Database
from ..errors import InvalidTimeRangeError, TimeRangeTooLargeError
from ..repositories.events import EventRepository
from ..schemas import CountByRisk, CountByType, StatisticsSummary, TrendBucket

MAX_RANGE_DAYS = 90


class StatisticsService:
    def __init__(self, db: Database) -> None:
        self._db = db

    def summary(self, *, started_from: datetime, started_to: datetime) -> StatisticsSummary:
        if started_to < started_from:
            raise InvalidTimeRangeError()
        if started_to - started_from > timedelta(days=MAX_RANGE_DAYS):
            raise TimeRangeTooLargeError(MAX_RANGE_DAYS)

        with self._db.session() as session:
            raw = EventRepository(session).summary(started_from=started_from, started_to=started_to)

        return StatisticsSummary(
            started_from=started_from,
            started_to=started_to,
            total_events=raw["total_events"],
            by_type=[CountByType(**row) for row in raw["by_type"]],
            by_risk_level=[CountByRisk(**row) for row in raw["by_risk_level"]],
            trend=[TrendBucket(**row) for row in raw["trend"]],
        )


__all__ = ["MAX_RANGE_DAYS", "StatisticsService"]
