"""时间工具：统一 UTC，并在存储/传输时使用 `...Z` 形式。"""

from __future__ import annotations

from datetime import datetime, timezone

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_iso_z(moment: datetime | None = None) -> str:
    value = moment or utcnow()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    return f"{value.strftime(ISO_FORMAT)}.{value.microsecond // 1000:03d}Z"


def parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def duration_ms(started_at: str, ended_at: str) -> int:
    delta = parse_iso(ended_at) - parse_iso(started_at)
    return max(0, int(delta.total_seconds() * 1000))


__all__ = ["duration_ms", "parse_iso", "to_iso_z", "utcnow"]
