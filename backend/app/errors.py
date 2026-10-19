"""统一错误响应。

`docs/api/API_DESIGN.md` 第 2 节要求错误包含 `code`、`message`、`details`、
`request_id`，且不得暴露堆栈、绝对路径或凭证（NFR-004）。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    """可预期的业务错误。"""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []


class AuthRequiredError(AppError):
    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(401, "AUTH_REQUIRED", message)


class ForbiddenError(AppError):
    def __init__(self, message: str = "Permission denied") -> None:
        super().__init__(403, "FORBIDDEN", message)


class EventNotFoundError(AppError):
    def __init__(self, event_id: str) -> None:
        super().__init__(404, "EVENT_NOT_FOUND", "Event not found", [{"field": "event_id", "reason": event_id}])


class ConfigVersionConflictError(AppError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            409,
            "CONFIG_VERSION_CONFLICT",
            "Configuration version conflict",
            [{"field": "version", "reason": f"expected {actual}, received {expected}"}],
        )


class MonitoringNotActiveError(AppError):
    def __init__(self) -> None:
        super().__init__(409, "MONITORING_NOT_ACTIVE", "No active monitoring session")


class InvalidTimeRangeError(AppError):
    def __init__(self, reason: str = "started_to must not be earlier than started_from") -> None:
        super().__init__(400, "INVALID_TIME_RANGE", "Invalid time range", [{"field": "started_to", "reason": reason}])


class TimeRangeTooLargeError(AppError):
    def __init__(self, max_days: int) -> None:
        super().__init__(
            422,
            "TIME_RANGE_TOO_LARGE",
            "Requested time range exceeds the configured maximum",
            [{"field": "started_from", "reason": f"maximum range is {max_days} days"}],
        )


class ValidationError(AppError):
    def __init__(self, details: list[dict[str, Any]]) -> None:
        super().__init__(422, "VALIDATION_ERROR", "Request validation failed", details)


class PipelineUnavailableError(AppError):
    def __init__(self, message: str = "Inference pipeline unavailable") -> None:
        super().__init__(503, "PIPELINE_UNAVAILABLE", message)


class AnalysisJobNotFoundError(AppError):
    def __init__(self, job_id: str) -> None:
        super().__init__(404, "ANALYSIS_JOB_NOT_FOUND", "Analysis job not found", [{"field": "job_id", "reason": job_id}])


class AnalysisJobConflictError(AppError):
    """任务状态不允许该操作，例如取消已结束的任务。"""

    def __init__(self, reason: str) -> None:
        super().__init__(409, "ANALYSIS_JOB_CONFLICT", "Analysis job state conflict", [{"field": "status", "reason": reason}])


class InvalidAnalysisSourceError(AppError):
    """素材来源不合法：类型不支持、路径越界、目录不存在或帧数/磁盘超限。"""

    def __init__(self, reason: str, *, field: str = "source_path") -> None:
        super().__init__(400, "INVALID_ANALYSIS_SOURCE", "Invalid analysis source", [{"field": field, "reason": reason}])


class AnalysisSourceMissingError(AppError):
    """回放要用的素材不在了：原视频未保留、抽帧产物过期，或帧号越界。"""

    def __init__(self, reason: str) -> None:
        super().__init__(404, "ANALYSIS_SOURCE_MISSING", "Analysis source missing", [{"field": "source", "reason": reason}])


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _payload(code: str, message: str, details: list[dict[str, Any]], request_id: str | None) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "details": details,
        "request_id": request_id,
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.code, exc.message, exc.details, _request_id(request)),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())),
                "reason": str(error.get("msg", "invalid")),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_payload("VALIDATION_ERROR", "Request validation failed", details, _request_id(request)),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        mapping = {
            401: "AUTH_REQUIRED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
        }
        code = mapping.get(exc.status_code, "HTTP_ERROR")
        details: list[dict[str, Any]] = []
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(code, str(exc.detail), details, _request_id(request)),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover - 防御性分支
        # 不返回堆栈、绝对路径或凭证
        return JSONResponse(
            status_code=500,
            content=_payload("INTERNAL_ERROR", "Internal server error", [], _request_id(request)),
        )


__all__ = [
    "AppError",
    "AuthRequiredError",
    "ConfigVersionConflictError",
    "EventNotFoundError",
    "ForbiddenError",
    "InvalidTimeRangeError",
    "MonitoringNotActiveError",
    "PipelineUnavailableError",
    "TimeRangeTooLargeError",
    "ValidationError",
    "register_exception_handlers",
]
