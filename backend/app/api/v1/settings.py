"""API-006 / API-007 系统配置。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...deps import AppContext, get_context, get_request_id, require_permission
from ...schemas import SettingsResponse, SettingsUpdateRequest, SettingsUpdateResponse
from ...security import (
    PERMISSION_SETTINGS_READ,
    PERMISSION_SETTINGS_WRITE,
    Principal,
)

router = APIRouter(tags=["settings"])


@router.get(
    "/settings",
    response_model=SettingsResponse,
    dependencies=[Depends(require_permission(PERMISSION_SETTINGS_READ))],
)
def get_settings(context: AppContext = Depends(get_context)) -> SettingsResponse:
    return context.settings.get_settings()


@router.put("/settings", response_model=SettingsUpdateResponse)
def update_settings(
    payload: SettingsUpdateRequest,
    context: AppContext = Depends(get_context),
    principal: Principal = Depends(require_permission(PERMISSION_SETTINGS_WRITE)),
    request_id: str | None = Depends(get_request_id),
) -> SettingsUpdateResponse:
    return context.settings.update_settings(
        version=payload.version,
        changes=payload.changes,
        actor=principal.subject,
        request_id=request_id,
    )


__all__ = ["router"]
