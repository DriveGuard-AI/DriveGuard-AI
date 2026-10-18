"""API-008 模型元数据。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ...deps import AppContext, get_context, require_permission
from ...errors import ValidationError
from ...models import MODEL_STATUSES, MODEL_TYPES
from ...schemas import ModelInfo
from ...security import PERMISSION_MODELS_READ

router = APIRouter(tags=["models"])


def _check_enum(value: str | None, allowed: tuple[str, ...], field: str) -> None:
    if value is not None and value not in allowed:
        raise ValidationError([{"field": field, "reason": f"must be one of {list(allowed)}"}])


@router.get(
    "/models",
    response_model=list[ModelInfo],
    dependencies=[Depends(require_permission(PERMISSION_MODELS_READ))],
)
def list_models(
    context: AppContext = Depends(get_context),
    model_type: str | None = Query(None),
    status: str | None = Query(None),
) -> list[ModelInfo]:
    _check_enum(model_type, MODEL_TYPES, "model_type")
    _check_enum(status, MODEL_STATUSES, "status")
    return context.system.list_models(model_type=model_type, status=status)


__all__ = ["router"]
