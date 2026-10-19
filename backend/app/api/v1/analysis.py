"""离线分析任务接口。

创建一次分析会切换推理源并开启新会话，因此需要写权限；查询只需读权限。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from ...deps import AppContext, get_context, require_permission
from ...errors import ValidationError
from ...models import JOB_STATUSES
from ...repositories.events import MAX_PAGE_SIZE
from ...schemas import AnalysisJobCreate, AnalysisJobListResponse, AnalysisJobOut
from ...security import PERMISSION_ANALYSIS_READ, PERMISSION_ANALYSIS_WRITE

router = APIRouter(tags=["analysis"])


@router.post(
    "/analysis/jobs",
    response_model=AnalysisJobOut,
    status_code=201,
    dependencies=[Depends(require_permission(PERMISSION_ANALYSIS_WRITE))],
)
async def create_job(
    request: AnalysisJobCreate,
    context: AppContext = Depends(get_context),
) -> AnalysisJobOut:
    return await context.analysis.create(request)


@router.post(
    "/analysis/jobs/upload",
    response_model=AnalysisJobOut,
    status_code=201,
    dependencies=[Depends(require_permission(PERMISSION_ANALYSIS_WRITE))],
)
async def upload_job(
    file: UploadFile = File(...),
    step_ms: int = Form(40),
    size: str = Form("1280x720"),
    quality: int = Form(85),
    context: AppContext = Depends(get_context),
) -> AnalysisJobOut:
    """上传视频 → 抽帧 → 立即开始分析。"""

    try:
        return await context.analysis.create_from_upload(
            file.file,
            filename=file.filename,
            step_ms=step_ms,
            size=size,
            quality=quality,
        )
    finally:
        await file.close()


@router.get(
    "/analysis/jobs",
    response_model=AnalysisJobListResponse,
    dependencies=[Depends(require_permission(PERMISSION_ANALYSIS_READ))],
)
def list_jobs(
    context: AppContext = Depends(get_context),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=MAX_PAGE_SIZE),
    status: str | None = Query(None),
) -> AnalysisJobListResponse:
    if status is not None and status not in JOB_STATUSES:
        raise ValidationError(
            [{"field": "status", "reason": f"must be one of {list(JOB_STATUSES)}"}]
        )
    items, total = context.analysis.list(status=status, page=page, page_size=page_size)
    return AnalysisJobListResponse(
        items=items,
        page=page,
        page_size=page_size,
        total=total,
        has_next=page * page_size < total,
    )


@router.get(
    "/analysis/jobs/{job_id}",
    response_model=AnalysisJobOut,
    dependencies=[Depends(require_permission(PERMISSION_ANALYSIS_READ))],
)
def get_job(job_id: str, context: AppContext = Depends(get_context)) -> AnalysisJobOut:
    return context.analysis.get(job_id)


@router.delete(
    "/analysis/jobs/{job_id}",
    response_model=AnalysisJobOut,
    dependencies=[Depends(require_permission(PERMISSION_ANALYSIS_WRITE))],
)
async def cancel_job(job_id: str, context: AppContext = Depends(get_context)) -> AnalysisJobOut:
    return await context.analysis.cancel(job_id)
