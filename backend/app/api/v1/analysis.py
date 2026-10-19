"""离线分析任务接口。

创建一次分析会切换推理源并开启新会话，因此需要写权限；查询只需读权限。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse

from ...deps import AppContext, get_context, require_permission
from ...errors import ValidationError
from ...models import JOB_STATUSES
from ...repositories.events import MAX_PAGE_SIZE
from ...schemas import (
    AnalysisEvidence,
    AnalysisJobCreate,
    AnalysisJobListResponse,
    AnalysisJobOut,
)
from ...security import (
    PERMISSION_ANALYSIS_READ,
    PERMISSION_ANALYSIS_WRITE,
    ensure_permission,
    principal_from_request,
    resolve_principal,
)

router = APIRouter(tags=["analysis"])

VIDEO_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}


def _ensure_media_read(
    context: AppContext, request: Request, access_token: str | None
) -> None:
    """媒体端点额外支持 `?access_token=`。

    `<video src>` 与 `<img src>` 无法自定义请求头，只能把凭证放 URL
    （与 WS-001 同一取舍）。其余端点仍使用 Bearer 头。
    """

    principal = (
        resolve_principal(context.config, access_token)
        if access_token
        else principal_from_request(context.config, request)
    )
    ensure_permission(principal, PERMISSION_ANALYSIS_READ)


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
    retain_source: bool | None = Form(None),
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
            retain_source=retain_source,
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


@router.get(
    "/analysis/jobs/{job_id}/evidence",
    response_model=AnalysisEvidence,
    dependencies=[Depends(require_permission(PERMISSION_ANALYSIS_READ))],
)
def job_evidence(job_id: str, context: AppContext = Depends(get_context)) -> AnalysisEvidence:
    """回放坐标：逐 tick 风险曲线（降采样）+ 事件帧区间。"""

    return context.analysis.evidence(job_id)


@router.get("/analysis/jobs/{job_id}/source")
def job_source(
    job_id: str,
    request: Request,
    access_token: str | None = Query(None),
    context: AppContext = Depends(get_context),
) -> FileResponse:
    """原始视频（仅 retain_source=true 的任务）；支持 Range，浏览器可拖动。"""

    _ensure_media_read(context, request, access_token)
    path = context.analysis.source_file(job_id)
    media_type = VIDEO_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/analysis/jobs/{job_id}/frames/{index}")
def job_frame(
    job_id: str,
    index: int,
    request: Request,
    access_token: str | None = Query(None),
    context: AppContext = Depends(get_context),
) -> FileResponse:
    """第 `index` 帧 JPEG（保留原视频时不需要，但帧序列回放靠它）。"""

    _ensure_media_read(context, request, access_token)
    path = context.analysis.frame_file(job_id, index)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=600"})
