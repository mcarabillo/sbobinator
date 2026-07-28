"""API routes for audio upload and conversion.

Endpoints:
  - POST   /api/audio/upload              → Upload a file to MinIO/S3
  - POST   /api/audio/convert             → Start async transcription (returns task_id)
  - GET    /api/audio/convert/{task_id}   → Poll conversion status
  - GET    /api/audio/convert/{task_id}/output → Download generated transcription file
  - DELETE /api/audio/convert/{task_id}   → Cancel a running task

The transcription output is always plain text (raw transcription) by default.
Set ``SBO_OUTPUT__USE_FORMATTER=true`` in the environment to use
``OutputFormatter`` for formatted output (SRT, VTT, JSON, CSV, TSV).

Facade
------
All transcription logic is delegated to
``backend.transcription_pipeline.TranscriptionPipeline``, which is the
single point of entry for any transcription operation.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from backend.transcription_pipeline import TranscriptionPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audio", tags=["audio"])

# ---------------------------------------------------------------------------
# Shared facade instance
# ---------------------------------------------------------------------------

_pipeline = TranscriptionPipeline.get_instance()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    """Response after a successful upload."""

    key: str
    bucket: str
    size_bytes: int
    content_type: str
    uploaded_at: str


class ConvertRequest(BaseModel):
    """Payload for the convert endpoint."""

    key: str
    """S3/MinIO object key (e.g. ``'audio/uploads/file.mp3'``)."""
    bucket: str | None = None
    """Override the default bucket."""

    output_format: str = "txt"
    """Ignored when ``use_formatter`` is False. When True, the output
    format (``json``, ``srt``, ``vtt``, ``txt``, ``csv``, ``tsv``)."""


class TaskResponse(BaseModel):
    """Response for task status polling.

    Does NOT include the transcription text — only metadata.
    Use the **output endpoint** to download the full transcription file.
    """

    task_id: str
    status: str
    key: str
    bucket: str
    progress: float
    started_at: str | None = None
    completed_at: str | None = None
    output_key: str | None = None
    """S3 key of the generated transcription file."""
    duration: float | None = None
    sample_rate: int | None = None
    original_format: str | None = None
    original_sample_rate: int | None = None
    language: str | None = None
    segment_count: int | None = None
    processing_time_ms: float | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=UploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    bucket: str | None = Form(None),
    destination: str | None = Form(None),
) -> UploadResponse:
    """Upload an audio file to MinIO/S3.

    Parameters
    ----------
    file:
        Audio file to upload.
    bucket:
        Target bucket (defaults to config).
    destination:
        Custom key inside the bucket. Auto-generated if not provided.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    raw_data = await file.read()

    content_type = file.content_type or "application/octet-stream"

    # --- Upload via facade ---
    try:
        result = _pipeline.upload_audio(
            file_bytes=raw_data,
            content_type=content_type,
            bucket=bucket,
            destination=destination,
        )
    except Exception as exc:
        logger.error("Upload failed for %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info("Uploaded %s → s3://%s/%s (%d bytes)", file.filename, result.bucket, result.key, len(raw_data))

    return UploadResponse(
        key=result.key,
        bucket=result.bucket,
        size_bytes=result.size_bytes,
        content_type=result.content_type,
        uploaded_at=result.uploaded_at.isoformat(),
    )


@router.post("/convert", response_model=TaskResponse, status_code=202)
async def convert_audio(
    payload: ConvertRequest,
    background_tasks: BackgroundTasks,
) -> TaskResponse:
    """Start an async transcription job.

    Returns a ``task_id`` immediately. Use the **status endpoint** to poll
    for completion. Use the **output endpoint** to download the generated file.

    Parameters
    ----------
    payload:
        Object containing ``key`` (S3 path) and optional ``bucket``.
    background_tasks:
        FastAPI background task handler.
    """
    # Validate and start conversion via facade
    try:
        task_id = _pipeline.start_conversion(
            key=payload.key,
            bucket=payload.bucket,
            output_format=payload.output_format,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Conversion start failed for %s: %s", payload.key, exc)
        raise HTTPException(status_code=500, detail=f"Failed to start conversion: {exc}")

    # Queue background processing
    background_tasks.add_task(_pipeline.process_task, task_id)

    logger.info("Started conversion task %s for %s", task_id, payload.key)

    # Return current status
    info = _pipeline.get_task_status(task_id)
    return TaskResponse(**_task_info_to_dict(info))


@router.get("/convert/{task_id}", response_model=TaskResponse)
async def get_conversion_status(task_id: str) -> TaskResponse:
    """Poll the status of a conversion task.

    Returns metadata only (no transcription text). Use the **output endpoint**
    to download the generated transcription file.

    Parameters
    ----------
    task_id:
        The task ID returned by the ``/convert`` endpoint.

    Returns
    -------
    TaskResponse
        Current task state with metadata (no text content).
    """
    try:
        info = _pipeline.get_task_status(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return TaskResponse(**_task_info_to_dict(info))


@router.get("/convert/{task_id}/output")
async def download_output(task_id: str) -> Response:
    """Download the generated transcription file.

    Parameters
    ----------
    task_id:
        The task ID returned by the ``/convert`` endpoint.

    Returns
    -------
    Response
        The transcription file (plain text or formatted, depending on config).

    Raises
    ------
    HTTPException 404:
        If the task doesn't exist.
    HTTPException 400:
        If the task hasn't completed yet.
    """
    try:
        output = _pipeline.get_task_output(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Determine filename from the output_key stored in the task
    try:
        info = _pipeline.get_task_status(task_id)
        filename = info.output_key.split("/")[-1] if info.output_key else "transcription.txt"
    except KeyError:
        filename = "transcription.txt"

    return Response(
        content=output,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.delete("/convert/{task_id}", response_model=TaskResponse)
async def cancel_conversion(task_id: str) -> TaskResponse:
    """Cancel a running conversion task.

    Parameters
    ----------
    task_id:
        The task ID to cancel.

    Returns
    -------
    TaskResponse
        Updated task state.
    """
    try:
        info = _pipeline.cancel_task(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return TaskResponse(**_task_info_to_dict(info))


@router.get("/convert", response_model=list[TaskResponse])
async def list_conversions() -> list[TaskResponse]:
    """List all conversion tasks."""
    tasks = _pipeline.list_tasks()
    return [TaskResponse(**_task_info_to_dict(t)) for t in tasks]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_info_to_dict(info) -> dict:
    """Convert a TaskInfo dataclass to a dict for Pydantic serialization."""
    data = {
        "task_id": info.task_id,
        "status": info.status,
        "key": info.key,
        "bucket": info.bucket,
        "progress": info.progress,
        "started_at": info.started_at.isoformat() if info.started_at else None,
        "completed_at": info.completed_at.isoformat() if info.completed_at else None,
    }
    if info.output_key:
        data["output_key"] = info.output_key
        data["duration"] = info.duration
        data["sample_rate"] = info.sample_rate
        data["original_format"] = info.original_format
        data["original_sample_rate"] = info.original_sample_rate
        data["language"] = info.language
        data["segment_count"] = info.segment_count
        data["processing_time_ms"] = info.processing_time_ms
    if info.error:
        data["error"] = info.error
    return data
