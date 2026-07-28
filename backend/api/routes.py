"""API routes for audio upload and transcription.

Endpoints:
  - POST   /api/audio/convert             → Upload file, start transcription (async)
  - POST   /api/audio/convert/{task_id}/format → Set output format
  - GET    /api/audio/convert/{task_id}   → Poll conversion status
  - GET    /api/audio/convert/{task_id}/output → Download transcription
  - POST   /api/audio/convert/{task_id}/ack → Acknowledge receipt, delete temp files
  - DELETE /api/audio/convert/{task_id}   → Cancel a running task
  - GET    /api/audio/convert             → List all tasks

Facade
------
All transcription logic is delegated to
``backend.transcription_pipeline.TranscriptionPipeline``, which is the
single point of entry for any transcription operation.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
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


class ConvertTaskResponse(BaseModel):
    """Response after a successful audio upload (task created)."""

    task_id: str
    status: str
    filename: str
    content_type: str
    size_bytes: int


class ConvertFormatRequest(BaseModel):
    """Payload to set the output format for an existing task."""

    output_format: str = "txt"
    """Output format (``json``, ``srt``, ``vtt``, ``txt``, ``csv``, ``tsv``)."""


class AckResponse(BaseModel):
    """Response after acknowledging receipt of the transcription."""

    acknowledged: bool
    task_id: str


class TaskResponse(BaseModel):
    """Response for task status polling.

    Does NOT include the transcription text — only metadata.
    Use the **output endpoint** to download the full transcription file.
    """

    task_id: str
    status: str
    progress: float
    started_at: str | None = None
    completed_at: str | None = None
    filename: str | None = None
    content_type: str | None = None
    duration: float | None = None
    sample_rate: int | None = None
    original_format: str | None = None
    original_sample_rate: int | None = None
    language: str | None = None
    segment_count: int | None = None
    processing_time_ms: float | None = None
    error: str | None = None
    acknowledged: bool = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/convert", response_model=ConvertTaskResponse)
async def convert(file: UploadFile = File(...)) -> ConvertTaskResponse:
    """Upload an audio file and start transcription.

    The transcription runs asynchronously in a background thread.
    The caller receives a ``task_id`` immediately and can poll for status
    or download the output once completed.

    Parameters
    ----------
    file:
        Audio file to upload.

    Returns
    -------
    ConvertTaskResponse
        The ``task_id`` and metadata about the upload.
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
            filename=file.filename,
        )
    except Exception as exc:
        logger.error("Upload failed for %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info("Uploaded %s → task=%s (%d bytes)", file.filename, result.task_id, len(raw_data))

    return ConvertTaskResponse(
        task_id=result.task_id,
        status=result.status,
        filename=result.filename,
        content_type=result.content_type,
        size_bytes=result.size_bytes,
    )


@router.post("/convert/{task_id}/format", response_model=TaskResponse)
async def set_convert_format(
    task_id: str,
    payload: ConvertFormatRequest,
) -> TaskResponse:
    """Set the output format for an existing upload task.

    Must be called before the transcription completes (or quickly after).

    Parameters
    ----------
    task_id:
        The task ID returned by the ``/convert`` endpoint.
    payload:
        Contains ``output_format`` (json, srt, vtt, txt, csv, tsv).

    Returns
    -------
    TaskResponse
        Updated task state.
    """
    try:
        _pipeline.convert_task(
            task_id=task_id,
            output_format=payload.output_format,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    except Exception as exc:
        logger.error("Format update failed for %s: %s", task_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

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
    """Download the generated transcription.

    Parameters
    ----------
    task_id:
        The task ID returned by the ``/convert`` endpoint.

    Returns
    -------
    Response
        The transcription content.

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

    # Determine mime type from content
    try:
        info = _pipeline.get_task_status(task_id)
        # Try to detect format from filename or default to txt
        if info.filename:
            ext = Path(info.filename).suffix.lower()
            mime_map = {
                ".json": "application/json",
                ".srt": "text/srt",
                ".vtt": "text/vtt",
                ".csv": "text/csv",
                ".tsv": "text/tab-separated-values",
            }
            mime_type = mime_map.get(ext, "text/plain")
        else:
            mime_type = "text/plain"
    except KeyError:
        mime_type = "text/plain"

    return Response(
        content=output,
        media_type=mime_type,
        headers={
            "Content-Disposition": 'attachment; filename="transcription"',
        },
    )


@router.post("/convert/{task_id}/ack", response_model=AckResponse)
async def acknowledge_output(task_id: str) -> AckResponse:
    """Acknowledge receipt of the transcription output.

    After acknowledgment, all temporary files associated with the task
    are deleted. Subsequent calls to the output endpoint will return 404.

    Parameters
    ----------
    task_id:
        The task ID returned by the ``/convert`` endpoint.

    Returns
    -------
    AckResponse
        Confirmation that the task has been acknowledged.

    Raises
    ------
    HTTPException 404:
        If the task doesn't exist.
    HTTPException 400:
        If the task hasn't completed yet or has already been acknowledged.
    """
    try:
        _pipeline.acknowledge_task(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return AckResponse(acknowledged=True, task_id=task_id)


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
        "progress": info.progress,
        "started_at": info.started_at.isoformat() if info.started_at else None,
        "completed_at": info.completed_at.isoformat() if info.completed_at else None,
        "acknowledged": info.acknowledged,
    }
    if info.filename:
        data["filename"] = info.filename
        data["content_type"] = info.content_type
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
