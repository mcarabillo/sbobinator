"""API routes for audio upload and conversion.

Endpoints:
  - POST   /api/audio/upload              → Upload a file to MinIO/S3
  - POST   /api/audio/convert             → Start async transcription (returns task_id)
  - GET    /api/audio/convert/{task_id}   → Poll conversion status
  - GET    /api/audio/convert/{task_id}/output → Download generated transcription file
  - DELETE /api/audio/convert/{task_id}   → Cancel a running task

The transcription output is always plain text (raw transcription).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from backend.modules.preprocessing.audio_processor import (
    AudioProcessor,
    AudioProcessingError,
)
from backend.modules.storage.fetcher import AudioFetcher, FetchError
from backend.modules.transcription.whisper_engine import (
    TranscriptionEngine,
    TranscriptionError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audio", tags=["audio"])

# ---------------------------------------------------------------------------
# Task state management (in-memory, thread-safe)
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    """Possible states of a conversion task."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConversionTask:
    """Represents a single async conversion task."""

    def __init__(
        self,
        task_id: str,
        key: str,
        bucket: str,
    ) -> None:
        self.task_id = task_id
        self.key = key
        self.bucket = bucket
        self.status = TaskStatus.PENDING
        self.progress: float = 0.0
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.output_key: str | None = None
        """S3 key where the transcription file was saved (e.g. ``audio/uploads/file.srt``)."""
        self.duration: float | None = None
        """Audio duration in seconds."""
        self.sample_rate: int | None = None
        """Audio sample rate."""
        self.original_format: str | None = None
        """Original audio format (e.g. ``mp3``)."""
        self.original_sample_rate: int | None = None
        """Original audio sample rate."""
        self.language: str | None = None
        """Detected language."""
        self.segment_count: int | None = None
        """Number of transcription segments."""
        self.processing_time_ms: float | None = None
        """Total processing time in milliseconds."""
        self.error: str | None = None
        self._cancelled = threading.Event()

    def to_dict(self) -> dict[str, Any]:
        """Serialize task state for API response."""
        data: dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status.value,
            "key": self.key,
            "bucket": self.bucket,
            "progress": self.progress,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
        # Only include metadata (no text content)
        if self.status == TaskStatus.COMPLETED and self.output_key:
            data["output_key"] = self.output_key
            data["duration"] = self.duration
            data["sample_rate"] = self.sample_rate
            data["original_format"] = self.original_format
            data["original_sample_rate"] = self.original_sample_rate
            data["language"] = self.language
            data["segment_count"] = self.segment_count
            data["processing_time_ms"] = self.processing_time_ms
        if self.error:
            data["error"] = self.error
        return data


# Global task store (thread-safe via lock)
_tasks: dict[str, ConversionTask] = {}
_tasks_lock = threading.Lock()


def _get_task(task_id: str) -> ConversionTask:
    """Get a task by ID or raise 404."""
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return task


def _set_task(task: ConversionTask) -> None:
    """Store a task in the global store."""
    with _tasks_lock:
        _tasks[task.task_id] = task


# ---------------------------------------------------------------------------
# Shared instances (lazy-init to avoid loading Whisper at startup)
# ---------------------------------------------------------------------------

_audio_processor: AudioProcessor | None = None
_audio_fetcher: AudioFetcher | None = None
_transcription_engine: TranscriptionEngine | None = None


def _get_processor() -> AudioProcessor:
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor()
    return _audio_processor


def _get_fetcher() -> AudioFetcher:
    global _audio_fetcher
    if _audio_fetcher is None:
        _audio_fetcher = AudioFetcher()
    return _audio_fetcher


def _get_engine() -> TranscriptionEngine:
    global _transcription_engine
    if _transcription_engine is None:
        _transcription_engine = TranscriptionEngine()
    return _transcription_engine


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
    """Ignored — output is always plain text."""


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
# Background conversion worker
# ---------------------------------------------------------------------------


def _run_conversion(task_id: str) -> None:
    """Background worker that executes the full conversion pipeline.

    Updates the task state as it progresses and saves the output file
    to S3/MinIO.
    """
    task = _get_task(task_id)

    try:
        from utils.config import settings

        cfg = settings()
        target_bucket = task.bucket or cfg.storage.bucket

        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now(timezone.utc)
        task.progress = 0.1

        # --- Step 1: Fetch audio from S3/MinIO ---
        logger.info("[%s] Fetching audio from s3://%s/%s", task_id, target_bucket, task.key)
        fetcher = _get_fetcher()
        fetch_result = fetcher.fetch(key=task.key, bucket=target_bucket)
        task.progress = 0.3

        # --- Step 2: Preprocess audio ---
        logger.info("[%s] Preprocessing audio...", task_id)
        processor = _get_processor()
        processed = processor.process(
            raw_bytes=fetch_result.data,
            content_type=fetch_result.content_type,
            filename=fetch_result.key,
        )
        task.progress = 0.6

        # --- Step 3: Transcribe ---
        logger.info("[%s] Transcribing (%.2fs)...", task_id, processed.duration)
        engine = _get_engine()
        result = engine.transcribe(
            audio_data=processed.data,
            sr=processed.sample_rate,
            audio_duration=processed.duration,
        )
        task.progress = 0.8

        # --- Step 4: Build raw text output ---
        output_content = result.text
        task.progress = 0.9

        # --- Step 5: Save output file to S3/MinIO ---
        # Replace the audio extension with .txt
        parts = task.key.rsplit(".", 1)
        output_key = f"{parts[0]}.txt" if len(parts) == 2 else f"{task.key}.txt"
        mime_type = "text/plain"

        logger.info("[%s] Saving output to s3://%s/%s", task_id, target_bucket, output_key)
        fetcher.upload(
            key=output_key,
            data=output_content.encode("utf-8"),
            content_type=mime_type,
            bucket=target_bucket,
        )

        # --- Finalize task ---
        elapsed_ms = (time.time() - task.started_at.timestamp()) * 1000

        task.output_key = output_key
        task.duration = processed.duration
        task.sample_rate = processed.sample_rate
        task.original_format = processed.original_format
        task.original_sample_rate = processed.original_sample_rate
        task.language = result.language
        task.segment_count = len(result.segments)
        task.processing_time_ms = round(elapsed_ms, 1)
        task.status = TaskStatus.COMPLETED
        task.progress = 1.0
        task.completed_at = datetime.now(timezone.utc)

        logger.info(
            "[%s] Completed: %d segments, %.2fs, output=%s (%.1fms)",
            task_id,
            len(result.segments),
            processed.duration,
            output_key,
            elapsed_ms,
        )

    except FetchError as exc:
        task.status = TaskStatus.FAILED
        task.error = f"Fetch error: {exc}"
        task.completed_at = datetime.now(timezone.utc)
        logger.error("[%s] Fetch failed: %s", task_id, exc)

    except AudioProcessingError as exc:
        task.status = TaskStatus.FAILED
        task.error = f"Processing error: {exc}"
        task.completed_at = datetime.now(timezone.utc)
        logger.error("[%s] Processing failed: %s", task_id, exc)

    except TranscriptionError as exc:
        task.status = TaskStatus.FAILED
        task.error = f"Transcription error: {exc}"
        task.completed_at = datetime.now(timezone.utc)
        logger.error("[%s] Transcription failed: %s", task_id, exc)

    except ValueError as exc:
        task.status = TaskStatus.FAILED
        task.error = f"Format error: {exc}"
        task.completed_at = datetime.now(timezone.utc)
        logger.error("[%s] Format error: %s", task_id, exc)

    except Exception as exc:
        task.status = TaskStatus.FAILED
        task.error = f"Unexpected error: {exc}"
        task.completed_at = datetime.now(timezone.utc)
        logger.error("[%s] Unexpected error: %s", task_id, exc, exc_info=True)


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
    from utils.config import settings

    cfg = settings()
    target_bucket = bucket or cfg.storage.bucket

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    raw_data = await file.read()

    # --- Size check ---
    max_size = cfg.storage.max_audio_size
    if len(raw_data) > max_size:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {len(raw_data)} bytes "
                f"(max {max_size:,})"
            ),
        )

    content_type = file.content_type or "application/octet-stream"

    # --- Build destination key ---
    if destination:
        key = destination
    else:
        ext = Path(file.filename).suffix.lstrip(".") or "audio"
        key = f"audio/uploads/{uuid.uuid4().hex}.{ext}"

    # --- Upload to S3/MinIO ---
    fetcher = _get_fetcher()
    try:
        fetcher.upload(
            key=key,
            data=raw_data,
            content_type=content_type,
            bucket=target_bucket,
        )
    except FetchError as exc:
        logger.error("Upload failed for %s: %s", key, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info("Uploaded %s → s3://%s/%s (%d bytes)", file.filename, target_bucket, key, len(raw_data))

    return UploadResponse(
        key=key,
        bucket=target_bucket,
        size_bytes=len(raw_data),
        content_type=content_type,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
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
    from utils.config import settings

    cfg = settings()
    target_bucket = payload.bucket or cfg.storage.bucket

    # --- Validate: file must exist in S3/MinIO ---
    fetcher = _get_fetcher()
    try:
        fetcher.fetch(key=payload.key, bucket=target_bucket)
    except FetchError as exc:
        logger.error("File not found in storage: %s", payload.key)
        raise HTTPException(status_code=404, detail=f"File not found in storage: {payload.key}")

    # --- Create task ---
    task_id = uuid.uuid4().hex
    task = ConversionTask(
        task_id=task_id,
        key=payload.key,
        bucket=target_bucket,
    )
    _set_task(task)

    # --- Queue background work ---
    background_tasks.add_task(_run_conversion, task_id)

    logger.info("Started conversion task %s for %s", task_id, payload.key)

    return TaskResponse(**task.to_dict())


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
    task = _get_task(task_id)
    return TaskResponse(**task.to_dict())


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
        The transcription file (plain text).

    Raises
    ------
    HTTPException 404:
        If the task doesn't exist or the output file hasn't been generated.
    HTTPException 400:
        If the task hasn't completed yet.
    """
    task = _get_task(task_id)

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed (status: {task.status.value}).",
        )

    if not task.output_key:
        raise HTTPException(
            status_code=404,
            detail="Output file not found.",
        )

    # --- Fetch output file from S3/MinIO ---
    fetcher = _get_fetcher()
    try:
        fetch_result = fetcher.fetch(key=task.output_key, bucket=task.bucket)
    except FetchError as exc:
        logger.error("Failed to fetch output file %s: %s", task.output_key, exc)
        raise HTTPException(status_code=500, detail=f"Failed to fetch output file: {exc}")

    # --- Return file ---
    filename = Path(task.output_key).name

    return Response(
        content=fetch_result.data,
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
    task = _get_task(task_id)

    if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
        return TaskResponse(**task.to_dict())

    task._cancelled.set()
    task.status = TaskStatus.CANCELLED
    task.completed_at = datetime.now(timezone.utc)

    logger.info("Cancelled task %s", task_id)
    return TaskResponse(**task.to_dict())


@router.get("/convert", response_model=list[TaskResponse])
async def list_conversions() -> list[TaskResponse]:
    """List all conversion tasks."""
    with _tasks_lock:
        return [TaskResponse(**t.to_dict()) for t in _tasks.values()]
