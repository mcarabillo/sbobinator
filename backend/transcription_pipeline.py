"""Transcription pipeline — facade for all transcription operations.

This module is the single point of entry for any transcription-related
operation. It orchestrates the business logic by calling the appropriate
backend modules and returns results to the caller.

Public API
----------
    from backend.transcription_pipeline import TranscriptionPipeline

    pipeline = TranscriptionPipeline.get_instance()

    # Upload an audio file to S3/MinIO
    result = pipeline.upload_audio(file_bytes, content_type, bucket, destination)

    # Start an async transcription job
    task_id = pipeline.start_conversion(key, bucket, output_format)

    # Poll task status
    info = pipeline.get_task_status(task_id)

    # Get the transcription output (raw text or formatted)
    output = pipeline.get_task_output(task_id)

    # Cancel a running task
    info = pipeline.cancel_task(task_id)

    # List all tasks
    tasks = pipeline.list_tasks()

"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from backend.modules.output.formatter import OutputFormatter
from backend.modules.preprocessing.audio_processor import (
    AudioProcessor,
    AudioProcessingError,
)
from backend.modules.storage.fetcher import (
    AudioFetcher,
    FetchError,
)
from backend.modules.transcription.whisper_engine import (
    TranscriptionEngine,
    TranscriptionError,
    TranscriptionResult,
)
from utils.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    """Possible states of a conversion task."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UploadResult:
    """Result of a successful audio upload."""

    key: str
    bucket: str
    size_bytes: int
    content_type: str
    uploaded_at: datetime


@dataclass
class TaskInfo:
    """Immutable snapshot of a task's state."""

    task_id: str
    status: str
    key: str
    bucket: str
    progress: float
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output_key: str | None = None
    duration: float | None = None
    sample_rate: int | None = None
    original_format: str | None = None
    original_sample_rate: int | None = None
    language: str | None = None
    segment_count: int | None = None
    processing_time_ms: float | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Internal task representation (mutable, thread-safe)
# ---------------------------------------------------------------------------


class _ConversionTask:
    """Mutable task state used internally."""

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
        self.duration: float | None = None
        self.sample_rate: int | None = None
        self.original_format: str | None = None
        self.original_sample_rate: int | None = None
        self.language: str | None = None
        self.segment_count: int | None = None
        self.processing_time_ms: float | None = None
        self.error: str | None = None
        self._cancelled = threading.Event()
        self.output_format: str = "txt"

    def to_info(self) -> TaskInfo:
        """Convert to an immutable TaskInfo snapshot."""
        return TaskInfo(
            task_id=self.task_id,
            status=self.status.value,
            key=self.key,
            bucket=self.bucket,
            progress=self.progress,
            started_at=self.started_at,
            completed_at=self.completed_at,
            output_key=self.output_key,
            duration=self.duration,
            sample_rate=self.sample_rate,
            original_format=self.original_format,
            original_sample_rate=self.original_sample_rate,
            language=self.language,
            segment_count=self.segment_count,
            processing_time_ms=self.processing_time_ms,
            error=self.error,
        )


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class TranscriptionPipeline:
    """Facade for all transcription operations.

    Orchestrates the full pipeline:
        1. Fetch audio from S3/MinIO
        2. Preprocess audio (decode, resample, clean)
        3. Transcribe with Whisper
        4. Format output (optional, controlled by config)
        5. Save result to S3/MinIO

    This is a singleton — use ``get_instance()`` to obtain the shared
    instance.
    """

    _instance: TranscriptionPipeline | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._cfg = settings()
        self._use_formatter = self._cfg.output.use_formatter
        self._output_format = self._cfg.output.format

        # Lazy-initialized modules
        self._audio_fetcher: AudioFetcher | None = None
        self._audio_processor: AudioProcessor | None = None
        self._transcription_engine: TranscriptionEngine | None = None
        self._output_formatter: OutputFormatter | None = None

        # Thread-safe task store
        self._tasks: dict[str, _ConversionTask] = {}
        self._tasks_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> TranscriptionPipeline:
        """Return the singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Lazy module initialisation
    # ------------------------------------------------------------------

    @property
    def _fetcher(self) -> AudioFetcher:
        if self._audio_fetcher is None:
            self._audio_fetcher = AudioFetcher()
        return self._audio_fetcher

    @property
    def _processor(self) -> AudioProcessor:
        if self._audio_processor is None:
            self._audio_processor = AudioProcessor()
        return self._audio_processor

    @property
    def _engine(self) -> TranscriptionEngine:
        if self._transcription_engine is None:
            self._transcription_engine = TranscriptionEngine()
        return self._transcription_engine

    @property
    def _formatter(self) -> OutputFormatter:
        if self._output_formatter is None:
            self._output_formatter = OutputFormatter()
        return self._output_formatter

    # ------------------------------------------------------------------
    # Task store helpers
    # ------------------------------------------------------------------

    def _get_task(self, task_id: str) -> _ConversionTask:
        """Get a task by ID or raise KeyError."""
        with self._tasks_lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return task

    def _set_task(self, task: _ConversionTask) -> None:
        """Store a task in the global store."""
        with self._tasks_lock:
            self._tasks[task.task_id] = task

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_audio(
        self,
        file_bytes: bytes,
        content_type: str,
        bucket: str | None = None,
        destination: str | None = None,
    ) -> UploadResult:
        """Upload an audio file to S3/MinIO.

        Parameters
        ----------
        file_bytes:
            Raw audio file bytes.
        content_type:
            MIME type of the audio file.
        bucket:
            Target bucket (defaults to config).
        destination:
            Custom key inside the bucket. Auto-generated if not provided.

        Returns
        -------
        UploadResult
            Metadata about the uploaded file.
        """
        from utils.config import settings as get_settings

        cfg = get_settings()
        target_bucket = bucket or cfg.storage.bucket

        if destination:
            key = destination
        else:
            ext = Path(content_type).suffix.lstrip(".") or "audio"
            key = f"audio/uploads/{uuid.uuid4().hex}.{ext}"

        self._fetcher.upload(
            key=key,
            data=file_bytes,
            content_type=content_type,
            bucket=target_bucket,
        )

        logger.info("Uploaded %d bytes → s3://%s/%s", len(file_bytes), target_bucket, key)

        return UploadResult(
            key=key,
            bucket=target_bucket,
            size_bytes=len(file_bytes),
            content_type=content_type,
            uploaded_at=datetime.now(timezone.utc),
        )

    def start_conversion(
        self,
        key: str,
        bucket: str | None = None,
        output_format: str = "txt",
    ) -> str:
        """Start an async transcription job.

        Validates that the source file exists in S3/MinIO, creates a task,
        and returns the ``task_id`` immediately. The actual work is performed
        by calling ``process_task(task_id)`` in a background thread.

        Parameters
        ----------
        key:
            S3/MinIO object key (e.g. ``'audio/uploads/file.mp3'``).
        bucket:
            Override the default bucket.
        output_format:
            Output format (``json``, ``srt``, ``vtt``, ``txt``, ``csv``,
            ``tsv``). Only used when ``use_formatter`` is True.

        Returns
        -------
        str
            The ``task_id`` for this conversion job.
        """
        from utils.config import settings as get_settings

        cfg = get_settings()
        target_bucket = bucket or cfg.storage.bucket

        # Validate: file must exist in S3/MinIO
        self._fetcher.fetch(key=key, bucket=target_bucket)

        # Create task
        task_id = uuid.uuid4().hex
        task = _ConversionTask(
            task_id=task_id,
            key=key,
            bucket=target_bucket,
        )
        task.output_format = output_format
        self._set_task(task)

        logger.info("Started conversion task %s for %s", task_id, key)
        return task_id

    def process_task(self, task_id: str) -> None:
        """Execute the full conversion pipeline for a given task.

        This is the core orchestration method called by background workers.
        """
        task = self._get_task(task_id)

        try:
            from utils.config import settings as get_settings

            cfg = get_settings()
            target_bucket = task.bucket or cfg.storage.bucket

            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now(timezone.utc)
            task.progress = 0.1

            # --- Step 1: Fetch audio from S3/MinIO ---
            logger.info("[%s] Fetching audio from s3://%s/%s", task_id, target_bucket, task.key)
            fetch_result = self._fetcher.fetch(key=task.key, bucket=target_bucket)
            task.progress = 0.3

            # --- Step 2: Preprocess audio ---
            logger.info("[%s] Preprocessing audio...", task_id)
            processed = self._processor.process(
                raw_bytes=fetch_result.data,
                content_type=fetch_result.content_type,
                filename=fetch_result.key,
            )
            task.progress = 0.6

            # --- Step 3: Transcribe ---
            logger.info("[%s] Transcribing (%.2fs)...", task_id, processed.duration)
            result = self._engine.transcribe(
                audio_data=processed.data,
                sr=processed.sample_rate,
                audio_duration=processed.duration,
            )
            task.progress = 0.8

            # --- Step 4: Build output ---
            if self._use_formatter:
                # Use OutputFormatter for formatted output
                fmt = task.output_format or self._output_format
                output_content = self._formatter.format(
                    segments=result.segments,
                    fmt=fmt,
                )
                output_key = self._formatter.build_output_key(task.key, fmt)
                mime_type = self._formatter.get_mime_type(fmt)
            else:
                # Raw text — no timestamps, no formatting
                output_content = result.text
                parts = task.key.rsplit(".", 1)
                output_key = f"{parts[0]}.txt" if len(parts) == 2 else f"{task.key}.txt"
                mime_type = "text/plain"
            task.progress = 0.9

            logger.info("[%s] Saving output to s3://%s/%s", task_id, target_bucket, output_key)
            self._fetcher.upload(
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

    def get_task_status(self, task_id: str) -> TaskInfo:
        """Get the current status of a task.

        Parameters
        ----------
        task_id:
            The task ID returned by ``start_conversion()``.

        Returns
        -------
        TaskInfo
            Current task state (metadata only, no transcription text).
        """
        task = self._get_task(task_id)
        return task.to_info()

    def get_task_output(self, task_id: str) -> str:
        """Get the transcription output for a completed task.

        Parameters
        ----------
        task_id:
            The task ID returned by ``start_conversion()``.

        Returns
        -------
        str
            The transcription content (raw text or formatted, depending on config).

        Raises
        ------
        KeyError:
            If the task doesn't exist.
        ValueError:
            If the task hasn't completed yet.
        """
        task = self._get_task(task_id)

        if task.status != TaskStatus.COMPLETED:
            raise ValueError(
                f"Task is not completed (status: {task.status.value})."
            )

        if not task.output_key:
            raise ValueError("Output file not found.")

        # Fetch output file from S3/MinIO
        fetch_result = self._fetcher.fetch(key=task.output_key, bucket=task.bucket)
        return fetch_result.data.decode("utf-8")

    def cancel_task(self, task_id: str) -> TaskInfo:
        """Cancel a running task.

        Parameters
        ----------
        task_id:
            The task ID to cancel.

        Returns
        -------
        TaskInfo
            Updated task state.
        """
        task = self._get_task(task_id)

        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return task.to_info()

        task._cancelled.set()
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now(timezone.utc)

        logger.info("Cancelled task %s", task_id)
        return task.to_info()

    def list_tasks(self) -> list[TaskInfo]:
        """List all conversion tasks.

        Returns
        -------
        list[TaskInfo]
            All tasks with their current state.
        """
        with self._tasks_lock:
            return [task.to_info() for task in self._tasks.values()]
