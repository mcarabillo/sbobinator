"""Transcription pipeline — facade for all transcription operations.

This module is the single point of entry for any transcription-related
operation. It orchestrates the business logic by calling the appropriate
backend modules and returns results to the caller.

Public API
----------
    from backend.transcription_pipeline import TranscriptionPipeline

    pipeline = TranscriptionPipeline.get_instance()

    # Upload audio and start transcription (async, returns task_id)
    result = pipeline.upload_audio(file_bytes, content_type, filename)

    # Poll task status
    info = pipeline.get_task_status(task_id)

    # Get the transcription output (raw text or formatted)
    output = pipeline.get_task_output(task_id)

    # Acknowledge receipt (triggers temp file cleanup)
    pipeline.acknowledge_task(task_id)

    # Cancel a running task
    info = pipeline.cancel_task(task_id)

    # List all tasks
    tasks = pipeline.list_tasks()

"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile

from backend.modules.output.formatter import OutputFormatter
from backend.modules.preprocessing.audio_processor import (
    AudioProcessor,
    AudioProcessingError,
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
class UploadTaskResult:
    """Result of a successful audio upload (transcription started)."""

    task_id: str
    status: str
    filename: str
    content_type: str
    size_bytes: int


@dataclass
class TaskInfo:
    """Immutable snapshot of a task's state."""

    task_id: str
    status: str
    progress: float
    started_at: datetime | None = None
    completed_at: datetime | None = None
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
# Internal task representation (mutable, thread-safe)
# ---------------------------------------------------------------------------


class _ConversionTask:
    """Mutable task state used internally."""

    def __init__(
        self,
        task_id: str,
        filename: str,
        content_type: str,
    ) -> None:
        self.task_id = task_id
        self.filename = filename
        self.content_type = content_type
        self.status = TaskStatus.PENDING
        self.progress: float = 0.0
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
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
        self.acknowledged: bool = False
        # Temporary files: [0] = audio tempfile, [1] = output tempfile
        self.temp_files: list[Path] = []

    def to_info(self) -> TaskInfo:
        """Convert to an immutable TaskInfo snapshot."""
        return TaskInfo(
            task_id=self.task_id,
            status=self.status.value,
            progress=self.progress,
            started_at=self.started_at,
            completed_at=self.completed_at,
            filename=self.filename,
            content_type=self.content_type,
            duration=self.duration,
            sample_rate=self.sample_rate,
            original_format=self.original_format,
            original_sample_rate=self.original_sample_rate,
            language=self.language,
            segment_count=self.segment_count,
            processing_time_ms=self.processing_time_ms,
            error=self.error,
            acknowledged=self.acknowledged,
        )


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class TranscriptionPipeline:
    """Facade for all transcription operations.

    Orchestrates the full pipeline:
        1. Save uploaded audio to a temporary file
        2. Preprocess audio (decode, resample, clean)
        3. Transcribe with Whisper
        4. Format output (optional, controlled by config)
        5. Save transcription to a temporary file
        6. Return transcription on request
        7. Clean up temp files after ACK

    This is a singleton — use ``get_instance()`` to obtain the shared
    instance.
    """

    _instance: TranscriptionPipeline | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._cfg = settings()
        self._use_formatter = self._cfg.output.use_formatter
        self._output_format = self._cfg.output.format
        self._temp_dir = Path(self._cfg.temp.dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-initialized modules
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
        filename: str,
    ) -> UploadTaskResult:
        """Save uploaded audio to a temp file and start transcription.

        The transcription runs asynchronously in a background thread.
        The caller receives a ``task_id`` immediately.

        Parameters
        ----------
        file_bytes:
            Raw audio file bytes.
        content_type:
            MIME type of the audio file.
        filename:
            Original filename (used for extension detection).

        Returns
        -------
        UploadTaskResult
            Metadata about the upload and the transcription task.
        """
        # Determine extension from content_type or filename
        ext = self._guess_extension(content_type, filename)

        # Create a real temporary file on disk (not in-memory) so the
        # audio processor can read it with ffmpeg/soundfile.
        tmp = NamedTemporaryFile(
            suffix=f".{ext}",
            dir=str(self._temp_dir),
            delete=False,
            prefix=f"audio_{uuid.uuid4().hex[:8]}_",
        )
        tmp.write(file_bytes)
        tmp.close()
        audio_path = Path(tmp.name)

        # Create task
        task_id = uuid.uuid4().hex
        task = _ConversionTask(
            task_id=task_id,
            filename=filename,
            content_type=content_type,
        )
        task.output_format = "txt"  # default, overridden by convert endpoint
        task.temp_files = [audio_path]
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now(timezone.utc)
        task.progress = 0.1
        self._set_task(task)

        # Start transcription in background
        threading.Thread(
            target=self.process_task,
            args=(task_id,),
            daemon=True,
        ).start()

        logger.info(
            "Uploaded %d bytes (%s) → task=%s (audio=%s)",
            len(file_bytes), filename, task_id, audio_path,
        )

        return UploadTaskResult(
            task_id=task_id,
            status=task.status.value,
            filename=filename,
            content_type=content_type,
            size_bytes=len(file_bytes),
        )

    def convert_task(
        self,
        task_id: str,
        output_format: str = "txt",
    ) -> str:
        """Set the output format for an existing task.

        Must be called after ``upload_audio()`` and before the transcription
        completes. If the task is already processing, the format is applied
        when the task reaches the formatting step.

        Parameters
        ----------
        task_id:
            The task ID returned by ``upload_audio()``.
        output_format:
            Output format (``json``, ``srt``, ``vtt``, ``txt``, ``csv``,
            ``tsv``).

        Returns
        -------
        str
            The ``task_id``.

        Raises
        ------
        KeyError:
            If the task doesn't exist.
        """
        task = self._get_task(task_id)
        task.output_format = output_format
        return task_id

    def process_task(self, task_id: str) -> None:
        """Execute the full conversion pipeline for a given task.

        This is the core orchestration method called in a background thread.
        """
        task = self._get_task(task_id)

        try:
            # --- Step 1: Read audio from temp file ---
            audio_path = task.temp_files[0]
            logger.info("[%s] Reading audio from %s", task_id, audio_path)
            with open(audio_path, "rb") as f:
                raw_bytes = f.read()
            task.progress = 0.2

            # --- Step 2: Preprocess audio ---
            logger.info("[%s] Preprocessing audio...", task_id)
            processed = self._processor.process(
                raw_bytes=raw_bytes,
                content_type=task.content_type,
                filename=task.filename,
            )
            task.progress = 0.4

            # --- Step 3: Transcribe ---
            logger.info("[%s] Transcribing (%.2fs)...", task_id, processed.duration)
            result = self._engine.transcribe(
                audio_data=processed.data,
                sr=processed.sample_rate,
                audio_duration=processed.duration,
            )
            task.progress = 0.7

            # --- Step 4: Build output ---
            fmt = task.output_format or self._output_format
            if self._use_formatter:
                output_content = self._formatter.format(
                    segments=result.segments,
                    fmt=fmt,
                )
                mime_type = self._formatter.get_mime_type(fmt)
            else:
                output_content = result.text
                mime_type = "text/plain"

            # Save output to a second temp file
            output_ext = ".txt" if not self._use_formatter else self._formatter.get_extension(fmt)
            output_tmp = NamedTemporaryFile(
                suffix=output_ext,
                dir=str(self._temp_dir),
                delete=False,
                mode="w",
                encoding="utf-8",
                prefix=f"output_{uuid.uuid4().hex[:8]}_",
            )
            output_tmp.write(output_content)
            output_tmp.close()
            output_path = Path(output_tmp.name)
            task.temp_files.append(output_path)

            task.progress = 0.9

            # --- Finalize task ---
            started_at = task.started_at or datetime.now(timezone.utc)
            elapsed_ms = (time.time() - started_at.timestamp()) * 1000

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
                output_path,
                elapsed_ms,
            )

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

    def _guess_extension(self, content_type: str, filename: str) -> str:
        """Guess file extension from content_type or filename."""
        # Try content_type first (e.g. "audio/mpeg" → "mp3")
        ext_map: dict[str, str] = {
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/flac": "flac",
            "audio/ogg": "ogg",
            "audio/mp3": "mp3",
            "audio/mpeg": "mp3",
            "audio/m4a": "m4a",
            "audio/mp4": "mp4",
            "audio/webm": "webm",
        }
        if content_type in ext_map:
            return ext_map[content_type]
        # Fallback to filename extension
        ext = Path(filename).suffix.lstrip(".").lower()
        if ext:
            return ext
        return "wav"  # default fallback

    def get_task_status(self, task_id: str) -> TaskInfo:
        """Get the current status of a task.

        Parameters
        ----------
        task_id:
            The task ID returned by ``upload_audio()``.

        Returns
        -------
        TaskInfo
            Current task state (metadata only, no transcription text).
        """
        task = self._get_task(task_id)
        return task.to_info()

    def get_task_output(self, task_id: str) -> str:
        """Get the transcription output for a completed task.

        The temporary files are NOT deleted here — they are deleted only
        after the caller explicitly acknowledges receipt via
        ``acknowledge_task()``.

        Parameters
        ----------
        task_id:
            The task ID returned by ``upload_audio()``.

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

        # Output is the second temp file
        if len(task.temp_files) < 2:
            raise ValueError("Output file not found.")

        output_path = task.temp_files[1]
        if not output_path.exists():
            raise ValueError("Output file not found.")

        return output_path.read_text(encoding="utf-8")

    def acknowledge_task(self, task_id: str) -> None:
        """Acknowledge receipt of the transcription output.

        After acknowledgment, all temporary files associated with the task
        are deleted.

        Parameters
        ----------
        task_id:
            The task ID returned by ``upload_audio()``.

        Raises
        ------
        KeyError:
            If the task doesn't exist.
        ValueError:
            If the task hasn't completed yet or has already been acknowledged.
        """
        task = self._get_task(task_id)

        if task.acknowledged:
            raise ValueError("Task has already been acknowledged.")

        if task.status != TaskStatus.COMPLETED:
            raise ValueError(
                f"Task is not completed (status: {task.status.value})."
            )

        # Delete all temporary files
        for path in task.temp_files:
            try:
                if path.exists():
                    path.unlink()
                    logger.debug("Deleted temp file: %s", path)
            except OSError as exc:
                logger.warning("Failed to delete temp file %s: %s", path, exc)

        task.temp_files.clear()
        task.acknowledged = True
        logger.info("Task %s acknowledged and temp files cleaned up", task_id)

    def cancel_task(self, task_id: str) -> TaskInfo:
        """Cancel a running task and clean up temp files.

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

        # Clean up temp files
        self._cleanup_temp_files(task)

        logger.info("Cancelled task %s", task_id)
        return task.to_info()

    def _cleanup_temp_files(self, task: _ConversionTask) -> None:
        """Delete all temporary files associated with a task."""
        for path in task.temp_files:
            try:
                if path.exists():
                    path.unlink()
                    logger.debug("Deleted temp file: %s", path)
            except OSError as exc:
                logger.warning("Failed to delete temp file %s: %s", path, exc)
        task.temp_files.clear()

    def list_tasks(self) -> list[TaskInfo]:
        """List all conversion tasks.

        Returns
        -------
        list[TaskInfo]
            All tasks with their current state.
        """
        with self._tasks_lock:
            return [task.to_info() for task in self._tasks.values()]
