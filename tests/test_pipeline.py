"""Tests for backend.transcription_pipeline — full orchestration."""

from __future__ import annotations

import io
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from backend.transcription_pipeline import (
    TaskInfo,
    TaskStatus,
    TranscriptionPipeline,
    UploadResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_wav_bytes(duration_sec=1.0, sr=16000):
    """Generate test WAV bytes (mono, 16kHz, 440 Hz sine)."""
    buf = io.BytesIO()
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    data = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sf.write(buf, data, sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


def _wav_bytes_utf8():
    """Generate WAV bytes that decode as valid UTF-8 text."""
    # WAV files won't decode as UTF-8, so we use plain text bytes
    return b"Transcription output text"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipeline_instance():
    """Return a TranscriptionPipeline instance (resets singleton)."""
    from backend import transcription_pipeline
    transcription_pipeline.TranscriptionPipeline._instance = None
    return TranscriptionPipeline.get_instance()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_singleton_same_instance(self):
        from backend import transcription_pipeline
        transcription_pipeline.TranscriptionPipeline._instance = None
        try:
            p1 = TranscriptionPipeline.get_instance()
            p2 = TranscriptionPipeline.get_instance()
            assert p1 is p2
        finally:
            transcription_pipeline.TranscriptionPipeline._instance = None


# ---------------------------------------------------------------------------
# upload_audio
# ---------------------------------------------------------------------------


class TestUploadAudio:
    def test_upload_audio(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test uploading audio to S3/MinIO."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            result = pipeline_instance.upload_audio(
                file_bytes=_create_test_wav_bytes(),
                content_type="audio/wav",
                bucket="test-bucket",
                destination="audio/uploads/test.wav",
            )

        assert isinstance(result, UploadResult)
        assert result.key == "audio/uploads/test.wav"
        assert result.bucket == "test-bucket"
        assert result.size_bytes > 0
        assert result.content_type == "audio/wav"
        assert result.uploaded_at is not None

    def test_upload_audio_auto_key(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test auto-generated key when destination not provided."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            result = pipeline_instance.upload_audio(
                file_bytes=_create_test_wav_bytes(),
                content_type="audio/wav",
            )

        assert result.key.startswith("audio/uploads/")
        assert result.bucket == "sbobinator"

    def test_upload_audio_auto_key_with_ext(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test auto-generated key with content_type that has an extension-like suffix."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            result = pipeline_instance.upload_audio(
                file_bytes=_create_test_wav_bytes(),
                content_type="audio/x-wav",
            )

        assert result.key.startswith("audio/uploads/")

    def test_upload_audio_custom_bucket(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test upload to custom bucket."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            result = pipeline_instance.upload_audio(
                file_bytes=_create_test_wav_bytes(),
                content_type="audio/flac",
                bucket="custom-bucket",
            )

        assert result.bucket == "custom-bucket"


# ---------------------------------------------------------------------------
# start_conversion / task lifecycle
# ---------------------------------------------------------------------------


class TestTaskLifecycle:
    def test_start_conversion_creates_task(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test that start_conversion creates a task and returns a task_id."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_s3_client.get_object.return_value = {
                "Body": MagicMock(read=lambda: _create_test_wav_bytes()),
                "ContentType": "audio/wav",
            }
            task_id = pipeline_instance.start_conversion(
                key="audio/test.wav",
                bucket="test-bucket",
                output_format="txt",
            )

        assert isinstance(task_id, str)
        assert len(task_id) == 32  # hex UUID

    def test_get_task_status_pending(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test task status is PENDING after creation."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_s3_client.get_object.return_value = {
                "Body": MagicMock(read=lambda: _create_test_wav_bytes()),
                "ContentType": "audio/wav",
            }
            task_id = pipeline_instance.start_conversion(
                key="audio/test.wav", bucket="test-bucket"
            )

        info = pipeline_instance.get_task_status(task_id)
        assert isinstance(info, TaskInfo)
        assert info.task_id == task_id
        assert info.status == TaskStatus.PENDING.value
        assert info.progress == 0.0
        assert info.key == "audio/test.wav"
        assert info.bucket == "test-bucket"

    def test_get_task_not_found(self, pipeline_instance):
        """Test KeyError for non-existent task."""
        with pytest.raises(KeyError, match="Task not found"):
            pipeline_instance.get_task_status("nonexistent")

    def test_list_tasks_empty(self, pipeline_instance):
        """Test list_tasks returns empty list initially."""
        tasks = pipeline_instance.list_tasks()
        assert tasks == []

    def test_list_tasks_after_creation(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test list_tasks returns tasks after creation."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_s3_client.get_object.return_value = {
                "Body": MagicMock(read=lambda: _create_test_wav_bytes()),
                "ContentType": "audio/wav",
            }
            pipeline_instance.start_conversion(key="audio/test.wav", bucket="test-bucket")
            pipeline_instance.start_conversion(key="audio/test2.wav", bucket="test-bucket")

        tasks = pipeline_instance.list_tasks()
        assert len(tasks) == 2

    def test_cancel_task(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test cancelling a pending task."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_s3_client.get_object.return_value = {
                "Body": MagicMock(read=lambda: _create_test_wav_bytes()),
                "ContentType": "audio/wav",
            }
            task_id = pipeline_instance.start_conversion(
                key="audio/test.wav", bucket="test-bucket"
            )

        info = pipeline_instance.cancel_task(task_id)
        assert info.status == TaskStatus.CANCELLED.value
        assert info.completed_at is not None

    def test_cancel_completed_task(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test cancelling an already completed task returns same status."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_s3_client.get_object.return_value = {
                "Body": MagicMock(read=lambda: _create_test_wav_bytes()),
                "ContentType": "audio/wav",
            }
            task_id = pipeline_instance.start_conversion(
                key="audio/test.wav", bucket="test-bucket"
            )

        # Manually set to completed
        task = pipeline_instance._get_task(task_id)
        task.status = TaskStatus.COMPLETED

        info = pipeline_instance.cancel_task(task_id)
        assert info.status == TaskStatus.COMPLETED.value  # unchanged

    def test_cancel_nonexistent_task(self, pipeline_instance):
        """Test cancelling a non-existent task raises KeyError."""
        with pytest.raises(KeyError, match="Task not found"):
            pipeline_instance.cancel_task("nonexistent")


# ---------------------------------------------------------------------------
# process_task — happy path (mocked Whisper + S3)
# ---------------------------------------------------------------------------


class TestProcessTask:
    def test_process_task_completed(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test that process_task transitions from PENDING → COMPLETED."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_s3_client.get_object.return_value = {
                "Body": MagicMock(read=lambda: _create_test_wav_bytes()),
                "ContentType": "audio/wav",
            }
            task_id = pipeline_instance.start_conversion(
                key="audio/test.wav", bucket="test-bucket"
            )

        # Patch the module-level imports used by process_task
        with patch("backend.transcription_pipeline.AudioFetcher") as MockFetcher, \
             patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine, \
             patch("backend.transcription_pipeline.OutputFormatter") as MockFormatter:

            # Setup fetcher mock
            fetcher_inst = MockFetcher.return_value
            fetcher_inst.fetch.return_value = MagicMock(
                key="audio/test.wav",
                data=_create_test_wav_bytes(),
                content_type="audio/wav",
                size_bytes=32004,
            )

            # Setup processor mock
            processor_inst = MockProcessor.return_value
            processor_inst.process.return_value = MagicMock(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                original_format="wav",
                original_sample_rate=16000,
            )

            # Setup engine mock
            seg1 = MagicMock(start=0.0, end=1.0, text="Test segment")
            seg2 = MagicMock(start=1.5, end=2.5, text="Second segment")
            engine_inst = MockEngine.return_value
            engine_inst.transcribe.return_value = MagicMock(
                segments=[seg1, seg2],
                text="Test segment Second segment",
                language="it",
            )

            # Setup formatter mock
            formatter_inst = MockFormatter.return_value
            formatter_inst.format.return_value = "Test segment Second segment"
            formatter_inst.build_output_key.return_value = "audio/test.txt"
            formatter_inst.get_mime_type.return_value = "text/plain"

            pipeline_instance.process_task(task_id)

        # Verify task completed
        info = pipeline_instance.get_task_status(task_id)
        assert info.status == TaskStatus.COMPLETED.value
        assert info.progress == 1.0
        assert info.started_at is not None
        assert info.completed_at is not None
        assert info.output_key == "audio/test.txt"
        assert info.duration == 1.0
        assert info.sample_rate == 16000
        assert info.original_format == "wav"
        assert info.original_sample_rate == 16000
        assert info.language == "it"
        assert info.segment_count == 2
        assert info.processing_time_ms is not None

    def test_process_task_with_formatter(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test process_task with formatter enabled."""
        from backend import transcription_pipeline
        transcription_pipeline.TranscriptionPipeline._instance = None

        # Patch settings BEFORE creating the instance so _use_formatter is True
        with patch("utils.config.settings") as mock_settings:
            mock_cfg = MagicMock()
            mock_cfg.output.use_formatter = True
            mock_cfg.output.format = "srt"
            mock_cfg.storage.bucket = "test-bucket"
            mock_settings.return_value = mock_cfg

            # Re-create pipeline instance with patched settings
            formatter_pipeline = TranscriptionPipeline.get_instance()
            formatter_pipeline._use_formatter = True  # Force formatter mode
            formatter_pipeline._output_format = "srt"

            with patch("boto3.Session", return_value=mock_boto3_session):
                task_id = formatter_pipeline.start_conversion(
                    key="audio/test.wav", bucket="test-bucket", output_format="srt"
                )

        with patch("backend.transcription_pipeline.AudioFetcher") as MockFetcher, \
             patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine, \
             patch("backend.transcription_pipeline.OutputFormatter") as MockFormatter:

            fetcher_inst = MockFetcher.return_value
            fetcher_inst.fetch.return_value = MagicMock(
                key="audio/test.wav",
                data=_create_test_wav_bytes(),
                content_type="audio/wav",
                size_bytes=32004,
            )
            processor_inst = MockProcessor.return_value
            processor_inst.process.return_value = MagicMock(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                original_format="wav",
                original_sample_rate=16000,
            )
            seg1 = MagicMock(start=0.0, end=1.0, text="Test")
            engine_inst = MockEngine.return_value
            engine_inst.transcribe.return_value = MagicMock(
                segments=[seg1], text="Test", language="it"
            )
            formatter_inst = MockFormatter.return_value
            formatter_inst.format.return_value = "1\n00:00:00,000 --> 00:00:01,000\nTest"
            formatter_inst.build_output_key.return_value = "audio/test.srt"
            formatter_inst.get_mime_type.return_value = "text/srt"

            formatter_pipeline.process_task(task_id)

        info = formatter_pipeline.get_task_status(task_id)
        assert info.status == TaskStatus.COMPLETED.value
        # The formatter mock's build_output_key should have been called
        formatter_inst.build_output_key.assert_called_once()
        assert info.output_key == "audio/test.srt"

    def test_process_task_raw_text_mode(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test process_task in raw text mode (formatter disabled)."""
        from backend import transcription_pipeline
        transcription_pipeline.TranscriptionPipeline._instance = None

        with patch("utils.config.settings") as mock_settings:
            mock_cfg = MagicMock()
            mock_cfg.output.use_formatter = False
            mock_cfg.output.format = "json"
            mock_cfg.storage.bucket = "test-bucket"
            mock_settings.return_value = mock_cfg

            with patch("boto3.Session", return_value=mock_boto3_session):
                task_id = pipeline_instance.start_conversion(
                    key="audio/test.wav", bucket="test-bucket"
                )

        with patch("backend.transcription_pipeline.AudioFetcher") as MockFetcher, \
             patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine:

            fetcher_inst = MockFetcher.return_value
            fetcher_inst.fetch.return_value = MagicMock(
                key="audio/test.wav",
                data=_create_test_wav_bytes(),
                content_type="audio/wav",
                size_bytes=32004,
            )
            processor_inst = MockProcessor.return_value
            processor_inst.process.return_value = MagicMock(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                original_format="wav",
                original_sample_rate=16000,
            )
            seg1 = MagicMock(start=0.0, end=1.0, text="Raw text")
            engine_inst = MockEngine.return_value
            engine_inst.transcribe.return_value = MagicMock(
                segments=[seg1], text="Raw text", language="it"
            )

            pipeline_instance.process_task(task_id)

        info = pipeline_instance.get_task_status(task_id)
        assert info.status == TaskStatus.COMPLETED.value
        assert info.output_key == "audio/test.txt"


# ---------------------------------------------------------------------------
# process_task — error paths
# ---------------------------------------------------------------------------


class TestProcessTaskErrors:
    def test_fetch_error(self, pipeline_instance):
        """Test that FetchError marks task as FAILED."""
        from backend.modules.storage.fetcher import FetchError

        # Create task directly (skip start_conversion which calls real S3)
        task_id = pipeline_instance.start_conversion.__wrapped__(
            pipeline_instance, "audio/test.wav", "test-bucket"
        ) if hasattr(pipeline_instance.start_conversion, '__wrapped__') else None

        if task_id is None:
            # Manually create the task
            import uuid
            task_id = uuid.uuid4().hex
            from backend.transcription_pipeline import _ConversionTask
            task = _ConversionTask(
                task_id=task_id,
                key="audio/test.wav",
                bucket="test-bucket",
            )
            pipeline_instance._set_task(task)

        with patch("backend.transcription_pipeline.AudioFetcher") as MockFetcher:
            MockFetcher.return_value.fetch.side_effect = FetchError("File not found")
            pipeline_instance.process_task(task_id)

        info = pipeline_instance.get_task_status(task_id)
        assert info.status == TaskStatus.FAILED.value
        assert "Fetch error" in info.error

    def test_audio_processing_error(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test that AudioProcessingError marks task as FAILED."""
        from backend.modules.preprocessing.audio_processor import AudioProcessingError

        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_s3_client.get_object.return_value = {
                "Body": MagicMock(read=lambda: _create_test_wav_bytes()),
                "ContentType": "audio/wav",
            }
            task_id = pipeline_instance.start_conversion(
                key="audio/test.wav", bucket="test-bucket"
            )

        with patch("backend.transcription_pipeline.AudioFetcher") as MockFetcher, \
             patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor:

            MockFetcher.return_value.fetch.return_value = MagicMock(
                key="audio/test.wav",
                data=_create_test_wav_bytes(),
                content_type="audio/wav",
                size_bytes=32004,
            )
            MockProcessor.return_value.process.side_effect = AudioProcessingError("Decode failed")
            pipeline_instance.process_task(task_id)

        info = pipeline_instance.get_task_status(task_id)
        assert info.status == TaskStatus.FAILED.value
        assert "Processing error" in info.error

    def test_transcription_error(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test that TranscriptionError marks task as FAILED."""
        from backend.modules.transcription.whisper_engine import TranscriptionError

        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_s3_client.get_object.return_value = {
                "Body": MagicMock(read=lambda: _create_test_wav_bytes()),
                "ContentType": "audio/wav",
            }
            task_id = pipeline_instance.start_conversion(
                key="audio/test.wav", bucket="test-bucket"
            )

        with patch("backend.transcription_pipeline.AudioFetcher") as MockFetcher, \
             patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine:

            MockFetcher.return_value.fetch.return_value = MagicMock(
                key="audio/test.wav",
                data=_create_test_wav_bytes(),
                content_type="audio/wav",
                size_bytes=32004,
            )
            MockProcessor.return_value.process.return_value = MagicMock(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                original_format="wav",
                original_sample_rate=16000,
            )
            MockEngine.return_value.transcribe.side_effect = TranscriptionError("Whisper failed")
            pipeline_instance.process_task(task_id)

        info = pipeline_instance.get_task_status(task_id)
        assert info.status == TaskStatus.FAILED.value
        assert "Transcription error" in info.error

    def test_unexpected_error(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test that unexpected exceptions mark task as FAILED."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_s3_client.get_object.return_value = {
                "Body": MagicMock(read=lambda: _create_test_wav_bytes()),
                "ContentType": "audio/wav",
            }
            task_id = pipeline_instance.start_conversion(
                key="audio/test.wav", bucket="test-bucket"
            )

        with patch("backend.transcription_pipeline.AudioFetcher") as MockFetcher, \
             patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine:

            MockFetcher.return_value.fetch.return_value = MagicMock(
                key="audio/test.wav",
                data=_create_test_wav_bytes(),
                content_type="audio/wav",
                size_bytes=32004,
            )
            MockProcessor.return_value.process.return_value = MagicMock(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                original_format="wav",
                original_sample_rate=16000,
            )
            MockEngine.return_value.transcribe.side_effect = RuntimeError("Something weird happened")
            pipeline_instance.process_task(task_id)

        info = pipeline_instance.get_task_status(task_id)
        assert info.status == TaskStatus.FAILED.value
        assert "Unexpected error" in info.error

    def test_task_not_found_get_output(self, pipeline_instance):
        """Test get_task_output raises KeyError for non-existent task."""
        with pytest.raises(KeyError, match="Task not found"):
            pipeline_instance.get_task_output("nonexistent")

    def test_task_not_completed_get_output(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test get_task_output raises ValueError for non-completed task."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_s3_client.get_object.return_value = {
                "Body": MagicMock(read=lambda: _create_test_wav_bytes()),
                "ContentType": "audio/wav",
            }
            task_id = pipeline_instance.start_conversion(
                key="audio/test.wav", bucket="test-bucket"
            )

        with pytest.raises(ValueError, match="not completed"):
            pipeline_instance.get_task_output(task_id)


# ---------------------------------------------------------------------------
# get_task_output
# ---------------------------------------------------------------------------


class TestGetTaskOutput:
    def test_get_output_success(self, pipeline_instance, mock_s3_client, mock_boto3_session):
        """Test successful output retrieval."""
        from backend.modules.storage.fetcher import AudioFetchResult

        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_s3_client.get_object.return_value = {
                "Body": MagicMock(read=lambda: _create_test_wav_bytes()),
                "ContentType": "audio/wav",
            }
            task_id = pipeline_instance.start_conversion(
                key="audio/test.wav", bucket="test-bucket"
            )

        # Manually set task to completed with output_key
        task = pipeline_instance._get_task(task_id)
        task.status = TaskStatus.COMPLETED
        task.output_key = "audio/test.txt"
        task.bucket = "test-bucket"

        # Reset cached fetcher so the mock is used
        pipeline_instance._audio_fetcher = None

        # Mock fetch for the output file - use plain text bytes (valid UTF-8)
        with patch("backend.transcription_pipeline.AudioFetcher") as MockFetcher:
            MockFetcher.return_value.fetch.return_value = AudioFetchResult(
                key="audio/test.txt",
                data=b"Transcription output text",  # Plain ASCII text, valid UTF-8
                content_type="text/plain",
                size_bytes=26,
            )
            output = pipeline_instance.get_task_output(task_id)

        assert output == "Transcription output text"
        # Verify the fetcher was called with the output key
        MockFetcher.return_value.fetch.assert_called_once_with(
            key="audio/test.txt", bucket="test-bucket"
        )
