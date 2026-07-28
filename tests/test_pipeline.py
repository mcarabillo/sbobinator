"""Tests for backend.transcription_pipeline — full orchestration."""

from __future__ import annotations

import io
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from backend.transcription_pipeline import (
    TaskInfo,
    TaskStatus,
    TranscriptionPipeline,
    UploadTaskResult,
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
    def test_upload_audio(self, pipeline_instance):
        """Test uploading audio creates temp file and returns task_id."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )

        assert isinstance(result, UploadTaskResult)
        assert result.task_id is not None
        assert result.status == "processing"
        assert result.filename == "test.wav"
        assert result.content_type == "audio/wav"
        assert result.size_bytes == len(wav_bytes)

    def test_upload_audio_creates_temp_file(self, pipeline_instance):
        """Test that upload creates a temporary audio file on disk."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )

        task = pipeline_instance._get_task(result.task_id)
        assert len(task.temp_files) == 1
        assert task.temp_files[0].exists()

    def test_upload_audio_auto_extension(self, pipeline_instance):
        """Test that extension is guessed from content_type."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/mpeg",
            filename="podcast",  # no extension
        )

        task = pipeline_instance._get_task(result.task_id)
        assert task.temp_files[0].suffix == ".mp3"

    def test_upload_audio_from_filename_extension(self, pipeline_instance):
        """Test that extension is guessed from filename when content_type is generic."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="application/octet-stream",
            filename="recording.flac",
        )

        task = pipeline_instance._get_task(result.task_id)
        assert task.temp_files[0].suffix == ".flac"

    def test_upload_audio_starts_processing(self, pipeline_instance):
        """Test that upload starts processing immediately."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )

        info = pipeline_instance.get_task_status(result.task_id)
        assert info.status == TaskStatus.PROCESSING.value
        assert info.started_at is not None

    def test_upload_audio_no_s3_dependency(self, pipeline_instance):
        """Test that upload does NOT depend on S3/MinIO."""
        wav_bytes = _create_test_wav_bytes()

        # This should NOT raise even without S3 credentials
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )

        assert result.task_id is not None


# ---------------------------------------------------------------------------
# convert_task (format setting)
# ---------------------------------------------------------------------------


class TestConvertTask:
    def test_convert_task_sets_format(self, pipeline_instance):
        """Test setting output format on an existing task."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )

        pipeline_instance.convert_task(
            task_id=result.task_id,
            output_format="json",
        )

        task = pipeline_instance._get_task(result.task_id)
        assert task.output_format == "json"

    def test_convert_task_not_found(self, pipeline_instance):
        """Test setting format on non-existent task raises KeyError."""
        with pytest.raises(KeyError, match="Task not found"):
            pipeline_instance.convert_task(
                task_id="nonexistent",
                output_format="srt",
            )


# ---------------------------------------------------------------------------
# Task lifecycle
# ---------------------------------------------------------------------------


class TestTaskLifecycle:
    def test_get_task_status(self, pipeline_instance):
        """Test getting task status."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )

        info = pipeline_instance.get_task_status(result.task_id)
        assert isinstance(info, TaskInfo)
        assert info.task_id == result.task_id
        assert info.status == TaskStatus.PROCESSING.value
        assert info.progress >= 0.1  # progress may have advanced due to background thread
        assert info.filename == "test.wav"

    def test_get_task_not_found(self, pipeline_instance):
        """Test KeyError for non-existent task."""
        with pytest.raises(KeyError, match="Task not found"):
            pipeline_instance.get_task_status("nonexistent")

    def test_list_tasks_empty(self, pipeline_instance):
        """Test list_tasks returns empty list initially."""
        tasks = pipeline_instance.list_tasks()
        assert tasks == []

    def test_list_tasks_after_creation(self, pipeline_instance):
        """Test list_tasks returns tasks after creation."""
        wav_bytes = _create_test_wav_bytes()
        pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test1.wav",
        )
        pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test2.wav",
        )

        tasks = pipeline_instance.list_tasks()
        assert len(tasks) == 2

    def test_task_info_has_no_s3_fields(self, pipeline_instance):
        """Test that TaskInfo does not contain S3 fields."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )

        info = pipeline_instance.get_task_status(result.task_id)
        assert not hasattr(info, "key")
        assert not hasattr(info, "bucket")
        assert not hasattr(info, "output_key")


# ---------------------------------------------------------------------------
# process_task — happy path (mocked Whisper)
# ---------------------------------------------------------------------------


class TestProcessTask:
    def test_process_task_completed(self, pipeline_instance):
        """Test that process_task transitions PROCESSING → COMPLETED."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )
        task_id = result.task_id

        with patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine:

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

            pipeline_instance.process_task(task_id)

        # Verify task completed
        info = pipeline_instance.get_task_status(task_id)
        assert info.status == TaskStatus.COMPLETED.value
        assert info.progress == 1.0
        assert info.started_at is not None
        assert info.completed_at is not None
        assert info.duration == 1.0
        assert info.sample_rate == 16000
        assert info.original_format == "wav"
        assert info.original_sample_rate == 16000
        assert info.language == "it"
        assert info.segment_count == 2
        assert info.processing_time_ms is not None

    def test_process_task_creates_output_temp_file(self, pipeline_instance):
        """Test that process_task creates a second temp file for output."""
        wav_bytes = _create_test_wav_bytes()

        # Patch threading to prevent background thread from starting
        with patch("threading.Thread") as MockThread:
            mock_thread = MagicMock()
            MockThread.return_value = mock_thread

            result = pipeline_instance.upload_audio(
                file_bytes=wav_bytes,
                content_type="audio/wav",
                filename="test.wav",
            )
            task_id = result.task_id

        with patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine:

            processor_inst = MockProcessor.return_value
            processor_inst.process.return_value = MagicMock(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                original_format="wav",
                original_sample_rate=16000,
            )

            seg = MagicMock(start=0.0, end=1.0, text="Test")
            engine_inst = MockEngine.return_value
            engine_inst.transcribe.return_value = MagicMock(
                segments=[seg], text="Test", language="it",
            )

            pipeline_instance.process_task(task_id)

        task = pipeline_instance._get_task(task_id)
        assert len(task.temp_files) == 2  # audio + output
        assert task.temp_files[1].exists()

    def test_process_task_with_formatter(self, pipeline_instance):
        """Test process_task with formatter enabled."""
        from backend import transcription_pipeline
        transcription_pipeline.TranscriptionPipeline._instance = None

        with patch("utils.config.settings") as mock_settings:
            mock_cfg = MagicMock()
            mock_cfg.output.use_formatter = True
            mock_cfg.output.format = "srt"
            mock_cfg.temp.dir = "/tmp/sbobinator_test"
            mock_settings.return_value = mock_cfg

            formatter_pipeline = TranscriptionPipeline.get_instance()
            formatter_pipeline._use_formatter = True
            formatter_pipeline._output_format = "srt"

            wav_bytes = _create_test_wav_bytes()
            result = formatter_pipeline.upload_audio(
                file_bytes=wav_bytes,
                content_type="audio/wav",
                filename="test.wav",
            )
            formatter_pipeline.convert_task(result.task_id, "srt")
            task_id = result.task_id

        with patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine, \
             patch("backend.transcription_pipeline.OutputFormatter") as MockFormatter:

            processor_inst = MockProcessor.return_value
            processor_inst.process.return_value = MagicMock(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                original_format="wav",
                original_sample_rate=16000,
            )

            seg = MagicMock(start=0.0, end=1.0, text="Test")
            engine_inst = MockEngine.return_value
            engine_inst.transcribe.return_value = MagicMock(
                segments=[seg], text="Test", language="it",
            )

            formatter_inst = MockFormatter.return_value
            formatter_inst.format.return_value = "1\n00:00:00,000 --> 00:00:01,000\nTest"
            formatter_inst.get_extension.return_value = ".srt"
            formatter_inst.get_mime_type.return_value = "text/srt"

            formatter_pipeline.process_task(task_id)

        info = formatter_pipeline.get_task_status(task_id)
        assert info.status == TaskStatus.COMPLETED.value

    def test_process_task_raw_text_mode(self, pipeline_instance):
        """Test process_task in raw text mode (formatter disabled)."""
        from backend import transcription_pipeline
        transcription_pipeline.TranscriptionPipeline._instance = None

        with patch("utils.config.settings") as mock_settings:
            mock_cfg = MagicMock()
            mock_cfg.output.use_formatter = False
            mock_cfg.output.format = "json"
            mock_cfg.temp.dir = "/tmp/sbobinator_test"
            mock_settings.return_value = mock_cfg

            pipeline_instance = TranscriptionPipeline.get_instance()
            pipeline_instance._use_formatter = False
            pipeline_instance._output_format = "json"

            wav_bytes = _create_test_wav_bytes()
            result = pipeline_instance.upload_audio(
                file_bytes=wav_bytes,
                content_type="audio/wav",
                filename="test.wav",
            )
            task_id = result.task_id

        with patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine:

            processor_inst = MockProcessor.return_value
            processor_inst.process.return_value = MagicMock(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                original_format="wav",
                original_sample_rate=16000,
            )

            seg = MagicMock(start=0.0, end=1.0, text="Raw text")
            engine_inst = MockEngine.return_value
            engine_inst.transcribe.return_value = MagicMock(
                segments=[seg], text="Raw text", language="it",
            )

            pipeline_instance.process_task(task_id)

        info = pipeline_instance.get_task_status(task_id)
        assert info.status == TaskStatus.COMPLETED.value


# ---------------------------------------------------------------------------
# process_task — error paths
# ---------------------------------------------------------------------------


class TestProcessTaskErrors:
    def test_audio_processing_error(self, pipeline_instance):
        """Test that AudioProcessingError marks task as FAILED."""
        from backend.modules.preprocessing.audio_processor import AudioProcessingError

        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )
        task_id = result.task_id

        with patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor:
            MockProcessor.return_value.process.side_effect = AudioProcessingError("Decode failed")
            pipeline_instance.process_task(task_id)

        info = pipeline_instance.get_task_status(task_id)
        assert info.status == TaskStatus.FAILED.value
        assert "Processing error" in info.error

    def test_transcription_error(self, pipeline_instance):
        """Test that TranscriptionError marks task as FAILED."""
        from backend.modules.transcription.whisper_engine import TranscriptionError

        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )
        task_id = result.task_id

        with patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine:

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

    def test_unexpected_error(self, pipeline_instance):
        """Test that unexpected exceptions mark task as FAILED."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )
        task_id = result.task_id

        with patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine:

            MockProcessor.return_value.process.return_value = MagicMock(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                original_format="wav",
                original_sample_rate=16000,
            )
            MockEngine.return_value.transcribe.side_effect = RuntimeError("Something weird")
            pipeline_instance.process_task(task_id)

        info = pipeline_instance.get_task_status(task_id)
        assert info.status == TaskStatus.FAILED.value
        assert "Unexpected error" in info.error


# ---------------------------------------------------------------------------
# get_task_output
# ---------------------------------------------------------------------------


class TestGetTaskOutput:
    def test_get_output_success(self, pipeline_instance):
        """Test successful output retrieval from temp file."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )
        task_id = result.task_id

        with patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine:

            processor_inst = MockProcessor.return_value
            processor_inst.process.return_value = MagicMock(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                original_format="wav",
                original_sample_rate=16000,
            )

            seg = MagicMock(start=0.0, end=1.0, text="Test")
            engine_inst = MockEngine.return_value
            engine_inst.transcribe.return_value = MagicMock(
                segments=[seg], text="Test", language="it",
            )

            pipeline_instance.process_task(task_id)

        output = pipeline_instance.get_task_output(task_id)
        assert output == "Test"

    def test_get_output_not_completed(self, pipeline_instance):
        """Test get_task_output raises ValueError for non-completed task."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )

        with pytest.raises(ValueError, match="not completed"):
            pipeline_instance.get_task_output(result.task_id)

    def test_get_output_not_found(self, pipeline_instance):
        """Test get_task_output raises KeyError for non-existent task."""
        with pytest.raises(KeyError, match="Task not found"):
            pipeline_instance.get_task_output("nonexistent")


# ---------------------------------------------------------------------------
# acknowledge_task
# ---------------------------------------------------------------------------


class TestAcknowledgeTask:
    def test_acknowledge_success(self, pipeline_instance):
        """Test successful acknowledgment cleans up temp files."""
        wav_bytes = _create_test_wav_bytes()

        # Patch threading to prevent background thread from starting
        with patch("threading.Thread") as MockThread:
            mock_thread = MagicMock()
            MockThread.return_value = mock_thread

            result = pipeline_instance.upload_audio(
                file_bytes=wav_bytes,
                content_type="audio/wav",
                filename="test.wav",
            )
            task_id = result.task_id

        with patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine:

            processor_inst = MockProcessor.return_value
            processor_inst.process.return_value = MagicMock(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                original_format="wav",
                original_sample_rate=16000,
            )

            seg = MagicMock(start=0.0, end=1.0, text="Test")
            engine_inst = MockEngine.return_value
            engine_inst.transcribe.return_value = MagicMock(
                segments=[seg], text="Test", language="it",
            )

            pipeline_instance.process_task(task_id)

        # Verify temp files exist before ACK (1 audio + 1 output)
        task = pipeline_instance._get_task(task_id)
        assert len(task.temp_files) == 2
        assert all(f.exists() for f in task.temp_files)

        # Acknowledge
        pipeline_instance.acknowledge_task(task_id)

        # Verify temp files are deleted
        assert len(task.temp_files) == 0
        assert task.acknowledged is True

    def test_acknowledge_not_completed(self, pipeline_instance):
        """Test ACK for non-completed task raises ValueError."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )

        with pytest.raises(ValueError, match="not completed"):
            pipeline_instance.acknowledge_task(result.task_id)

    def test_acknowledge_already_acked(self, pipeline_instance):
        """Test double ACK raises ValueError."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )
        task_id = result.task_id

        with patch("backend.transcription_pipeline.AudioProcessor") as MockProcessor, \
             patch("backend.transcription_pipeline.TranscriptionEngine") as MockEngine:

            processor_inst = MockProcessor.return_value
            processor_inst.process.return_value = MagicMock(
                data=np.random.randn(16000).astype(np.float32),
                sample_rate=16000,
                duration=1.0,
                original_format="wav",
                original_sample_rate=16000,
            )

            seg = MagicMock(start=0.0, end=1.0, text="Test")
            engine_inst = MockEngine.return_value
            engine_inst.transcribe.return_value = MagicMock(
                segments=[seg], text="Test", language="it",
            )

            pipeline_instance.process_task(task_id)

        pipeline_instance.acknowledge_task(task_id)

        with pytest.raises(ValueError, match="already been acknowledged"):
            pipeline_instance.acknowledge_task(task_id)

    def test_acknowledge_not_found(self, pipeline_instance):
        """Test ACK for non-existent task raises KeyError."""
        with pytest.raises(KeyError, match="Task not found"):
            pipeline_instance.acknowledge_task("nonexistent")


# ---------------------------------------------------------------------------
# cancel_task
# ---------------------------------------------------------------------------


class TestCancelTask:
    def test_cancel_success(self, pipeline_instance):
        """Test cancelling a task."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )
        task_id = result.task_id

        info = pipeline_instance.cancel_task(task_id)
        assert info.status == TaskStatus.CANCELLED.value
        assert info.completed_at is not None

    def test_cancel_completed_task(self, pipeline_instance):
        """Test cancelling an already completed task returns same status."""
        wav_bytes = _create_test_wav_bytes()
        result = pipeline_instance.upload_audio(
            file_bytes=wav_bytes,
            content_type="audio/wav",
            filename="test.wav",
        )
        task_id = result.task_id

        # Manually set to completed
        task = pipeline_instance._get_task(task_id)
        task.status = TaskStatus.COMPLETED

        info = pipeline_instance.cancel_task(task_id)
        assert info.status == TaskStatus.COMPLETED.value

    def test_cancel_nonexistent_task(self, pipeline_instance):
        """Test cancelling a non-existent task raises KeyError."""
        with pytest.raises(KeyError, match="Task not found"):
            pipeline_instance.cancel_task("nonexistent")


# ---------------------------------------------------------------------------
# _guess_extension
# ---------------------------------------------------------------------------


class TestGuessExtension:
    def test_from_content_type(self, pipeline_instance):
        """Test extension guessed from content_type."""
        assert pipeline_instance._guess_extension("audio/wav", "anything") == "wav"
        assert pipeline_instance._guess_extension("audio/mpeg", "anything") == "mp3"
        assert pipeline_instance._guess_extension("audio/flac", "anything") == "flac"
        assert pipeline_instance._guess_extension("audio/ogg", "anything") == "ogg"
        assert pipeline_instance._guess_extension("audio/m4a", "anything") == "m4a"

    def test_from_filename(self, pipeline_instance):
        """Test extension guessed from filename when content_type is unknown."""
        assert pipeline_instance._guess_extension("application/octet-stream", "file.mp3") == "mp3"
        assert pipeline_instance._guess_extension("application/octet-stream", "file.wav") == "wav"
        assert pipeline_instance._guess_extension("application/octet-stream", "file.flac") == "flac"

    def test_default_fallback(self, pipeline_instance):
        """Test default fallback to wav."""
        assert pipeline_instance._guess_extension("application/octet-stream", "noext") == "wav"
