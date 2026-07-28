"""Tests for backend.api.routes — HTTP endpoints via TestClient."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, UTC

import numpy as np
import pytest
import soundfile as sf

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wav_file_bytes():
    """Generate WAV file bytes for upload tests."""
    buf = io.BytesIO()
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    data = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sf.write(buf, data, sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_pipeline():
    """Return a mock TranscriptionPipeline."""
    pipeline = MagicMock()
    return pipeline


@pytest.fixture()
def app(mock_pipeline):
    """Return the FastAPI app with the pipeline mocked at import time."""
    with patch("backend.api.routes._pipeline", mock_pipeline):
        from main import create_app
        yield create_app()


@pytest.fixture()
def client(app, mock_pipeline):
    """Return a TestClient."""
    yield TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/audio/convert (upload + start)
# ---------------------------------------------------------------------------


class TestConvertUploadEndpoint:
    def test_convert_upload_success(self, client, mock_pipeline):
        """Test successful audio upload via /convert returns task_id."""
        mock_result = MagicMock()
        mock_result.task_id = "task_abc123"
        mock_result.status = "processing"
        mock_result.filename = "test.wav"
        mock_result.content_type = "audio/wav"
        mock_result.size_bytes = 32000
        mock_pipeline.upload_audio.return_value = mock_result

        response = client.post(
            "/api/audio/convert",
            files={"file": ("test.wav", io.BytesIO(_wav_file_bytes()), "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task_abc123"
        assert data["status"] == "processing"
        assert data["filename"] == "test.wav"
        assert data["content_type"] == "audio/wav"
        assert data["size_bytes"] == 32000

    def test_convert_upload_mp3(self, client, mock_pipeline):
        """Test upload with MP3 content type."""
        mock_result = MagicMock()
        mock_result.task_id = "task_mp3_001"
        mock_result.status = "processing"
        mock_result.filename = "podcast.mp3"
        mock_result.content_type = "audio/mpeg"
        mock_result.size_bytes = 128000
        mock_pipeline.upload_audio.return_value = mock_result

        response = client.post(
            "/api/audio/convert",
            files={"file": ("podcast.mp3", io.BytesIO(_wav_file_bytes()), "audio/mpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task_mp3_001"
        assert data["filename"] == "podcast.mp3"

    def test_convert_upload_missing_filename(self, client, mock_pipeline):
        """Test upload with missing filename returns 422 (FastAPI validation)."""
        response = client.post(
            "/api/audio/convert",
            files={"file": ("", io.BytesIO(_wav_file_bytes()), "audio/wav")},
        )

        assert response.status_code == 422

    def test_convert_upload_failure(self, client, mock_pipeline):
        """Test upload failure returns 500."""
        mock_pipeline.upload_audio.side_effect = Exception("Processing error")

        response = client.post(
            "/api/audio/convert",
            files={"file": ("test.wav", io.BytesIO(_wav_file_bytes()), "audio/wav")},
        )

        assert response.status_code == 500
        assert "Processing error" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/audio/convert/{task_id}/format
# ---------------------------------------------------------------------------


class TestSetConvertFormat:
    def test_set_format_success(self, client, mock_pipeline):
        """Test setting output format returns 200."""
        mock_info = MagicMock()
        mock_info.task_id = "task123"
        mock_info.status = "processing"
        mock_info.progress = 0.1
        mock_info.started_at = None
        mock_info.completed_at = None
        mock_info.filename = "test.wav"
        mock_info.content_type = "audio/wav"
        mock_info.duration = None
        mock_info.sample_rate = None
        mock_info.original_format = None
        mock_info.original_sample_rate = None
        mock_info.language = None
        mock_info.segment_count = None
        mock_info.processing_time_ms = None
        mock_info.error = None
        mock_info.acknowledged = False

        mock_pipeline.convert_task.return_value = "task123"
        mock_pipeline.get_task_status.return_value = mock_info

        response = client.post(
            "/api/audio/convert/task123/format",
            json={"output_format": "json"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task123"
        mock_pipeline.convert_task.assert_called_once_with(
            task_id="task123", output_format="json"
        )

    def test_set_format_not_found(self, client, mock_pipeline):
        """Test setting format for non-existent task returns 404."""
        mock_pipeline.convert_task.side_effect = KeyError("Task not found")

        response = client.post(
            "/api/audio/convert/nonexistent/format",
            json={"output_format": "srt"},
        )

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/audio/convert/{task_id}
# ---------------------------------------------------------------------------


def _make_task_info(task_id="task123", status="pending", **kwargs):
    """Helper to create a mock TaskInfo."""
    info = MagicMock()
    info.task_id = task_id
    info.status = status
    info.progress = 0.0
    info.started_at = None
    info.completed_at = None
    info.filename = "audio/uploads/test.wav"
    info.content_type = "audio/wav"
    info.duration = None
    info.sample_rate = None
    info.original_format = None
    info.original_sample_rate = None
    info.language = None
    info.segment_count = None
    info.processing_time_ms = None
    info.error = None
    info.acknowledged = False
    for k, v in kwargs.items():
        setattr(info, k, v)
    return info


class TestGetConversionStatus:
    def test_status_completed(self, client, mock_pipeline):
        """Test getting status of a completed task."""
        mock_info = _make_task_info(
            task_id="task123",
            status="completed",
            progress=1.0,
            started_at=datetime(2024, 1, 1, tzinfo=UTC),
            completed_at=datetime(2024, 1, 1, tzinfo=UTC),
            duration=10.5,
            sample_rate=16000,
            original_format="wav",
            original_sample_rate=48000,
            language="it",
            segment_count=5,
            processing_time_ms=3200.0,
        )
        mock_pipeline.get_task_status.return_value = mock_info

        response = client.get("/api/audio/convert/task123")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task123"
        assert data["status"] == "completed"
        assert data["progress"] == 1.0
        assert data["duration"] == 10.5
        assert data["language"] == "it"
        assert data["segment_count"] == 5
        assert data["processing_time_ms"] == 3200.0
        assert "key" not in data
        assert "bucket" not in data
        assert "output_key" not in data

    def test_status_pending(self, client, mock_pipeline):
        """Test getting status of a pending task."""
        mock_pipeline.get_task_status.return_value = _make_task_info(
            task_id="task456", status="processing", progress=0.3
        )

        response = client.get("/api/audio/convert/task456")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["progress"] == 0.3

    def test_status_failed(self, client, mock_pipeline):
        """Test getting status of a failed task."""
        mock_info = _make_task_info(
            task_id="task789",
            status="failed",
            progress=0.3,
            started_at=datetime(2024, 1, 1, tzinfo=UTC),
            completed_at=datetime(2024, 1, 1, tzinfo=UTC),
            error="Processing error: Invalid audio format",
        )
        mock_pipeline.get_task_status.return_value = mock_info

        response = client.get("/api/audio/convert/task789")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Processing error: Invalid audio format"

    def test_status_not_found(self, client, mock_pipeline):
        """Test getting status of non-existent task returns 404."""
        mock_pipeline.get_task_status.side_effect = KeyError("Task not found")

        response = client.get("/api/audio/convert/nonexistent")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/audio/convert/{task_id}/output
# ---------------------------------------------------------------------------


class TestDownloadOutput:
    def test_output_success(self, client, mock_pipeline):
        """Test downloading output of a completed task."""
        mock_pipeline.get_task_output.return_value = "Ciao mondo, questo è un test."
        mock_pipeline.get_task_status.return_value = _make_task_info(
            task_id="task123",
            status="completed",
            filename="test.wav",
        )

        response = client.get("/api/audio/convert/task123/output")

        assert response.status_code == 200
        assert response.text == "Ciao mondo, questo è un test."
        assert "attachment" in response.headers["content-disposition"]

    def test_output_not_completed(self, client, mock_pipeline):
        """Test downloading output of a non-completed task returns 400."""
        mock_pipeline.get_task_output.side_effect = ValueError("Task is not completed (status: processing).")

        response = client.get("/api/audio/convert/task123/output")

        assert response.status_code == 400

    def test_output_not_found(self, client, mock_pipeline):
        """Test downloading output of non-existent task returns 404."""
        mock_pipeline.get_task_output.side_effect = KeyError("Task not found")

        response = client.get("/api/audio/convert/nonexistent/output")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/audio/convert/{task_id}/ack
# ---------------------------------------------------------------------------


class TestAcknowledgeOutput:
    def test_ack_success(self, client, mock_pipeline):
        """Test acknowledging output returns 200."""
        mock_pipeline.acknowledge_task.return_value = None

        response = client.post("/api/audio/convert/task123/ack")

        assert response.status_code == 200
        data = response.json()
        assert data["acknowledged"] is True
        assert data["task_id"] == "task123"
        mock_pipeline.acknowledge_task.assert_called_once_with("task123")

    def test_ack_not_found(self, client, mock_pipeline):
        """Test ACK for non-existent task returns 404."""
        mock_pipeline.acknowledge_task.side_effect = KeyError("Task not found")

        response = client.post("/api/audio/convert/nonexistent/ack")

        assert response.status_code == 404

    def test_ack_already_acked(self, client, mock_pipeline):
        """Test double ACK returns 400."""
        mock_pipeline.acknowledge_task.side_effect = ValueError("Task has already been acknowledged.")

        response = client.post("/api/audio/convert/task123/ack")

        assert response.status_code == 400
        assert "already been acknowledged" in response.json()["detail"]

    def test_ack_not_completed(self, client, mock_pipeline):
        """Test ACK for non-completed task returns 400."""
        mock_pipeline.acknowledge_task.side_effect = ValueError("Task is not completed (status: processing).")

        response = client.post("/api/audio/convert/task123/ack")

        assert response.status_code == 400
        assert "not completed" in response.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE /api/audio/convert/{task_id}
# ---------------------------------------------------------------------------


class TestCancelConversion:
    def test_cancel_success(self, client, mock_pipeline):
        """Test cancelling a running task."""
        mock_info = _make_task_info(
            task_id="task123",
            status="cancelled",
            progress=0.5,
            started_at=datetime(2024, 1, 1, tzinfo=UTC),
            completed_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        mock_pipeline.cancel_task.return_value = mock_info

        response = client.delete("/api/audio/convert/task123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

    def test_cancel_not_found(self, client, mock_pipeline):
        """Test cancelling non-existent task returns 404."""
        mock_pipeline.cancel_task.side_effect = KeyError("Task not found")

        response = client.delete("/api/audio/convert/nonexistent")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/audio/convert (list)
# ---------------------------------------------------------------------------


class TestListConversions:
    def test_list_empty(self, client, mock_pipeline):
        """Test listing tasks when none exist."""
        mock_pipeline.list_tasks.return_value = []

        response = client.get("/api/audio/convert")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_multiple_tasks(self, client, mock_pipeline):
        """Test listing multiple tasks."""
        task1 = _make_task_info(
            task_id="task1",
            status="completed",
            progress=1.0,
            started_at=datetime(2024, 1, 1, tzinfo=UTC),
            completed_at=datetime(2024, 1, 1, tzinfo=UTC),
            duration=5.0,
            sample_rate=16000,
            original_format="wav",
            original_sample_rate=48000,
            language="it",
            segment_count=3,
            processing_time_ms=1500.0,
        )
        task2 = _make_task_info(
            task_id="task2",
            status="processing",
            progress=0.3,
        )

        mock_pipeline.list_tasks.return_value = [task1, task2]

        response = client.get("/api/audio/convert")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["task_id"] == "task1"
        assert data[0]["status"] == "completed"
        assert data[1]["task_id"] == "task2"
        assert data[1]["status"] == "processing"
        # Verify no S3 fields
        assert "key" not in data[0]
        assert "bucket" not in data[0]


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------


class TestCORS:
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present when CORS is configured."""
        response = client.options(
            "/api/audio/convert",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Integration: full upload → status → output → ack flow
# ---------------------------------------------------------------------------


class TestFullFlow:
    """Test the full HTTP flow: upload → status → output → ack."""

    def test_full_flow_mocked(self, client, mock_pipeline):
        """Simulate the full flow with mocked pipeline."""

        # 1. Upload via /convert
        mock_upload_result = MagicMock()
        mock_upload_result.task_id = "integration_task_123"
        mock_upload_result.status = "processing"
        mock_upload_result.filename = "integration_test.wav"
        mock_upload_result.content_type = "audio/wav"
        mock_upload_result.size_bytes = 32000
        mock_pipeline.upload_audio.return_value = mock_upload_result

        upload_resp = client.post(
            "/api/audio/convert",
            files={"file": ("integration_test.wav", io.BytesIO(_wav_file_bytes()), "audio/wav")},
        )
        assert upload_resp.status_code == 200
        task_id = upload_resp.json()["task_id"]
        assert task_id == "integration_task_123"

        # 2. Poll status (processing)
        mock_pipeline.get_task_status.return_value = _make_task_info(
            task_id=task_id,
            status="processing",
            progress=0.3,
            started_at=datetime(2024, 1, 1, tzinfo=UTC),
        )

        status_resp = client.get(f"/api/audio/convert/{task_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "processing"
        assert "key" not in status_resp.json()
        assert "bucket" not in status_resp.json()

        # 3. Poll status (completed)
        mock_pipeline.get_task_status.return_value = _make_task_info(
            task_id=task_id,
            status="completed",
            progress=1.0,
            started_at=datetime(2024, 1, 1, tzinfo=UTC),
            completed_at=datetime(2024, 1, 1, tzinfo=UTC),
            duration=1.0,
            sample_rate=16000,
            original_format="wav",
            original_sample_rate=16000,
            language="it",
            segment_count=2,
            processing_time_ms=500.0,
        )

        status_resp = client.get(f"/api/audio/convert/{task_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "completed"

        # 4. Download output
        mock_pipeline.get_task_output.return_value = "Ciao mondo, questo è un test."
        output_resp = client.get(f"/api/audio/convert/{task_id}/output")
        assert output_resp.status_code == 200
        assert "Ciao mondo" in output_resp.text

        # 5. Acknowledge
        mock_pipeline.acknowledge_task.return_value = None
        ack_resp = client.post(f"/api/audio/convert/{task_id}/ack")
        assert ack_resp.status_code == 200
        assert ack_resp.json()["acknowledged"] is True

        # 6. Output after ACK should fail
        mock_pipeline.get_task_output.side_effect = ValueError("Output file not found.")
        output_resp_after = client.get(f"/api/audio/convert/{task_id}/output")
        assert output_resp_after.status_code == 400

        # 7. List all tasks
        mock_pipeline.list_tasks.return_value = [mock_pipeline.get_task_status.return_value]
        list_resp = client.get("/api/audio/convert")
        assert list_resp.status_code == 200
        assert len(list_resp.json()) == 1

        # Verify no S3 fields in list response
        for task_data in list_resp.json():
            assert "key" not in task_data
            assert "bucket" not in task_data
            assert "output_key" not in task_data
