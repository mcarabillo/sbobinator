"""Tests for backend.api.routes — HTTP endpoints via TestClient."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

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
    # Patch the module-level _pipeline variable directly
    with patch("backend.api.routes._pipeline", mock_pipeline):
        from main import create_app
        yield create_app()


@pytest.fixture()
def client(app, mock_pipeline):
    """Return a TestClient."""
    yield TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/audio/upload
# ---------------------------------------------------------------------------


class TestUploadEndpoint:
    def test_upload_success(self, client, mock_pipeline):
        """Test successful audio upload."""
        mock_result = MagicMock()
        mock_result.key = "audio/uploads/abc123.wav"
        mock_result.bucket = "sbobinator"
        mock_result.size_bytes = 32000
        mock_result.content_type = "audio/wav"
        mock_result.uploaded_at = MagicMock(isoformat=lambda: "2024-01-01T00:00:00+00:00")
        mock_pipeline.upload_audio.return_value = mock_result

        response = client.post(
            "/api/audio/upload",
            files={"file": ("test.wav", io.BytesIO(_wav_file_bytes()), "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["key"] == "audio/uploads/abc123.wav"
        assert data["bucket"] == "sbobinator"
        assert data["size_bytes"] == 32000
        assert data["content_type"] == "audio/wav"

    def test_upload_with_bucket_and_destination(self, client, mock_pipeline):
        """Test upload with custom bucket and destination."""
        mock_result = MagicMock()
        mock_result.key = "custom-bucket/my-recording.wav"
        mock_result.bucket = "custom-bucket"
        mock_result.size_bytes = 1000
        mock_result.content_type = "audio/wav"
        mock_result.uploaded_at = MagicMock(isoformat=lambda: "2024-01-01T00:00:00+00:00")
        mock_pipeline.upload_audio.return_value = mock_result

        response = client.post(
            "/api/audio/upload",
            files={"file": ("test.wav", io.BytesIO(_wav_file_bytes()), "audio/wav")},
            data={"bucket": "custom-bucket", "destination": "my-recording.wav"},
        )

        assert response.status_code == 200
        mock_pipeline.upload_audio.assert_called_once()
        call_kwargs = mock_pipeline.upload_audio.call_args
        assert call_kwargs.kwargs["bucket"] == "custom-bucket"
        assert call_kwargs.kwargs["destination"] == "my-recording.wav"

    def test_upload_missing_filename(self, client, mock_pipeline):
        """Test upload with missing filename returns 422 (validation error).

        FastAPI validates the UploadFile before our route code runs,
        so an empty filename results in a 422 Unprocessable Entity.
        """
        response = client.post(
            "/api/audio/upload",
            files={"file": ("", io.BytesIO(_wav_file_bytes()), "audio/wav")},
        )

        assert response.status_code == 422

    def test_upload_failure(self, client, mock_pipeline):
        """Test upload failure returns 500."""
        mock_pipeline.upload_audio.side_effect = Exception("S3 error")

        response = client.post(
            "/api/audio/upload",
            files={"file": ("test.wav", io.BytesIO(_wav_file_bytes()), "audio/wav")},
        )

        assert response.status_code == 500
        assert "S3 error" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/audio/convert
# ---------------------------------------------------------------------------


def _make_task_info(task_id="task123", status="pending", **kwargs):
    """Helper to create a mock TaskInfo."""
    info = MagicMock()
    info.task_id = task_id
    info.status = status
    info.key = "audio/uploads/test.wav"
    info.bucket = "sbobinator"
    info.progress = 0.0
    info.started_at = None
    info.completed_at = None
    info.output_key = None
    info.duration = None
    info.sample_rate = None
    info.original_format = None
    info.original_sample_rate = None
    info.language = None
    info.segment_count = None
    info.processing_time_ms = None
    info.error = None
    for k, v in kwargs.items():
        setattr(info, k, v)
    return info


class TestConvertEndpoint:
    def test_convert_success(self, client, mock_pipeline):
        """Test starting a conversion job returns 202 with task_id."""
        mock_pipeline.start_conversion.return_value = "abc123def456"
        mock_pipeline.get_task_status.return_value = _make_task_info(
            task_id="abc123def456", status="pending"
        )

        response = client.post(
            "/api/audio/convert",
            json={"key": "audio/uploads/test.wav", "bucket": "sbobinator", "output_format": "txt"},
        )

        assert response.status_code == 202
        data = response.json()
        assert data["task_id"] == "abc123def456"
        assert data["status"] == "pending"

    def test_convert_minimal_payload(self, client, mock_pipeline):
        """Test convert with minimal payload (only key required)."""
        mock_pipeline.start_conversion.return_value = "task123"
        mock_pipeline.get_task_status.return_value = _make_task_info(
            task_id="task123", status="pending"
        )

        response = client.post(
            "/api/audio/convert",
            json={"key": "audio/uploads/test.wav"},
        )

        assert response.status_code == 202
        data = response.json()
        assert data["task_id"] == "task123"

    def test_convert_failure(self, client, mock_pipeline):
        """Test convert start failure returns 500."""
        mock_pipeline.start_conversion.side_effect = Exception("File not found")

        response = client.post(
            "/api/audio/convert",
            json={"key": "audio/uploads/missing.wav"},
        )

        assert response.status_code == 500
        assert "Failed to start conversion" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/audio/convert/{task_id}
# ---------------------------------------------------------------------------


class TestGetConversionStatus:
    def test_status_completed(self, client, mock_pipeline):
        """Test getting status of a completed task."""
        from datetime import datetime, timezone

        mock_info = _make_task_info(
            task_id="task123",
            status="completed",
            progress=1.0,
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            output_key="audio/test.txt",
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

    def test_status_pending(self, client, mock_pipeline):
        """Test getting status of a pending task."""
        mock_pipeline.get_task_status.return_value = _make_task_info(
            task_id="task456", status="pending", progress=0.0
        )

        response = client.get("/api/audio/convert/task456")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["progress"] == 0.0

    def test_status_failed(self, client, mock_pipeline):
        """Test getting status of a failed task."""
        from datetime import datetime, timezone

        mock_info = _make_task_info(
            task_id="task789",
            status="failed",
            progress=0.3,
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            error="Fetch error: Object not found",
        )
        mock_pipeline.get_task_status.return_value = mock_info

        response = client.get("/api/audio/convert/task789")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Fetch error: Object not found"

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
            output_key="audio/test.txt",
        )

        response = client.get("/api/audio/convert/task123/output")

        assert response.status_code == 200
        assert response.text == "Ciao mondo, questo è un test."
        assert "attachment" in response.headers["content-disposition"]

    def test_output_not_completed(self, client, mock_pipeline):
        """Test downloading output of a non-completed task returns 400."""
        mock_pipeline.get_task_output.side_effect = ValueError("Task is not completed (status: pending).")
        mock_pipeline.get_task_status.return_value = _make_task_info(
            task_id="task123", status="pending"
        )

        response = client.get("/api/audio/convert/task123/output")

        assert response.status_code == 400

    def test_output_not_found(self, client, mock_pipeline):
        """Test downloading output of non-existent task returns 404."""
        mock_pipeline.get_task_output.side_effect = KeyError("Task not found")

        response = client.get("/api/audio/convert/nonexistent/output")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/audio/convert/{task_id}
# ---------------------------------------------------------------------------


class TestCancelConversion:
    def test_cancel_success(self, client, mock_pipeline):
        """Test cancelling a running task."""
        from datetime import datetime, timezone

        mock_info = _make_task_info(
            task_id="task123",
            status="cancelled",
            progress=0.5,
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
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
        from datetime import datetime, timezone

        task1 = _make_task_info(
            task_id="task1",
            status="completed",
            progress=1.0,
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            output_key="audio/test1.txt",
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
            status="pending",
            progress=0.0,
        )

        mock_pipeline.list_tasks.return_value = [task1, task2]

        response = client.get("/api/audio/convert")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["task_id"] == "task1"
        assert data[0]["status"] == "completed"
        assert data[1]["task_id"] == "task2"
        assert data[1]["status"] == "pending"


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------


class TestCORS:
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present when CORS is configured."""
        response = client.options(
            "/api/audio/upload",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Integration: full upload → convert → poll → download flow
# ---------------------------------------------------------------------------


class TestFullFlow:
    """Test the full HTTP flow: upload → convert → poll → download."""

    def test_full_flow_mocked(self, client, mock_pipeline):
        """Simulate the full flow with mocked pipeline."""
        from datetime import datetime, timezone

        # 1. Upload
        mock_upload_result = MagicMock()
        mock_upload_result.key = "audio/uploads/integration_test.wav"
        mock_upload_result.bucket = "sbobinator"
        mock_upload_result.size_bytes = 32000
        mock_upload_result.content_type = "audio/wav"
        mock_upload_result.uploaded_at = MagicMock(isoformat=lambda: "2024-01-01T00:00:00+00:00")
        mock_pipeline.upload_audio.return_value = mock_upload_result

        upload_resp = client.post(
            "/api/audio/upload",
            files={"file": ("integration_test.wav", io.BytesIO(_wav_file_bytes()), "audio/wav")},
        )
        assert upload_resp.status_code == 200
        upload_key = upload_resp.json()["key"]

        # 2. Convert (start)
        mock_pipeline.start_conversion.return_value = "integration_task_123"
        mock_pipeline.get_task_status.return_value = _make_task_info(
            task_id="integration_task_123",
            status="pending",
            key=upload_key,
        )

        convert_resp = client.post(
            "/api/audio/convert",
            json={"key": upload_key, "output_format": "txt"},
        )
        assert convert_resp.status_code == 202
        task_id = convert_resp.json()["task_id"]

        # 3. Poll status (simulate completed)
        mock_pipeline.get_task_status.return_value = _make_task_info(
            task_id=task_id,
            status="completed",
            progress=1.0,
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            output_key="audio/test.txt",
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

        # 5. List all tasks
        mock_pipeline.list_tasks.return_value = [mock_pipeline.get_task_status.return_value]
        list_resp = client.get("/api/audio/convert")
        assert list_resp.status_code == 200
        assert len(list_resp.json()) == 1
