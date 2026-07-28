"""Shared fixtures for all tests."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Fixtures: Audio data
# ---------------------------------------------------------------------------


@pytest.fixture()
def wav_bytes_16khz_mono() -> bytes:
    """Generate a short WAV file (16 kHz, mono, 1 second, 440 Hz sine wave)."""
    buf = io.BytesIO()
    sr = 16000
    t = np.linspace(0, 1, sr)
    data = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sf.write(buf, data, sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


@pytest.fixture()
def wav_bytes_48khz_stereo() -> bytes:
    """Generate a WAV file (48 kHz, stereo, 0.5 second)."""
    buf = io.BytesIO()
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5))
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    data = np.column_stack([left, right])
    sf.write(buf, data, sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


@pytest.fixture()
def wav_bytes_silence() -> bytes:
    """Generate a WAV file that is entirely silence."""
    buf = io.BytesIO()
    sr = 16000
    data = np.zeros(sr, dtype=np.float32)
    sf.write(buf, data, sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


@pytest.fixture()
def mp3_like_bytes() -> bytes:
    """Return garbage bytes that look like an unsupported format.

    We use this to test error paths for unsupported formats.
    """
    return b"NOT_A_REAL_MP3_FILE"


# ---------------------------------------------------------------------------
# Fixtures: Mock S3 / MinIO
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_s3_client():
    """Return a mock boto3 S3 client."""
    client = MagicMock()
    client.get_object.return_value = {
        "Body": MagicMock(read=lambda: b"fake audio data"),
        "ContentType": "audio/wav",
    }
    client.put_object.return_value = {}
    return client


@pytest.fixture()
def mock_boto3_session(mock_s3_client):
    """Return a mocked boto3.Session that yields mock_s3_client."""
    session = MagicMock()
    session.client.return_value = mock_s3_client
    return session


# ---------------------------------------------------------------------------
# Fixtures: Mock Whisper
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_whisper_segments():
    """Return mock Whisper segment objects."""
    seg1 = MagicMock()
    seg1.start = 0.0
    seg1.end = 1.5
    seg1.text = "Ciao mondo"

    seg2 = MagicMock()
    seg2.start = 2.0
    seg2.end = 4.0
    seg2.text = "Questo è un test"

    return [seg1, seg2]


@pytest.fixture()
def mock_whisper_info():
    """Return mock Whisper transcription info."""
    info = MagicMock()
    info.language = "it"
    info.language_probability = 0.99
    return info


@pytest.fixture()
def mock_batched_pipeline(mock_whisper_segments, mock_whisper_info):
    """Return a mock BatchedInferencePipeline."""
    pipeline = MagicMock()
    pipeline.transcribe.return_value = (iter(mock_whisper_segments), mock_whisper_info)
    return pipeline
