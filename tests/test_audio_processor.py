"""Tests for backend.modules.preprocessing.audio_processor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from backend.modules.preprocessing.audio_processor import (
    AudioProcessor,
    AudioProcessingError,
    _estimate_noise_profile,
    _guess_format,
    convert_to_mono,
    load_audio,
    normalize_peak,
    remove_noise,
    remove_silence,
    resample,
    trim_audio,
)


# ---------------------------------------------------------------------------
# _guess_format
# ---------------------------------------------------------------------------


class TestGuessFormat:
    def test_wav_mime(self):
        assert _guess_format("audio/wav", "file.wav") == "wav"

    def test_wav_x_wav_mime(self):
        assert _guess_format("audio/x-wav", "file.wav") == "wav"

    def test_mp3_mime(self):
        assert _guess_format("audio/mpeg", "file.mp3") == "mp3"

    def test_mp3_mp3_mime(self):
        assert _guess_format("audio/mp3", "file.mp3") == "mp3"

    def test_flac_mime(self):
        assert _guess_format("audio/flac", "file.flac") == "flac"

    def test_ogg_mime(self):
        assert _guess_format("audio/ogg", "file.ogg") == "ogg"

    def test_m4a_video_mime(self):
        assert _guess_format("video/mp4", "file.m4a") == "m4a"

    def test_webm_video_mime(self):
        assert _guess_format("video/webm", "file.webm") == "webm"

    def test_filename_fallback(self):
        assert _guess_format("application/octet-stream", "recording.wav") == "wav"

    def test_filename_fallback_mp3(self):
        assert _guess_format("application/octet-stream", "recording.mp3") == "mp3"

    def test_unsupported_format_raises(self):
        with pytest.raises(AudioProcessingError, match="Unsupported audio format"):
            _guess_format("application/octet-stream", "file.xyz")

    def test_mime_with_charset(self):
        """MIME type with charset parameter (e.g. 'audio/wav; charset=utf-8')."""
        assert _guess_format("audio/wav; charset=utf-8", "file.wav") == "wav"


# ---------------------------------------------------------------------------
# convert_to_mono
# ---------------------------------------------------------------------------


class TestConvertToMono:
    def test_already_mono(self):
        data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = convert_to_mono(data)
        np.testing.assert_array_equal(result, data)

    def test_stereo_to_mono(self):
        data = np.array([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]], dtype=np.float32)
        result = convert_to_mono(data)
        assert result.ndim == 1
        np.testing.assert_array_almost_equal(result, [0.5, 0.5, 0.5])

    def test_multichannel_to_mono(self):
        data = np.random.rand(100, 4).astype(np.float32)
        result = convert_to_mono(data)
        assert result.ndim == 1
        assert len(result) == 100


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------


class TestResample:
    def test_same_sample_rate_no_change(self):
        data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = resample(data, 16000, 16000)
        np.testing.assert_array_equal(result, data)

    def test_upsample(self):
        data = np.array([0.1, 0.2], dtype=np.float32)
        result = resample(data, 8000, 16000)
        assert len(result) > len(data)
        assert result.dtype == np.float32

    def test_downsample(self):
        data = np.random.rand(32000).astype(np.float32)
        result = resample(data, 48000, 16000)
        # 32000 * (16000/48000) ≈ 10666
        assert len(result) < len(data)

    def test_zero_length_raises(self):
        data = np.array([], dtype=np.float32)
        with pytest.raises(AudioProcessingError, match="zero samples"):
            resample(data, 16000, 32000)


# ---------------------------------------------------------------------------
# trim_audio
# ---------------------------------------------------------------------------


class TestTrimAudio:
    def test_no_trim_when_max_zero(self):
        data = np.random.rand(32000).astype(np.float32)
        result = trim_audio(data, 16000, 0)
        np.testing.assert_array_equal(result, data)

    def test_no_trim_when_max_negative(self):
        data = np.random.rand(32000).astype(np.float32)
        result = trim_audio(data, 16000, -1)
        np.testing.assert_array_equal(result, data)

    def test_trim_shorter_than_max(self):
        data = np.random.rand(16000).astype(np.float32)  # 1 second
        result = trim_audio(data, 16000, 10)  # max 10 seconds
        np.testing.assert_array_equal(result, data)

    def test_trim_longer_than_max(self):
        data = np.random.rand(48000).astype(np.float32)  # 3 seconds
        result = trim_audio(data, 16000, 1)  # max 1 second
        assert len(result) == 16000


# ---------------------------------------------------------------------------
# remove_silence
# ---------------------------------------------------------------------------


class TestRemoveSilence:
    def test_remove_leading_trailing_silence(self):
        sr = 16000
        silence = np.zeros(int(sr * 0.5), dtype=np.float32)
        speech = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5))).astype(np.float32)
        data = np.concatenate([silence, speech, silence])

        result = remove_silence(data, sr)
        assert len(result) < len(data)
        assert len(result) > 0

    def test_all_silence_raises(self):
        sr = 16000
        data = np.zeros(sr, dtype=np.float32)
        with pytest.raises(AudioProcessingError, match="entirely silent"):
            remove_silence(data, sr)

    def test_no_silence_to_remove(self):
        sr = 16000
        # Create a signal well above the silence threshold
        data = np.full(sr, 0.9, dtype=np.float32)
        result = remove_silence(data, sr)
        np.testing.assert_array_equal(result, data)


# ---------------------------------------------------------------------------
# normalize_peak
# ---------------------------------------------------------------------------


class TestNormalizePeak:
    def test_normalize_below_target(self):
        data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = normalize_peak(data)
        np.testing.assert_array_equal(result, data)

    def test_normalize_above_target(self):
        data = np.array([0.98, 0.99, 1.0], dtype=np.float32)
        result = normalize_peak(data)
        assert np.max(np.abs(result)) <= 0.97 + 1e-6

    def test_zero_signal_unchanged(self):
        data = np.zeros(100, dtype=np.float32)
        result = normalize_peak(data)
        np.testing.assert_array_equal(result, data)


# ---------------------------------------------------------------------------
# remove_noise
# ---------------------------------------------------------------------------


class TestRemoveNoise:
    def test_noise_reduction_returns_array(self):
        sr = 16000
        speech = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
        noise = np.random.randn(sr).astype(np.float32) * 0.1
        data = (speech + noise).astype(np.float32)

        result = remove_noise(data, sr)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_noise_reduction_with_invalid_noise_profile(self):
        """When noise reduction fails, it should return the original data."""
        sr = 16000
        data = np.random.randn(sr).astype(np.float32)
        # Mock reduce_noise to raise an exception
        with patch("backend.modules.preprocessing.audio_processor.nr.reduce_noise", side_effect=Exception("fail")):
            result = remove_noise(data, sr)
            np.testing.assert_array_equal(result, data)


# ---------------------------------------------------------------------------
# _estimate_noise_profile
# ---------------------------------------------------------------------------


class TestEstimateNoiseProfile:
    def test_estimate_noise_profile(self):
        sr = 16000
        data = np.random.randn(sr * 10).astype(np.float32)  # 10 seconds
        profile = _estimate_noise_profile(data, sr, frame_length=0.5)
        assert len(profile) == sr * 0.5  # 0.5 seconds

    def test_short_audio(self):
        sr = 16000
        data = np.random.randn(sr).astype(np.float32)  # 1 second
        profile = _estimate_noise_profile(data, sr, frame_length=2.0)  # request 2s
        assert len(profile) == sr  # returns entire audio


# ---------------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------------


class TestLoadAudio:
    def test_load_wav(self, wav_bytes_16khz_mono):
        samples, sr = load_audio(wav_bytes_16khz_mono, "wav")
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32
        assert sr == 16000
        assert len(samples) == 16000

    def test_load_wav_stereo(self, wav_bytes_48khz_stereo):
        samples, sr = load_audio(wav_bytes_48khz_stereo, "wav")
        assert sr == 48000
        assert samples.ndim == 2  # stereo
        assert samples.shape[1] == 2

    def test_unsupported_format_raises(self):
        with pytest.raises(AudioProcessingError, match="Unsupported format"):
            load_audio(b"garbage", "xyz")

    def test_invalid_wav_raises(self):
        with pytest.raises(AudioProcessingError):
            load_audio(b"not a real wav", "wav")


# ---------------------------------------------------------------------------
# AudioProcessor (full pipeline)
# ---------------------------------------------------------------------------


class TestAudioProcessor:
    def test_full_pipeline(self, wav_bytes_16khz_mono):
        """Test the full preprocessing pipeline produces valid output."""
        processor = AudioProcessor()
        result = processor.process(
            raw_bytes=wav_bytes_16khz_mono,
            content_type="audio/wav",
            filename="test.wav",
        )
        assert result.sample_rate == 16000
        assert result.original_format == "wav"
        assert result.duration > 0
        assert result.data.dtype == np.float32
        assert result.data.ndim == 1  # mono
        assert np.max(np.abs(result.data)) <= 1.0

    def test_full_pipeline_stereo(self, wav_bytes_48khz_stereo):
        """Test the full pipeline handles stereo 48kHz input."""
        processor = AudioProcessor()
        result = processor.process(
            raw_bytes=wav_bytes_48khz_stereo,
            content_type="audio/x-wav",
            filename="test_stereo.wav",
        )
        assert result.sample_rate == 16000
        assert result.original_format == "wav"
        assert result.original_sample_rate == 48000
        assert result.data.ndim == 1  # converted to mono

    def test_processor_with_mp3_content_type(self, mp3_like_bytes):
        """Test that unsupported format raises AudioProcessingError."""
        processor = AudioProcessor()
        with pytest.raises(AudioProcessingError):
            processor.process(
                raw_bytes=mp3_like_bytes,
                content_type="audio/mpeg",
                filename="fake.mp3",
            )

    def test_processor_custom_config(self, wav_bytes_16khz_mono):
        """Test with a custom AudioConfig."""
        from utils.config import AudioConfig

        custom_config = AudioConfig(
            sample_rate=8000,
            max_duration=1,
            noise_reduction=False,
        )
        processor = AudioProcessor(config=custom_config)
        result = processor.process(
            raw_bytes=wav_bytes_16khz_mono,
            content_type="audio/wav",
        )
        assert result.sample_rate == 8000
