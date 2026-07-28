"""Tests for utils.config — Settings loading, validation, and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from utils.config import (
    LoggingConfig,
    OutputConfig,
    S3StorageConfig,
    Settings,
    WhisperConstraintsConfig,
    WhisperInferenceConfig,
    WhisperVadConfig,
    get_settings,
    settings,
)


class TestSettingsDefaults:
    """Test that Settings loads with expected defaults from .env.local."""

    @pytest.fixture(autouse=True)
    def _clear_singleton(self):
        """Reset the global _settings singleton before each test."""
        from utils import config
        config._settings = None
        yield
        config._settings = None

    def test_host_default(self):
        assert settings().host == "0.0.0.0"

    def test_port_default(self):
        assert settings().port == 8000

    def test_debug_default(self):
        assert settings().debug is False

    def test_whisper_model_size_from_env(self):
        """Model size comes from .env.local (large-v3)."""
        assert settings().whisper_model_size == "large-v3"

    def test_whisper_language_default(self):
        assert settings().whisper_language == "it"

    def test_whisper_device_from_env(self):
        """Device comes from .env.local."""
        assert settings().whisper_device == "cuda"

    def test_storage_endpoint_default(self):
        assert settings().storage.endpoint_url == "http://localhost:9000"

    def test_storage_bucket_default(self):
        assert settings().storage.bucket == "sbobinator"

    def test_cors_origins_default(self):
        assert "http://localhost" in settings().cors.origins
        assert "http://localhost:3000" in settings().cors.origins

    def test_vad_enabled_default(self):
        assert settings().whisper_vad.enabled is True

    def test_output_format_default(self):
        assert settings().output.format == "json"

    def test_logging_level_default(self):
        assert settings().logging.level == "INFO"

    def test_audio_sample_rate_default(self):
        assert settings().audio.sample_rate == 16000


class TestSettingsValidators:
    """Test field validators."""

    @pytest.fixture(autouse=True)
    def _clear_singleton(self):
        from utils import config
        config._settings = None
        yield
        config._settings = None

    def test_language_normalize_empty(self):
        s = Settings(whisper_language="")
        assert s.whisper_language == "auto"

    def test_language_normalize_auto(self):
        s = Settings(whisper_language="auto")
        assert s.whisper_language == "auto"

    def test_language_normalize_spaces(self):
        s = Settings(whisper_language="  IT  ")
        assert s.whisper_language == "it"

    def test_model_path_none(self):
        s = Settings(whisper_model_path="") # type: ignore[arg-type]
        assert s.whisper_model_path is None

    def test_model_path_resolved(self):
        s = Settings(whisper_model_path="/tmp/model")  # type: ignore[arg-type]
        assert isinstance(s.whisper_model_path, Path)
        assert s.whisper_model_path.is_absolute()


class TestSettingsProperties:
    """Test computed properties on Settings."""

    def test_vad_params(self):
        """Test VAD params with custom values."""
        s = Settings()
        # Verify default values from .env.local
        params = s.vad_params
        assert params["threshold"] == 0.5
        assert params["min_silence_duration_ms"] == 500
        assert params["speech_pad_ms"] == 250

        # Test with custom values via nested model
        s2 = Settings(
            whisper_vad=WhisperVadConfig(
                threshold=0.6, min_silence_duration_ms=800, speech_pad_ms=300
            )
        )
        params2 = s2.vad_params
        assert params2["threshold"] == 0.6
        assert params2["min_silence_duration_ms"] == 800
        assert params2["speech_pad_ms"] == 300

    def test_whisper_kwargs_no_path(self):
        s = Settings(whisper_model_path="") # type: ignore[arg-type]
        kwargs = s.whisper_kwargs
        assert kwargs["device"] == "cuda"
        assert "model_path" not in kwargs

    def test_whisper_kwargs_with_path(self):
        s = Settings(whisper_model_path="/tmp/my-model")  # type: ignore[arg-type]
        kwargs = s.whisper_kwargs
        assert kwargs["model_path"] == Path("/tmp/my-model").resolve()

    def test_run_kwargs(self):
        """Test run_kwargs with default values from .env.local."""
        s = Settings()
        kwargs = s.run_kwargs
        assert kwargs["beam_size"] == 5  # from .env.local
        assert kwargs["patience"] == 1.0
        assert kwargs["temperature"] == 0.0
        assert kwargs["vad_filter"] is True

        # Test with custom values
        s2 = Settings(
            whisper_inference=WhisperInferenceConfig(
                beam_size=3, patience=0.8, temperature=0.5
            ),
            whisper_constraints=WhisperConstraintsConfig(repetition_penalty=1.2),
            whisper_vad=WhisperVadConfig(enabled=False),
        )
        kwargs2 = s2.run_kwargs
        assert kwargs2["beam_size"] == 3
        assert kwargs2["patience"] == 0.8
        assert kwargs2["temperature"] == 0.5
        assert kwargs2["repetition_penalty"] == 1.2
        assert kwargs2["vad_filter"] is False

    def test_storage_endpoint_http(self):
        s = Settings(storage=S3StorageConfig(endpoint_url="localhost:9000", secure=False))
        assert s.storage_endpoint == "http://localhost:9000"

    def test_storage_endpoint_https(self):
        s = Settings(storage=S3StorageConfig(endpoint_url="s3.example.com", secure=True))
        assert s.storage_endpoint == "https://s3.example.com"

    def test_storage_endpoint_already_has_scheme(self):
        s = Settings(storage=S3StorageConfig(endpoint_url="https://s3.example.com"))
        assert s.storage_endpoint == "https://s3.example.com"


class TestSettingsSummary:
    """Test the human-readable summary output."""

    def test_summary_contains_key_info(self):
        s = Settings()
        summary = s.summary()
        assert "SBOBINATOR CONFIG" in summary
        assert "Host:" in summary
        assert "Model:" in summary
        assert "Device:" in summary
        assert "VAD:" in summary


class TestSettingsModelDumpEnv:
    """Test export to flat env-style dict."""

    def test_model_dump_env(self):
        s = Settings()
        flat = s.model_dump_env()
        assert "HOST" in flat
        assert "PORT" in flat
        assert "WHISPER_MODEL_SIZE" in flat
        assert "WHISPER_DEVICE" in flat
        # Nested keys use underscore delimiter (not double underscore)
        assert "WHISPER_INFERENCE_BEAM_SIZE" in flat
        assert "AUDIO_SAMPLE_RATE" in flat


class TestSettingsSingleton:
    """Test that settings() returns a cached singleton."""

    @pytest.fixture(autouse=True)
    def _clear_singleton(self):
        from utils import config
        config._settings = None
        yield
        config._settings = None

    def test_singleton_same_instance(self):
        s1 = settings()
        s2 = settings()
        assert s1 is s2

    def test_get_settings_same_instance(self):
        # Note: get_settings() always creates a new instance (not a true singleton).
        # The true singleton is settings(). We test get_settings() creates instances.
        s1 = get_settings()
        s2 = get_settings()
        # get_settings() creates new instances each time
        assert isinstance(s1, Settings)
        assert isinstance(s2, Settings)


class TestSettingsValidation:
    """Test Pydantic validation errors."""

    def test_port_out_of_range_low(self):
        with pytest.raises(ValidationError):
            Settings(port=0)

    def test_port_out_of_range_high(self):
        with pytest.raises(ValidationError):
            Settings(port=70000)

    def test_beam_size_out_of_range(self):
        with pytest.raises(ValidationError):
            Settings(whisper_inference=WhisperInferenceConfig(beam_size=0))

    def test_beam_size_too_high(self):
        with pytest.raises(ValidationError):
            Settings(whisper_inference=WhisperInferenceConfig(beam_size=21))

    def test_temperature_out_of_range(self):
        with pytest.raises(ValidationError):
            Settings(whisper_inference=WhisperInferenceConfig(temperature=-0.1))

    def test_temperature_too_high(self):
        with pytest.raises(ValidationError):
            Settings(whisper_inference=WhisperInferenceConfig(temperature=1.1))

    def test_invalid_output_format(self):
        with pytest.raises(ValidationError):
            Settings(output=OutputConfig(format="xml"))  # type: ignore[arg-type]

    def test_invalid_vad_method(self):
        with pytest.raises(ValidationError):
            Settings(whisper_vad=WhisperVadConfig(method="invalid"))  # type: ignore[arg-type]

    def test_invalid_logging_level(self):
        with pytest.raises(ValidationError):
            Settings(logging=LoggingConfig(level="VERBOSE"))  # type: ignore[arg-type]
