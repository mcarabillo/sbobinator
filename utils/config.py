"""Centralized configuration for the sbobinator service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Nested Settings Models
# ---------------------------------------------------------------------------

class CorsConfig(BaseModel):
    """CORS configuration."""

    origins: list[str] = Field(
        default_factory=lambda: ["http://localhost", "http://localhost:3000"],
        description="Allowed CORS origins",
    )


class WhisperInferenceConfig(BaseModel):
    """Whisper inference parameters."""

    bip: bool = Field(default=True, description="Use BatchedInferencePipeline (True) or custom chunking pipeline (False)")
    beam_size: int = Field(default=5, ge=1, le=20, description="Beam search size")
    patience: float = Field(default=1.0, ge=0.1, le=10.0, description="Beam search patience")
    temperature: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Sampling temperature (0.0 = greedy)"
    )
    num_workers: int = Field(default=1, ge=0, le=8, description="Number of parallel workers (custom pipeline only)")
    batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Batch size for BatchedInferencePipeline",
    )


class WhisperConstraintsConfig(BaseModel):
    """Text generation constraints."""

    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0, description="Repetition penalty")
    length_ratio: float = Field(default=2.5, ge=0.1, le=10.0, description="Output/input length ratio")
    spm_silence_threshold: float = Field(default=0.01, ge=0.0, le=1.0, description="Silence threshold for Split-by-Piece")
    wer_silence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Silence threshold for Split-by-WER")
    hotwords: list[str] = Field(default_factory=list, description="Hotwords for prompt conditioning")
    ban_token_ids: list[int] = Field(default_factory=list, description="Token IDs to ban")
    allow_punctuation_tokens: list[int] = Field(default_factory=list, description="Allowed punctuation tokens")


class WhisperVadConfig(BaseModel):
    """Voice Activity Detection configuration."""

    enabled: bool = Field(default=True, description="Enable VAD filtering")
    method: Literal["silero", "webrtc", "pyannote"] = Field(
        default="silero", description="VAD method to use"
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="VAD threshold")
    min_silence_duration_ms: int = Field(default=500, ge=0, le=10000, description="Min silence duration in ms")
    speech_pad_ms: int = Field(default=250, ge=0, le=5000, description="Speech padding in ms")


class WhisperTimestampConfig(BaseModel):
    """Timestamp and segmentation parameters."""

    max_initial_timestamp: int = Field(default=1, ge=0, le=30, description="Max initial timestamp")
    one_segment: bool = Field(default=False, description="Force single segment")
    segment_duration: int = Field(default=30, ge=0, le=300, description="Max segment duration in seconds")


class AudioConfig(BaseModel):
    """Audio preprocessing configuration."""

    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="Audio sample rate in Hz")
    chunk_duration: int = Field(default=30, ge=1, le=300, description="Audio chunk duration in seconds")
    max_duration: int = Field(default=300, ge=0, le=3600, description="Max file duration in seconds (0 = unlimited)")
    noise_reduction: bool = Field(default=True, description="Enable noise reduction")
    noise_reduction_sample_rate: int = Field(default=16000, ge=8000, le=48000, description="Noise reduction sample rate")
    noise_reduction_frame_length: float = Field(default=0.5, ge=0.01, le=2.0, description="Noise reduction frame length")


class OutputConfig(BaseModel):
    """Output configuration."""

    format: Literal["json", "srt", "vtt", "txt", "csv", "tsv"] = Field(
        default="json", description="Default output format"
    )
    use_formatter: bool = Field(
        default=False,
        description="Use OutputFormatter for formatted output. When False, returns raw transcription text.",
    )
    include_metadata: bool = Field(default=True, description="Include metadata in response")
    save_transcripts: bool = Field(default=True, description="Save transcripts to disk")
    transcripts_dir: Path = Field(
        default=Path("./transcripts"), description="Directory for saved transcripts"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    format: Literal["json", "text"] = Field(default="json", description="Log format")


class S3StorageConfig(BaseModel):
    """S3 / MinIO storage configuration."""

    endpoint_url: str = Field(
        default="http://localhost:9000",
        description="S3/MinIO endpoint URL",
    )
    access_key_id: str = Field(
        default="minioadmin",
        description="Access key (AWS access key ID)",
    )
    secret_access_key: str = Field(
        default="minioadmin",
        description="Secret access key",
    )
    bucket: str = Field(
        default="sbobinator",
        description="Default S3 bucket name",
    )
    region: str = Field(
        default="us-east-1",
        description="AWS region (ignored by most MinIO setups)",
    )
    secure: bool = Field(
        default=False,
        description="Use HTTPS (True) or HTTP (False)",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed requests",
    )
    timeout: int = Field(
        default=120,
        ge=1,
        le=600,
        description="Request timeout in seconds",
    )
    max_audio_size: int = Field(
        default=2 * 1024 * 1024 * 1024,
        ge=1,
        description="Maximum audio file size in bytes (default 2 GB)",
    )
    upload_transcripts: bool = Field(
        default=True,
        description="Upload completed transcripts back to S3",
    )
    transcripts_bucket: str = Field(
        default="sbobinator-transcripts",
        description="S3 bucket for storing transcript outputs",
    )
    transcripts_prefix: str = Field(
        default="transcripts/",
        description="S3 prefix for transcript objects",
    )


# ---------------------------------------------------------------------------
# Main Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Main application settings loaded from .env.local."""

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="SBO_",
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # FastAPI Service
    # -----------------------------------------------------------------------
    host: str = Field(default="0.0.0.0", description="Bind address")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port")
    debug: bool = Field(default=False, description="Enable debug mode")
    cors: CorsConfig = Field(default_factory=CorsConfig)

    # -----------------------------------------------------------------------
    # Faster-Whisper - Model
    # -----------------------------------------------------------------------
    whisper_model_size: Literal[
        "base", "small", "medium", "large-v1", "large-v2", "large-v3"
    ] = Field(default="large-v3", description="Whisper model size")
    whisper_model_path: Path | None = Field(default=None, description="Local model path (auto-download if empty)")
    whisper_language: str = Field(default="en", description="Language code (auto-detect if 'auto')")
    whisper_task: Literal["transcribe", "translate"] = Field(default="transcribe", description="Task to perform")

    # -----------------------------------------------------------------------
    # Faster-Whisper - GPU
    # -----------------------------------------------------------------------
    whisper_device: Literal["cuda", "cpu"] = Field(default="cuda", description="Computation device")
    whisper_device_index: int = Field(default=0, ge=0, description="GPU device index")
    whisper_compute_type: Literal["float16", "int8_float16", "int8", "float32"] = Field(
        default="float16", description="Compute type (float16 recommended for 12GB VRAM)"
    )

    # -----------------------------------------------------------------------
    # Faster-Whisper - Sub-configs
    # -----------------------------------------------------------------------
    whisper_inference: WhisperInferenceConfig = Field(default_factory=WhisperInferenceConfig)
    whisper_constraints: WhisperConstraintsConfig = Field(default_factory=WhisperConstraintsConfig)
    whisper_vad: WhisperVadConfig = Field(default_factory=WhisperVadConfig)
    whisper_timestamp: WhisperTimestampConfig = Field(default_factory=WhisperTimestampConfig)

    # -----------------------------------------------------------------------
    # Audio
    # -----------------------------------------------------------------------
    audio: AudioConfig = Field(default_factory=AudioConfig)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    output: OutputConfig = Field(default_factory=OutputConfig)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # -----------------------------------------------------------------------
    # S3 / MinIO Storage
    # -----------------------------------------------------------------------
    storage: S3StorageConfig = Field(default_factory=S3StorageConfig)

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------

    @field_validator("whisper_model_path", mode="before")
    @classmethod
    def resolve_model_path(cls, v: str | None) -> Path | None:
        """Resolve model path to absolute, or None if empty."""
        if not v or v.strip() == "":
            return None
        return Path(v).expanduser().resolve()

    @field_validator("whisper_language", mode="before")
    @classmethod
    def normalize_language(cls, v: str) -> str:
        """Normalize language: empty or 'auto' → 'auto'."""
        if not v or v.strip() == "":
            return "auto"
        return v.strip().lower()

    # -----------------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------------

    @property
    def vad_params(self) -> dict:
        """Return VAD parameters as a dict compatible with faster-whisper."""
        return {
            "threshold": self.whisper_vad.threshold,
            "min_silence_duration_ms": self.whisper_vad.min_silence_duration_ms,
            "speech_pad_ms": self.whisper_vad.speech_pad_ms,
        }

    @property
    def whisper_kwargs(self) -> dict:
        """Build the kwargs dict for Whisper.load_model()."""
        kwargs: dict = {
            "device": self.whisper_device,
            "device_index": self.whisper_device_index,
            "compute_type": self.whisper_compute_type,
        }
        if self.whisper_model_path:
            kwargs["model_path"] = self.whisper_model_path
        return kwargs

    @property
    def run_kwargs(self) -> dict:
        """Build the kwargs dict for model.transcribe() / model.translate()."""
        return {
            "beam_size": self.whisper_inference.beam_size,
            "patience": self.whisper_inference.patience,
            "temperature": self.whisper_inference.temperature,
            "num_workers": self.whisper_inference.num_workers,
            "repetition_penalty": self.whisper_constraints.repetition_penalty,
            "length_ratio": self.whisper_constraints.length_ratio,
            "spm_silence_threshold": self.whisper_constraints.spm_silence_threshold,
            "wer_silence_threshold": self.whisper_constraints.wer_silence_threshold,
            "hotwords": self.whisper_constraints.hotwords or None,
            "ban_token_ids": self.whisper_constraints.ban_token_ids or None,
            "allow_punctuation_tokens": self.whisper_constraints.allow_punctuation_tokens or None,
            "vad_filter": self.whisper_vad.enabled,
            "max_initial_timestamp": self.whisper_timestamp.max_initial_timestamp,
            "one_segment": self.whisper_timestamp.one_segment,
            "segment_duration": self.whisper_timestamp.segment_duration or None,
        }

    # -----------------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║                   SBOBINATOR CONFIG                      ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Host:      {self.host:>26} ║",
            f"║  Port:      {self.port:>26} ║",
            f"║  Debug:     {str(self.debug):>26} ║",
            f"║  CORS:      {', '.join(self.cors.origins):>11} ║",
            "╠──────────────────────────────────────────────────────────╣",
            f"║  Model:     {self.whisper_model_size:>26} ║",
            f"║  Path:      {str(self.whisper_model_path or 'auto'):>26} ║",
            f"║  Language:  {self.whisper_language:>26} ║",
            f"║  Task:      {self.whisper_task:>26} ║",
            "╠──────────────────────────────────────────────────────────╣",
            f"║  Device:    {self.whisper_device:>26} ║",
            f"║  GPU Index: {self.whisper_device_index:>26} ║",
            f"║  Compute:   {self.whisper_compute_type:>26} ║",
            "╠──────────────────────────────────────────────────────────╣",
            f"║  Beam Size: {self.whisper_inference.beam_size:>26} ║",
            f"║  Patience:  {self.whisper_inference.patience:>26} ║",
            f"║  Temp:      {self.whisper_inference.temperature:>26} ║",
            f"║  Workers:   {self.whisper_inference.num_workers:>26} ║",
            "╠──────────────────────────────────────────────────────────╣",
            f"║  VAD:       {'ON' if self.whisper_vad.enabled else 'OFF':>26} ║",
            f"║  VAD Method:{self.whisper_vad.method:>26} ║",
            "╠──────────────────────────────────────────────────────────╣",
            f"║  Audio SR:  {self.audio.sample_rate:>26} ║",
            f"║  Chunk:     {self.audio.chunk_duration}s{' (unlimited)' if self.audio.max_duration == 0 else f' (max {self.audio.max_duration}s)':>13} ║",
            f"║  Noise Red: {'ON' if self.audio.noise_reduction else 'OFF':>26} ║",
            "╠──────────────────────────────────────────────────────────╣",
            f"║  Output:    {self.output.format:>26} ║",
            f"║  Save:      {'ON' if self.output.save_transcripts else 'OFF':>26} ║",
            f"║  Dir:       {str(self.output.transcripts_dir):>26} ║",
            "╠──────────────────────────────────────────────────────────╣",
            f"║  Endpoint:  {self.storage.endpoint_url:>26} ║",
            f"║  Bucket:    {self.storage.bucket:>26} ║",
            f"║  Region:    {self.storage.region:>26} ║",
            f"║  Secure:    {'HTTPS' if self.storage.secure else 'HTTP':>26} ║",
            "╠──────────────────────────────────────────────────────────╣",
            f"║  Log Level: {self.logging.level:>26} ║",
            f"║  Log Format:{self.logging.format:>26} ║",
            "╚══════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    @property
    def storage_endpoint(self) -> str:
        """Build the full storage endpoint URL with scheme."""
        if self.storage.endpoint_url.startswith(("http://", "https://")):
            return self.storage.endpoint_url
        scheme = "https" if self.storage.secure else "http"
        return f"{scheme}://{self.storage.endpoint_url}"

    def model_dump_env(self) -> dict[str, str]:
        """Export all settings as flat key-value pairs suitable for .env files."""
        data = self.model_dump()
        flat: dict[str, str] = {}

        def _flatten(obj: dict, prefix: str = "") -> None:
            for key, value in obj.items():
                env_key = f"{prefix}_{key.upper()}" if prefix else key.upper()
                if isinstance(value, dict):
                    _flatten(value, env_key)
                elif isinstance(value, bool):
                    flat[env_key] = str(value).lower()
                elif isinstance(value, Path):
                    flat[env_key] = str(value)
                elif isinstance(value, list):
                    flat[env_key] = json.dumps(value)
                else:
                    flat[env_key] = str(value)

        _flatten(data)
        return flat


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


# Lazy-loaded singleton
_settings: Settings | None = None


def settings() -> Settings:
    """Get the application settings (singleton with lazy loading)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
