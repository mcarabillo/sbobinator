"""Audio preprocessing for Whisper-ready input.

Takes raw audio bytes (from S3/MinIO) and returns a cleaned, mono, 16 kHz
float32 numpy array ready for transcription.

All parameters are read from ``settings().audio``.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from dataclasses import dataclass

import ffmpeg
import noisereduce as nr
import numpy as np
import soundfile as sf

from utils.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Silence threshold in dBFS below which audio is considered silence
SILENCE_THRESHOLD_DBFS = -45.0

# Peak normalization target (leave 3 dB headroom)
PEAK_NORMALIZATION_TARGET = 0.97

# Format → soundfile subtype mapping (native support)
_NATIVE_SUBTYPE_MAP: dict[str, str] = {
    "wav": "FLOAT",
    "flac": "FLAC",
    "ogg": "OGG",
}

# Formats that must go through ffmpeg (soundfile can't read them)
_FFMPEG_ONLY_FORMATS: set[str] = {"mp3", "m4a", "mp4", "webm"}

# MIME types we handle
_SUPPORTED_MIMES: set[str] = {
    "audio/wav",
    "audio/x-wav",
    "audio/flac",
    "audio/x-flac",
    "audio/ogg",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "video/mp4",
    "video/webm",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessedAudio:
    """Immutable result of the audio preprocessing pipeline."""

    data: np.ndarray
    """Clean audio samples (float32, mono, range [-1.0, 1.0])."""
    sample_rate: int
    """Sample rate in Hz (always 16000 for Whisper compatibility)."""
    duration: float
    """Duration in seconds."""
    original_format: str
    """Original file format (e.g. ``'mp3'``, ``'wav'``)."""
    original_sample_rate: int
    """Sample rate of the source file."""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AudioProcessingError(Exception):
    """Raised when audio processing fails."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _guess_format(content_type: str, filename: str | None = None) -> str:
    """Determine the audio format from MIME type or filename extension."""
    mime = content_type.lower().split(";")[0].strip()

    mime_to_fmt = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/flac": "flac",
        "audio/x-flac": "flac",
        "audio/ogg": "ogg",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/mp4": "m4a",
        "audio/x-m4a": "m4a",
        "video/mp4": "m4a",
        "video/webm": "webm",
    }

    if mime in mime_to_fmt:
        return mime_to_fmt[mime]

    # Fallback to filename extension
    if filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in _NATIVE_SUBTYPE_MAP or ext in _FFMPEG_ONLY_FORMATS:
            return ext

    raise AudioProcessingError(
        f"Unsupported audio format: content_type='{content_type}', "
        f"filename='{filename}'"
    )


def _to_mono(data: np.ndarray) -> np.ndarray:
    """Downmix multi-channel audio to mono by averaging channels."""
    if data.ndim == 1:
        return data
    return data.mean(axis=1)


def _estimate_noise_profile(
    data: np.ndarray, sr: int, frame_length: float = 0.5
) -> np.ndarray:
    """Estimate noise profile from the beginning of the audio.

    Uses the first *frame_length* seconds as a noise-only segment.
    If the audio is shorter, uses the entire signal.
    """
    frame_samples = int(sr * frame_length)
    noise_segment = data[: min(frame_samples, len(data))]
    return noise_segment


# ---------------------------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------------------------


def load_audio(data: bytes, fmt: str) -> tuple[np.ndarray, int]:
    """Decode raw audio bytes into a mono float32 numpy array.

    Uses ``soundfile`` for natively supported formats (wav, flac, ogg).
    Falls back to ``ffmpeg`` for formats like mp3, m4a, webm.

    Parameters
    ----------
    data:
        Raw audio bytes.
    fmt:
        Audio format string (``'wav'``, ``'mp3'``, ``'flac'``, etc.).

    Returns
    -------
    samples: np.ndarray
        Audio samples (float32, mono).
    sample_rate: int
        Original sample rate.

    Raises
    ------
    AudioProcessingError
        If the format is unsupported or decoding fails.
    """
    # --- Native soundfile path (wav, flac, ogg) ---
    if fmt in _NATIVE_SUBTYPE_MAP:
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            samples, sr = sf.read(tmp_path, dtype="float32")
            return samples, sr
        except sf.LibsndfileError as exc:
            raise AudioProcessingError(
                f"Failed to decode audio ({fmt}): {exc}"
            ) from exc
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # --- FFmpeg fallback (mp3, m4a, mp4, webm) ---
    if fmt in _FFMPEG_ONLY_FORMATS:
        try:
            process = (
                ffmpeg
                .input("pipe:")
                .output(
                    "pipe:",
                    format="s16le",
                    acodec="pcm_s16le",
                    ac=1,
                    ar="16000",
                )
                .run(
                    input=data,
                    capture_stdout=True,
                    capture_stderr=True,
                )
            )
            stdout = process[0]
            samples = np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32768.0
            sr = 16000
            return samples, sr
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            raise AudioProcessingError(
                f"FFmpeg failed to decode audio ({fmt}): {stderr}"
            ) from exc

    raise AudioProcessingError(f"Unsupported format for decoding: '{fmt}'")


def convert_to_mono(data: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono for Whisper compatibility."""
    mono = _to_mono(data)
    if mono.ndim != 1:
        raise AudioProcessingError(
            f"Expected 1-D array after mono conversion, got {mono.ndim}-D"
        )
    return mono


def resample(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample audio to *dst_sr* using scipy.signal.

    Parameters
    ----------
    data:
        Audio samples.
    src_sr:
        Source sample rate.
    dst_sr:
        Target sample rate (typically 16000).
    """
    if src_sr == dst_sr:
        return data

    ratio = dst_sr / src_sr
    new_length = int(len(data) * ratio)

    if new_length <= 0:
        raise AudioProcessingError(
            f"Resampling would produce zero samples "
            f"(src_sr={src_sr}, dst_sr={dst_sr}, len={len(data)})"
        )

    from scipy.signal import resample as scipy_resample

    return np.asarray(scipy_resample(data, new_length), dtype=np.float32)


def remove_noise(
    data: np.ndarray,
    sr: int,
    frame_length: float = 0.5,
) -> np.ndarray:
    """Apply spectral noise reduction.

    Parameters
    ----------
    data:
        Audio samples (mono, float32).
    sr:
        Sample rate.
    frame_length:
        Duration in seconds used for noise estimation (first N seconds).

    Returns
    -------
    cleaned: np.ndarray
        Noise-reduced audio samples.
    """
    noise_profile = _estimate_noise_profile(data, sr, frame_length)

    try:
        cleaned = nr.reduce_noise(
            y=data,
            sr=sr,
            y_noise=noise_profile,
            prop_decrease=0.75,
            stationary=True,
        )
    except Exception as exc:
        logger.warning("Noise reduction failed, returning original: %s", exc)
        return data

    return cleaned.astype(np.float32)


def trim_audio(data: np.ndarray, sr: int, max_duration: int) -> np.ndarray:
    """Trim audio to *max_duration* seconds.

    If ``max_duration`` is 0 or negative, no trimming is applied.
    """
    if max_duration <= 0:
        return data

    max_samples = sr * max_duration
    if len(data) > max_samples:
        logger.info(
            "Trimming audio from %.1fs to %ds",
            len(data) / sr,
            max_duration,
        )
        return data[:max_samples]
    return data


def remove_silence(data: np.ndarray, sr: int) -> np.ndarray:
    """Remove leading and trailing silence based on dBFS threshold.

    Silence is defined as any sample whose absolute amplitude is below
    ``10^(SILENCE_THRESHOLD_DBFS / 20)``.
    """
    threshold = 10 ** (SILENCE_THRESHOLD_DBFS / 20)

    # Find first non-silent sample
    start = 0
    for i in range(len(data)):
        if abs(data[i]) > threshold:
            start = i
            break
    else:
        raise AudioProcessingError("Audio is entirely silent after processing")

    # Find last non-silent sample
    end = len(data)
    for i in range(len(data) - 1, -1, -1):
        if abs(data[i]) > threshold:
            end = i + 1
            break

    trimmed = data[start:end]

    if len(trimmed) == 0:
        raise AudioProcessingError("Audio became empty after silence removal")

    if start > 0 or end < len(data):
        logger.info(
            "Removed silence: %.2fs → %.2fs (start=%.2fs, end=%.2fs)",
            len(data) / sr,
            len(trimmed) / sr,
            start / sr,
            (len(data) - end) / sr,
        )

    return trimmed


def normalize_peak(data: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to *PEAK_NORMALIZATION_TARGET*.

    Leaves the signal unchanged if the peak is already below the target.
    """
    peak = np.max(np.abs(data))
    if peak == 0:
        return data

    if peak < PEAK_NORMALIZATION_TARGET:
        return data

    return data * (PEAK_NORMALIZATION_TARGET / peak)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class AudioProcessor:
    """Complete audio preprocessing pipeline.

    Reads configuration from ``settings().audio``.

    Parameters
    ----------
    config:
        ``AudioConfig`` instance. Defaults to ``settings().audio``.
    """

    def __init__(self, config=None) -> None:
        self.config = config or settings().audio

    def process(
        self,
        raw_bytes: bytes,
        content_type: str,
        filename: str | None = None,
    ) -> ProcessedAudio:
        """Run the full preprocessing pipeline.

        Parameters
        ----------
        raw_bytes:
            Raw audio bytes from S3/MinIO.
        content_type:
            MIME type of the audio file.
        filename:
            Optional original filename (used for format detection fallback).

        Returns
        -------
        ProcessedAudio
            Cleaned, mono, 16 kHz audio ready for Whisper.

        Raises
        ------
        AudioProcessingError
            If any processing step fails.
        """
        # 1. Detect format
        fmt = _guess_format(content_type, filename)
        logger.info("Detected audio format: %s [%s]", fmt, content_type)

        # 2. Decode
        data, src_sr = load_audio(raw_bytes, fmt)
        logger.info("Decoded audio: %d samples @ %d Hz, channels=%d", len(data), src_sr, data.shape[-1] if data.ndim > 1 else 1)

        # 3. Mono
        data = convert_to_mono(data)
        logger.info("Converted to mono: %d samples", len(data))

        # 4. Resample
        target_sr = self.config.sample_rate
        data = resample(data, src_sr, target_sr)
        logger.info("Resampled to %d Hz: %d samples", target_sr, len(data))

        # 5. Noise reduction (optional)
        if self.config.noise_reduction:
            logger.info("Applying noise reduction...")
            data = remove_noise(
                data,
                target_sr,
                frame_length=self.config.noise_reduction_frame_length,
            )

        # 6. Trim
        data = trim_audio(data, target_sr, self.config.max_duration)
        logger.info("After trim: %.2fs", len(data) / target_sr)

        # 7. Silence removal
        data = remove_silence(data, target_sr)

        # 8. Peak normalization
        data = normalize_peak(data)

        duration = len(data) / target_sr

        logger.info(
            "Preprocessing complete: %.2fs, %.3f dBFS peak",
            duration,
            20 * np.log10(np.max(np.abs(data)) + 1e-10),
        )

        return ProcessedAudio(
            data=data,
            sample_rate=target_sr,
            duration=duration,
            original_format=fmt,
            original_sample_rate=src_sr,
        )
