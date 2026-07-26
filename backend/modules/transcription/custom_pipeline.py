"""Custom transcription pipeline with silence-based chunking and parallel processing.

Divides audio into chunks based on silence boundaries, transcribes each chunk
in parallel, and merges results in order.

Usage
-----
    from backend.modules.transcription.custom_pipeline import CustomPipeline
    from backend.modules.preprocessing.audio_processor import ProcessedAudio

    processor = CustomPipeline()
    result = processor.transcribe(processed_audio)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptionSegment:
    """A single segment from Whisper output."""

    start: float
    end: float
    text: str
    language: str | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    """Complete transcription result."""

    segments: list[TranscriptionSegment]
    text: str
    duration: float
    language: str


@dataclass(frozen=True)
class AudioChunk:
    """A chunk of audio with metadata for parallel transcription."""

    chunk_id: int
    data: np.ndarray
    start_offset: float
    end_offset: float
    duration: float


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TranscriptionError(Exception):
    """Raised when transcription fails."""


# ---------------------------------------------------------------------------
# Silence-based chunking
# ---------------------------------------------------------------------------


def _find_speech_regions(
    data: np.ndarray,
    sr: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
) -> list[tuple[int, int]]:
    """Find speech regions separated by silence.

    Parameters
    ----------
    data:
        Audio samples (float32, mono).
    sr:
        Sample rate (16000).
    min_silence_duration_ms:
        Minimum silence duration in milliseconds to consider as a boundary.
    speech_pad_ms:
        Padding in milliseconds added before and after each speech region.

    Returns
    -------
    regions:
        List of (start_sample, end_sample) tuples for each speech region.
    """
    threshold = 10 ** (-45.0 / 20)  # -45 dBFS
    min_silence_samples = int(sr * min_silence_duration_ms / 1000)
    pad_samples = int(sr * speech_pad_ms / 1000)

    silence_mask = np.abs(data) < threshold

    regions: list[tuple[int, int]] = []
    speech_start: int | None = None

    for i in range(len(data)):
        if not silence_mask[i]:
            # Speech detected
            if speech_start is None:
                speech_start = i
        else:
            # Silence detected — end of speech region
            if speech_start is not None:
                speech_end = i
                # Apply padding
                region_start = max(speech_start - pad_samples, 0)
                region_end = min(speech_end + pad_samples, len(data))
                regions.append((region_start, region_end))
                speech_start = None

    # Handle trailing speech (no trailing silence)
    if speech_start is not None:
        region_start = max(speech_start - pad_samples, 0)
        regions.append((region_start, len(data)))

    logger.info(
        "Found %d speech regions (total %.1fs / %.1fs)",
        len(regions),
        sum(end - start for start, end in regions) / sr,
        len(data) / sr,
    )

    return regions


def _create_chunks(
    data: np.ndarray,
    sr: int,
    regions: list[tuple[int, int]],
) -> list[AudioChunk]:
    """Create AudioChunk objects from speech regions.

    Parameters
    ----------
    data:
        Full audio samples.
    sr:
        Sample rate.
    regions:
        List of (start_sample, end_sample) tuples.

    Returns
    -------
    chunks:
        List of AudioChunk objects with incremental IDs.
    """
    chunks: list[AudioChunk] = []
    for chunk_id, (start, end) in enumerate(regions):
        chunk_data = data[start:end]
        duration = (end - start) / sr
        start_offset = start / sr
        end_offset = end / sr

        chunks.append(AudioChunk(
            chunk_id=chunk_id,
            data=chunk_data,
            start_offset=start_offset,
            end_offset=end_offset,
            duration=duration,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Transcription worker
# ---------------------------------------------------------------------------


def _transcribe_chunk(
    chunk: AudioChunk,
    model_path: str,
    device: str,
    device_index: int,
    compute_type: str,
    task: str,
    language: str | None,
    beam_size: int,
    patience: float,
    temperature: float,
    repetition_penalty: float,
    length_ratio: float,
    hotwords: str | None,
    ban_token_ids: list[int] | None,
    max_initial_timestamp: int,
    one_segment: bool,
    segment_duration: int | None,
    vad_filter: bool,
    vad_params: dict[str, Any],
) -> list[TranscriptionSegment]:
    """Transcribe a single audio chunk.

    This function runs in a worker thread and uses the standard
    ``WhisperModel.transcribe()`` API.

    Parameters
    ----------
    chunk:
        AudioChunk to transcribe.
    model_path:
        Whisper model path or size string.
    device:
        Computation device (cuda/cpu).
    device_index:
        GPU device index.
    compute_type:
        Compute type (float16, int8, etc.).
    task:
        Task type (transcribe/translate).
    language:
        Language code or None for auto-detect.
    beam_size:
        Beam search size.
    patience:
        Beam search patience.
    temperature:
        Sampling temperature.
    repetition_penalty:
        Repetition penalty.
    length_ratio:
        Output/input length ratio.
    hotwords:
        Hotwords for prompt conditioning.
    ban_token_ids:
        Token IDs to ban.
    max_initial_timestamp:
        Max initial timestamp.
    one_segment:
        Force single segment.
    segment_duration:
        Max segment duration.
    vad_filter:
        Enable VAD filtering.
    vad_params:
        VAD parameters dict.

    Returns
    -------
    segments:
        List of TranscriptionSegment for this chunk.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_path,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
    )

    kwargs: dict[str, Any] = {
        "beam_size": beam_size,
        "patience": patience,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "length_ratio": length_ratio,
        "hotwords": hotwords,
        "ban_token_ids": ban_token_ids,
        "max_initial_timestamp": max_initial_timestamp,
        "one_segment": one_segment,
        "segment_duration": segment_duration,
        "vad_filter": vad_filter,
        "vad_parameters": vad_params,
    }

    if language:
        kwargs["language"] = language
    if task:
        kwargs["task"] = task

    segments, info = model.transcribe(chunk.data, **kwargs)

    result_segments: list[TranscriptionSegment] = []
    for seg in segments:
        # Adjust timestamps to global audio timeline
        adjusted_start = chunk.start_offset + seg.start
        adjusted_end = chunk.start_offset + seg.end

        result_segments.append(TranscriptionSegment(
            start=adjusted_start,
            end=adjusted_end,
            text=seg.text.strip(),
            language=(
                info.language
                if info.language_probability > 0.5
                else None
            ),
        ))

    close = getattr(model, "close", None)
    if callable(close):
        close()

    return result_segments


# ---------------------------------------------------------------------------
# CustomPipeline
# ---------------------------------------------------------------------------


class CustomPipeline:
    """Transcription pipeline with silence-based chunking and parallel processing.

    1. Finds speech regions using silence boundaries
    2. Creates chunks with incremental IDs
    3. Transcribes chunks in parallel
    4. Merges results in chunk_id order

    Parameters
    ----------
    config:
        Settings instance. Defaults to ``settings()``.
    """

    def __init__(self, config=None) -> None:
        self.config = config or settings()

    def transcribe(
        self,
        audio_data: np.ndarray,
        sr: int = 16000,
        audio_duration: float | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio using silence-based chunking and parallel processing.

        Parameters
        ----------
        audio_data:
            Audio samples (float32, mono, 16 kHz).
        sr:
            Sample rate.
        audio_duration:
            Total duration of the original audio.

        Returns
        -------
        TranscriptionResult
            Complete transcription with merged segments.
        """
        audio_duration = audio_duration or len(audio_data) / sr

        # --- Step 1: Find speech regions ---
        min_silence_ms = self.config.whisper_vad.min_silence_duration_ms
        speech_pad_ms = self.config.whisper_vad.speech_pad_ms

        regions = _find_speech_regions(
            audio_data, sr, min_silence_ms, speech_pad_ms
        )

        if not regions:
            logger.warning("No speech regions found — transcribing entire audio")
            regions = [(0, len(audio_data))]

        # --- Step 2: Create chunks ---
        chunks = _create_chunks(audio_data, sr, regions)
        num_workers = self.config.whisper_inference.num_workers or 1

        logger.info(
            "Created %d chunks from %d regions, using %d workers",
            len(chunks),
            len(regions),
            num_workers,
        )

        # --- Step 3: Transcribe chunks in parallel ---
        start_time = time.time()

        try:
            all_segments: list[TranscriptionSegment] = []

            if num_workers <= 1:
                # Sequential transcription
                for chunk in chunks:
                    segments = _transcribe_chunk(chunk, **self._get_worker_kwargs())
                    all_segments.extend(segments)
            else:
                # Parallel transcription
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(
                            _transcribe_chunk, chunk, **self._get_worker_kwargs()
                        ): chunk.chunk_id
                        for chunk in chunks
                    }

                    for future in as_completed(futures):
                        chunk_id = futures[future]
                        try:
                            segments = future.result()
                            all_segments.extend(segments)
                        except Exception as exc:
                            logger.error(
                                "Chunk %d transcription failed: %s", chunk_id, exc
                            )
                            raise

            # --- Step 4: Sort by segment start time ---
            all_segments.sort(key=lambda s: s.start)

            full_text = " ".join(s.text for s in all_segments if s.text)
            language = (
                all_segments[0].language
                if all_segments
                and all_segments[0].language
                else "unknown"
            )

            elapsed = time.time() - start_time
            rt = elapsed / audio_duration if audio_duration > 0 else 0

            logger.info(
                "Custom transcription complete: %d segments from %d chunks in %.1fs (%.2fx RT)",
                len(all_segments),
                len(chunks),
                elapsed,
                rt,
            )

            return TranscriptionResult(
                segments=all_segments,
                text=full_text,
                duration=audio_duration,
                language=language,
            )

        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error(
                "Custom transcription failed after %.1fs: %s", elapsed, exc
            )
            raise TranscriptionError(f"Custom transcription failed: {exc}") from exc

    def _get_worker_kwargs(self) -> dict[str, Any]:
        """Build kwargs dict for the transcribe_chunk worker function."""
        hotwords = (
            " ".join(self.config.whisper_constraints.hotwords)
            if self.config.whisper_constraints.hotwords
            else None
        )

        return {
            "model_path": self.config.whisper_model_size,
            "device": self.config.whisper_device,
            "device_index": self.config.whisper_device_index,
            "compute_type": self.config.whisper_compute_type,
            "task": self.config.whisper_task,
            "language": (
                self.config.whisper_language
                if self.config.whisper_language != "auto"
                else None
            ),
            "beam_size": self.config.whisper_inference.beam_size,
            "patience": self.config.whisper_inference.patience,
            "temperature": self.config.whisper_inference.temperature,
            "repetition_penalty": self.config.whisper_constraints.repetition_penalty,
            "length_ratio": self.config.whisper_constraints.length_ratio,
            "hotwords": hotwords,
            "ban_token_ids": (
                self.config.whisper_constraints.ban_token_ids or None
            ),
            "max_initial_timestamp": self.config.whisper_timestamp.max_initial_timestamp,
            "one_segment": self.config.whisper_timestamp.one_segment,
            "segment_duration": self.config.whisper_timestamp.segment_duration,
            "vad_filter": self.config.whisper_vad.enabled,
            "vad_params": {
                "threshold": self.config.whisper_vad.threshold,
                "min_silence_duration_ms": self.config.whisper_vad.min_silence_duration_ms,
                "speech_pad_ms": self.config.whisper_vad.speech_pad_ms,
            },
        }
