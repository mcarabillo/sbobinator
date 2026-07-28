"""Batched inference pipeline using faster-whisper's BatchedInferencePipeline.

Passes the full processed audio to the pipeline which handles chunking and VAD
internally via Silero VAD.

Usage
-----
    from backend.modules.transcription.batched_pipeline import BatchedPipeline
    from backend.modules.preprocessing.audio_processor import ProcessedAudio

    processor = BatchedPipeline()
    result = processor.transcribe(processed_audio)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from utils.config import settings

logger = logging.getLogger(__name__)


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


class BatchedPipeline:
    """Transcription pipeline using ``BatchedInferencePipeline``.

    Passes the full audio to a single pipeline instance which handles
    chunking internally via Silero VAD and batched decoding via ``batch_size``.

    Parameters
    ----------
    config:
        Settings instance. Defaults to ``settings()``.
    """

    def __init__(self, config=None) -> None:
        self.config = config or settings()
        self.model_size = self.config.whisper_model_size or "base"

        logger.info(
            "Loading Whisper model: %s on %s (compute=%s)",
            self.model_size,
            self.config.whisper_device,
            self.config.whisper_compute_type,
        )

        from faster_whisper import BatchedInferencePipeline, WhisperModel

        model = WhisperModel(
            model_size_or_path=self.model_size,
            device=self.config.whisper_device,
            device_index=self.config.whisper_device_index,
            compute_type=self.config.whisper_compute_type,
        )

        self.pipeline = BatchedInferencePipeline(model=model)

        logger.info("BatchedInferencePipeline loaded successfully")

    def transcribe(
        self,
        audio_data: np.ndarray,
        sr: int = 16000,
        audio_duration: float | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio using batched inference.

        The full audio is passed to ``BatchedInferencePipeline.transcribe()``
        which internally:
        1. Splits audio into chunks using Silero VAD
        2. Batches the chunks for parallel GPU decoding
        3. Restores timestamps to the original audio timeline

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

        batch_size = self.config.whisper_inference.batch_size
        chunk_length = self.config.whisper_timestamp.segment_duration or 30

        logger.info(
            "Transcribing %.2fs audio with BatchedInferencePipeline "
            "(batch_size=%d, chunk_length=%ds)",
            audio_duration,
            batch_size,
            chunk_length,
        )

        start_time = time.time()

        try:
            segments_gen, info = self.pipeline.transcribe(
                audio_data,
                task=self.config.whisper_task,
                language=(
                    self.config.whisper_language
                    if self.config.whisper_language != "auto"
                    else None
                ),
                beam_size=self.config.whisper_inference.beam_size,
                patience=self.config.whisper_inference.patience,
                temperature=self.config.whisper_inference.temperature,
                chunk_length=chunk_length,
                batch_size=batch_size,
                repetition_penalty=self.config.whisper_constraints.repetition_penalty,
                length_penalty=self.config.whisper_constraints.length_ratio,
                hotwords=(
                    " ".join(self.config.whisper_constraints.hotwords)
                    if self.config.whisper_constraints.hotwords
                    else None
                ),
                suppress_tokens=(
                    self.config.whisper_constraints.ban_token_ids or None
                ),
                without_timestamps=False,
                word_timestamps=False,
                max_initial_timestamp=self.config.whisper_timestamp.max_initial_timestamp,
                condition_on_previous_text=False,
                multilingual=(
                    False
                    if self.config.whisper_language != "auto"
                    else True
                ),
                language_detection_threshold=0.5,
                language_detection_segments=5,
                vad_filter=True,
                vad_parameters=self.config.vad_params,
                log_progress=False,
            )

            all_segments: list[TranscriptionSegment] = []
            for seg in segments_gen:
                all_segments.append(TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    language=(
                        info.language
                        if info.language_probability > 0.5
                        else None
                    ),
                ))

            full_text = " ".join(s.text for s in all_segments if s.text)
            language = (
                info.language if info.language_probability > 0.5 else "unknown"
            )

            elapsed = time.time() - start_time
            rt = elapsed / audio_duration if audio_duration > 0 else 0

            logger.info(
                "Batched transcription complete: %d segments in %.1fs (%.2fx RT)",
                len(all_segments),
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
                "Batched transcription failed after %.1fs: %s", elapsed, exc
            )
            raise TranscriptionError(f"Batched transcription failed: {exc}") from exc

    def close(self) -> None:
        """Release resources."""
        if hasattr(self, "pipeline"):
            del self.pipeline
            logger.info("BatchedInferencePipeline released")


class TranscriptionError(Exception):
    """Raised when transcription fails."""
