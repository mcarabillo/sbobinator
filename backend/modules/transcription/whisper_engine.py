"""Whisper transcription engine — orchestrator.

Selects and instantiates the appropriate transcription pipeline based on
the ``bip`` setting:

- ``bip=True``  → ``BatchedPipeline`` (uses BatchedInferencePipeline)
- ``bip=False`` → ``CustomPipeline`` (silence-based chunking + parallel)

All configuration is read from ``settings()``.

Usage
-----
    from backend.modules.transcription.whisper_engine import TranscriptionEngine

    engine = TranscriptionEngine()
    result = engine.transcribe(processed_audio)
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

from utils.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-export data classes and exceptions
# ---------------------------------------------------------------------------

from .batched_pipeline import (
    TranscriptionSegment,
    TranscriptionResult,
    TranscriptionError,
)

__all__ = [
    "TranscriptionEngine",
    "TranscriptionSegment",
    "TranscriptionResult",
    "TranscriptionError",
]


# ---------------------------------------------------------------------------
# TranscriptionEngine
# ---------------------------------------------------------------------------


class TranscriptionEngine:
    """Transcription engine factory.

    Reads ``settings().whisper_inference.bip`` to select the pipeline:

    - ``True``  → ``BatchedPipeline`` (BatchedInferencePipeline)
    - ``False`` → ``CustomPipeline`` (silence-based chunking + parallel)

    Parameters
    ----------
    config:
        Settings instance. Defaults to ``settings()``.
    """

    def __init__(self, config=None) -> None:
        self.config = config or settings()
        self.use_bip = self.config.whisper_inference.bip

        if self.use_bip:
            logger.info("Using BatchedInferencePipeline (bip=True)")
            from .batched_pipeline import BatchedPipeline

            self._pipeline = BatchedPipeline(config=self.config)
        else:
            logger.info("Using CustomPipeline with silence-based chunking (bip=False)")
            from .custom_pipeline import CustomPipeline

            self._pipeline = CustomPipeline(config=self.config)

    def transcribe(
        self,
        audio_data: np.ndarray,
        sr: int = 16000,
        audio_duration: float | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio using the selected pipeline.

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
        return cast(
            TranscriptionResult,
            self._pipeline.transcribe(
                audio_data,
                sr=sr,
                audio_duration=audio_duration,
            ),
        )

    def close(self) -> None:
        """Release pipeline resources."""
        if hasattr(self, "_pipeline"):
            close = getattr(self._pipeline, "close", None)
            if callable(close):
                close()
            logger.info("Transcription engine released")
