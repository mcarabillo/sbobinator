"""Whisper transcription modules.

Exports
-------
- ``TranscriptionEngine`` — orchestrator (selects batched or custom pipeline)
- ``TranscriptionSegment`` — single transcription segment
- ``TranscriptionResult`` — complete transcription result
- ``TranscriptionError`` — exception for transcription failures
"""

from .whisper_engine import (
    TranscriptionEngine,
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
