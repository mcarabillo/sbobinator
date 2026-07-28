"""Transcription output formatter.

Generates transcription files in various formats (SRT, VTT, TXT, CSV, TSV, JSON)
from Whisper segments. It provides timestamps for each segment and various other metadata.

Usage
-----
    from backend.modules.output.formatter import OutputFormatter

    formatter = OutputFormatter()
    content = formatter.format(
        segments=segments,
        fmt="srt",
        metadata={"language": "it", "duration": 120.0},
    )
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.modules.transcription.batched_pipeline import TranscriptionSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FORMAT_EXTENSIONS: dict[str, str] = {
    "json": ".json",
    "srt": ".srt",
    "vtt": ".vtt",
    "txt": ".txt",
    "csv": ".csv",
    "tsv": ".tsv",
}

_MIME_TYPES: dict[str, str] = {
    "json": "application/json",
    "srt": "text/srt",
    "vtt": "text/vtt",
    "txt": "text/plain",
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _format_srt(segments: list[TranscriptionSegment]) -> str:
    """Format as SRT subtitle file."""
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(
            f"{_format_timestamp_srt(seg.start)} --> {_format_timestamp_srt(seg.end)}"
        )
        lines.append(seg.text)
        lines.append("")  # blank line between segments

    return "\n".join(lines)


def _format_vtt(segments: list[TranscriptionSegment]) -> str:
    """Format as WebVTT subtitle file."""
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        lines.append(
            f"{_format_timestamp_vtt(seg.start)} --> {_format_timestamp_vtt(seg.end)}"
        )
        lines.append(seg.text)
        lines.append("")

    return "\n".join(lines)


def _format_txt(segments: list[TranscriptionSegment]) -> str:
    """Format as plain text (all segments concatenated)."""
    return " ".join(seg.text for seg in segments)


def _format_csv(segments: list[TranscriptionSegment], include_header: bool = True) -> str:
    """Format as CSV (start, end, text)."""
    lines: list[str] = []
    if include_header:
        lines.append("start,end,text")
    for seg in segments:
        # Escape commas and quotes in text
        text = seg.text.replace('"', '""')
        lines.append(f'{seg.start:.2f},{seg.end:.2f},"{text}"')
    return "\n".join(lines)


def _format_tsv(segments: list[TranscriptionSegment], include_header: bool = True) -> str:
    """Format as TSV (start, end, text)."""
    lines: list[str] = []
    if include_header:
        lines.append("start\tend\ttext")
    for seg in segments:
        lines.append(f"{seg.start:.2f}\t{seg.end:.2f}\t{seg.text}")
    return "\n".join(lines)


def _format_json(
    segments: list[TranscriptionSegment],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Format as JSON with optional metadata."""
    data: dict[str, Any] = {
        "segments": [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in segments
        ],
        "text": " ".join(seg.text for seg in segments),
        "segment_count": len(segments),
    }
    if metadata:
        data["metadata"] = metadata
    return json.dumps(data, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class OutputFormatter:
    """Generates transcription output files in various formats.

    Parameters
    ----------
    formats:
        Supported output formats. Defaults to all.
    """

    def __init__(self, formats: list[str] | None = None) -> None:
        self.formats = formats or list(_FORMAT_EXTENSIONS.keys())

    def format(
        self,
        segments: list[TranscriptionSegment],
        fmt: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Format segments into the requested output format.

        Parameters
        ----------
        segments:
            Transcription segments from Whisper.
        fmt:
            Output format (``json``, ``srt``, ``vtt``, ``txt``, ``csv``, ``tsv``).
        metadata:
            Optional metadata to include in the output (for JSON format).

        Returns
        -------
        str
            Formatted transcription content.

        Raises
        ------
        ValueError
            If the format is not supported.
        """
        fmt = fmt.lower()

        if fmt not in self.formats:
            raise ValueError(
                f"Unsupported format: '{fmt}'. Supported: {', '.join(self.formats)}"
            )

        formatters = {
            "srt": _format_srt,
            "vtt": _format_vtt,
            "txt": _format_txt,
            "csv": _format_csv,
            "tsv": _format_tsv,
            "json": lambda segs: _format_json(segs, metadata),
        }

        return formatters[fmt](segments)

    def get_extension(self, fmt: str) -> str:
        """Get file extension for a format."""
        fmt = fmt.lower()
        if fmt not in _FORMAT_EXTENSIONS:
            raise ValueError(f"Unknown format: {fmt}")
        return _FORMAT_EXTENSIONS[fmt]

    def get_mime_type(self, fmt: str) -> str:
        """Get MIME type for a format."""
        fmt = fmt.lower()
        if fmt not in _MIME_TYPES:
            raise ValueError(f"Unknown format: {fmt}")
        return _MIME_TYPES[fmt]

    def build_output_key(self, input_key: str, fmt: str) -> str:
        """Build the output file key from the input key.

        Replaces the input file extension with the output format extension,
        keeping the same directory path.

        Examples
        --------
        >>> formatter.build_output_key("audio/uploads/podcast.mp3", "srt")
        'audio/uploads/podcast.srt'

        >>> formatter.build_output_key("audio/uploads/podcast.mp3", "json")
        'audio/uploads/podcast.json'
        """
        fmt = fmt.lower()
        ext = self.get_extension(fmt)

        # Replace the last extension
        parts = input_key.rsplit(".", 1)
        if len(parts) == 2:
            return f"{parts[0]}{ext}"
        return f"{input_key}{ext}"
