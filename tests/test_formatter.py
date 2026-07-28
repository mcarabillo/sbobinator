"""Tests for backend.modules.output.formatter."""

from __future__ import annotations

import json

import pytest

from backend.modules.output.formatter import (
    OutputFormatter,
    _format_csv,
    _format_json,
    _format_srt,
    _format_tsv,
    _format_txt,
    _format_vtt,
    _format_timestamp_srt,
    _format_timestamp_vtt,
)
from backend.modules.transcription.batched_pipeline import TranscriptionSegment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_segments():
    """Return sample TranscriptionSegment objects."""
    return [
        TranscriptionSegment(start=0.0, end=1.5, text="Ciao mondo"),
        TranscriptionSegment(start=2.0, end=4.0, text="Questo è un test"),
        TranscriptionSegment(start=5.0, end=6.5, text="Fine del test"),
    ]


@pytest.fixture()
def formatter():
    return OutputFormatter()


# ---------------------------------------------------------------------------
# _format_timestamp helpers
# ---------------------------------------------------------------------------


class TestTimestampHelpers:
    def test_srt_timestamp_basic(self):
        ts = _format_timestamp_srt(3661.123)
        assert ts == "01:01:01,123"

    def test_srt_timestamp_zero(self):
        ts = _format_timestamp_srt(0.0)
        assert ts == "00:00:00,000"

    def test_srt_timestamp_30_seconds(self):
        ts = _format_timestamp_srt(30.5)
        assert ts == "00:00:30,500"

    def test_vtt_timestamp_basic(self):
        ts = _format_timestamp_vtt(3661.123)
        assert ts == "01:01:01.123"

    def test_vtt_timestamp_zero(self):
        ts = _format_timestamp_vtt(0.0)
        assert ts == "00:00:00.000"

    def test_vtt_timestamp_30_seconds(self):
        ts = _format_timestamp_vtt(30.5)
        assert ts == "00:00:30.500"


# ---------------------------------------------------------------------------
# Format functions
# ---------------------------------------------------------------------------


class TestFormatSRT:
    def test_basic(self, sample_segments):
        result = _format_srt(sample_segments)
        lines = result.strip().split("\n")
        # Segment 1
        assert "1" in lines[0]
        assert "-->" in lines[1]
        assert "Ciao mondo" in lines[2]
        # Segment 2
        assert "2" in lines[4]
        assert "Questo è un test" in lines[6]

    def test_timestamps_correct(self, sample_segments):
        result = _format_srt(sample_segments)
        assert "00:00:00,000" in result
        assert "00:00:01,500" in result
        assert "00:00:04,000" in result


class TestFormatVTT:
    def test_basic(self, sample_segments):
        result = _format_vtt(sample_segments)
        assert result.startswith("WEBVTT")
        assert "-->" in result
        assert "Ciao mondo" in result

    def test_timestamps_correct(self, sample_segments):
        result = _format_vtt(sample_segments)
        assert "00:00:00.000" in result
        assert "00:00:01.500" in result


class TestFormatTxt:
    def test_basic(self, sample_segments):
        result = _format_txt(sample_segments)
        assert "Ciao mondo" in result
        assert "Questo è un test" in result
        assert "Fine del test" in result
        # All text on one line, separated by spaces
        assert result.count("\n") == 0


class TestFormatCsv:
    def test_basic(self, sample_segments):
        result = _format_csv(sample_segments)
        lines = result.strip().split("\n")
        assert lines[0] == "start,end,text"
        assert "Ciao mondo" in lines[1]

    def test_no_header(self, sample_segments):
        result = _format_csv(sample_segments, include_header=False)
        lines = result.strip().split("\n")
        assert lines[0] != "start,end,text"

    def test_escaping_quotes(self):
        segments = [
            TranscriptionSegment(start=0.0, end=1.0, text='He said "hello"'),
        ]
        result = _format_csv(segments)
        assert '""hello""' in result


class TestFormatTsv:
    def test_basic(self, sample_segments):
        result = _format_tsv(sample_segments)
        lines = result.strip().split("\n")
        assert lines[0] == "start\tend\ttext"
        assert "Ciao mondo" in lines[1]

    def test_no_header(self, sample_segments):
        result = _format_tsv(sample_segments, include_header=False)
        lines = result.strip().split("\n")
        assert lines[0] != "start\tend\ttext"


class TestFormatJson:
    def test_basic(self, sample_segments):
        result = _format_json(sample_segments)
        data = json.loads(result)
        assert "segments" in data
        assert "text" in data
        assert "segment_count" in data
        assert data["segment_count"] == 3
        assert len(data["segments"]) == 3
        assert data["segments"][0]["text"] == "Ciao mondo"

    def test_with_metadata(self, sample_segments):
        result = _format_json(sample_segments, metadata={"language": "it", "duration": 120.0})
        data = json.loads(result)
        assert data["metadata"]["language"] == "it"
        assert data["metadata"]["duration"] == 120.0

    def test_text_field(self, sample_segments):
        result = _format_json(sample_segments)
        data = json.loads(result)
        assert "Ciao mondo" in data["text"]
        assert "Questo è un test" in data["text"]


# ---------------------------------------------------------------------------
# OutputFormatter class
# ---------------------------------------------------------------------------


class TestOutputFormatter:
    def test_format_srt(self, formatter, sample_segments):
        result = formatter.format(sample_segments, "srt")
        assert "00:00:00,000" in result

    def test_format_vtt(self, formatter, sample_segments):
        result = formatter.format(sample_segments, "vtt")
        assert result.startswith("WEBVTT")

    def test_format_txt(self, formatter, sample_segments):
        result = formatter.format(sample_segments, "txt")
        assert "Ciao mondo" in result

    def test_format_csv(self, formatter, sample_segments):
        result = formatter.format(sample_segments, "csv")
        assert "start,end,text" in result

    def test_format_tsv(self, formatter, sample_segments):
        result = formatter.format(sample_segments, "tsv")
        assert "start\tend\ttext" in result

    def test_format_json(self, formatter, sample_segments):
        result = formatter.format(sample_segments, "json")
        data = json.loads(result)
        assert data["segment_count"] == 3

    def test_format_case_insensitive(self, formatter, sample_segments):
        """Format should work with uppercase format strings."""
        result = formatter.format(sample_segments, "SRT")
        assert "00:00:00,000" in result

    def test_format_uppercase_json(self, formatter, sample_segments):
        result = formatter.format(sample_segments, "JSON")
        data = json.loads(result)
        assert "segments" in data

    def test_unsupported_format_raises(self, formatter, sample_segments):
        with pytest.raises(ValueError, match="Unsupported format"):
            formatter.format(sample_segments, "xml")

    def test_get_extension(self, formatter):
        assert formatter.get_extension("json") == ".json"
        assert formatter.get_extension("srt") == ".srt"
        assert formatter.get_extension("vtt") == ".vtt"
        assert formatter.get_extension("txt") == ".txt"
        assert formatter.get_extension("csv") == ".csv"
        assert formatter.get_extension("tsv") == ".tsv"

    def test_get_extension_unknown_raises(self, formatter):
        with pytest.raises(ValueError, match="Unknown format"):
            formatter.get_extension("xml")

    def test_get_mime_type(self, formatter):
        assert formatter.get_mime_type("json") == "application/json"
        assert formatter.get_mime_type("srt") == "text/srt"
        assert formatter.get_mime_type("vtt") == "text/vtt"
        assert formatter.get_mime_type("txt") == "text/plain"
        assert formatter.get_mime_type("csv") == "text/csv"
        assert formatter.get_mime_type("tsv") == "text/tab-separated-values"

    def test_get_mime_type_unknown_raises(self, formatter):
        with pytest.raises(ValueError, match="Unknown format"):
            formatter.get_mime_type("xml")

    def test_build_output_key(self, formatter):
        result = formatter.build_output_key("audio/uploads/podcast.mp3", "srt")
        assert result == "audio/uploads/podcast.srt"

    def test_build_output_key_json(self, formatter):
        result = formatter.build_output_key("audio/uploads/podcast.mp3", "json")
        assert result == "audio/uploads/podcast.json"

    no_ext_key = "audio/uploads/recording"

    def test_build_output_key_no_extension(self, formatter):
        result = formatter.build_output_key("audio/uploads/recording", "txt")
        assert result == "audio/uploads/recording.txt"

    def test_custom_formats(self):
        """Test formatter with restricted format list."""
        f = OutputFormatter(formats=["srt", "vtt"])
        assert f.get_extension("srt") == ".srt"
        with pytest.raises(ValueError):
            f.format([], "json")
