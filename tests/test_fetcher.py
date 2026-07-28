"""Tests for backend.modules.storage.fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import boto3
import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from backend.modules.storage.fetcher import (
    AudioFetcher,
    AudioFetchResult,
    FetchError,
    S3StorageBackend,
)


# ---------------------------------------------------------------------------
# S3StorageBackend
# ---------------------------------------------------------------------------


class TestS3StorageBackendFetch:
    """Test S3StorageBackend.fetch()."""

    def test_fetch_success(self, mock_s3_client, mock_boto3_session):
        """Test successful fetch returns AudioFetchResult."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            backend = S3StorageBackend()
            result = backend.fetch(key="audio/test.wav", bucket="test-bucket")

        assert isinstance(result, AudioFetchResult)
        assert result.key == "audio/test.wav"
        assert result.data == b"fake audio data"
        assert result.content_type == "audio/wav"
        assert result.size_bytes == 15

    def test_fetch_uses_default_bucket(self, mock_s3_client, mock_boto3_session):
        """Test that default bucket is used when none provided."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            backend = S3StorageBackend()
            backend.fetch(key="audio/test.wav")

        mock_s3_client.get_object.assert_called_once_with(
            Bucket="sbobinator", Key="audio/test.wav"
        )

    def test_fetch_object_not_found(self, mock_s3_client, mock_boto3_session):
        """Test NoSuchKey raises FetchError."""
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
            "GetObject",
        )
        with patch("boto3.Session", return_value=mock_boto3_session):
            backend = S3StorageBackend()
            with pytest.raises(FetchError, match="Object not found"):
                backend.fetch(key="missing.wav", bucket="test-bucket")

    def test_fetch_access_denied(self, mock_s3_client, mock_boto3_session):
        """Test AccessDenied raises FetchError."""
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Forbidden"}},
            "GetObject",
        )
        with patch("boto3.Session", return_value=mock_boto3_session):
            backend = S3StorageBackend()
            with pytest.raises(FetchError, match="Access denied"):
                backend.fetch(key="protected.wav", bucket="test-bucket")

    def test_fetch_access_denied_403(self, mock_s3_client, mock_boto3_session):
        """Test 403 error code raises FetchError."""
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}},
            "GetObject",
        )
        with patch("boto3.Session", return_value=mock_boto3_session):
            backend = S3StorageBackend()
            with pytest.raises(FetchError, match="Access denied"):
                backend.fetch(key="protected.wav", bucket="test-bucket")

    def test_fetch_file_too_large(self, mock_s3_client, mock_boto3_session):
        """Test that files exceeding max_audio_size raise FetchError."""
        # Return data larger than the default 2GB limit
        large_data = b"x" * (2 * 1024 * 1024 * 1024 + 1)
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: large_data),
            "ContentType": "audio/wav",
        }
        with patch("boto3.Session", return_value=mock_boto3_session):
            backend = S3StorageBackend()
            with pytest.raises(FetchError, match="too large"):
                backend.fetch(key="huge.wav", bucket="test-bucket")

    def test_fetch_no_credentials(self, mock_s3_client, mock_boto3_session):
        """Test NoCredentialsError raises FetchError."""
        mock_s3_client.get_object.side_effect = NoCredentialsError()
        with patch("boto3.Session", return_value=mock_boto3_session):
            backend = S3StorageBackend()
            with pytest.raises(FetchError, match="AWS credentials not found"):
                backend.fetch(key="audio/test.wav", bucket="test-bucket")

    def test_fetch_default_content_type(self, mock_s3_client, mock_boto3_session):
        """Test that missing ContentType defaults to application/octet-stream."""
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: b"data"),
            # No ContentType key
        }
        with patch("boto3.Session", return_value=mock_boto3_session):
            backend = S3StorageBackend()
            result = backend.fetch(key="audio/test.wav", bucket="test-bucket")
            assert result.content_type == "application/octet-stream"

    def test_fetch_custom_session(self, mock_s3_client, mock_boto3_session):
        """Test that a custom session is used."""
        backend = S3StorageBackend(session=mock_boto3_session)
        backend.fetch(key="audio/test.wav", bucket="test-bucket")
        mock_boto3_session.client.assert_called_once()


class TestS3StorageBackendUpload:
    """Test S3StorageBackend.upload()."""

    def test_upload_success(self, mock_s3_client, mock_boto3_session):
        """Test successful upload."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            backend = S3StorageBackend()
            backend.upload(
                key="audio/test.wav",
                data=b"hello audio",
                content_type="audio/wav",
                bucket="test-bucket",
            )

        mock_s3_client.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="audio/test.wav",
            Body=b"hello audio",
            ContentType="audio/wav",
        )

    def test_upload_uses_default_bucket(self, mock_s3_client, mock_boto3_session):
        """Test default bucket is used when none provided."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            backend = S3StorageBackend()
            backend.upload(
                key="audio/test.wav",
                data=b"data",
                content_type="audio/wav",
            )

        mock_s3_client.put_object.assert_called_once_with(
            Bucket="sbobinator",
            Key="audio/test.wav",
            Body=b"data",
            ContentType="audio/wav",
        )

    def test_upload_access_denied(self, mock_s3_client, mock_boto3_session):
        """Test AccessDenied raises FetchError."""
        mock_s3_client.put_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Forbidden"}},
            "PutObject",
        )
        with patch("boto3.Session", return_value=mock_boto3_session):
            backend = S3StorageBackend()
            with pytest.raises(FetchError, match="Access denied"):
                backend.upload(
                    key="protected.wav",
                    data=b"data",
                    content_type="audio/wav",
                    bucket="test-bucket",
                )

    def test_upload_no_credentials(self, mock_s3_client, mock_boto3_session):
        """Test NoCredentialsError raises FetchError."""
        mock_s3_client.put_object.side_effect = NoCredentialsError()
        with patch("boto3.Session", return_value=mock_boto3_session):
            backend = S3StorageBackend()
            with pytest.raises(FetchError, match="AWS credentials not found"):
                backend.upload(
                    key="audio/test.wav",
                    data=b"data",
                    content_type="audio/wav",
                    bucket="test-bucket",
                )


# ---------------------------------------------------------------------------
# AudioFetcher
# ---------------------------------------------------------------------------


class TestAudioFetcher:
    """Test high-level AudioFetcher."""

    def test_fetch_via_backend(self, mock_s3_client, mock_boto3_session):
        """Test AudioFetcher delegates to backend."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            fetcher = AudioFetcher()
            result = fetcher.fetch(key="audio/test.wav", bucket="test-bucket")

        assert isinstance(result, AudioFetchResult)
        assert result.data == b"fake audio data"

    def test_upload_via_backend(self, mock_s3_client, mock_boto3_session):
        """Test AudioFetcher delegates upload to backend."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            fetcher = AudioFetcher()
            fetcher.upload(
                key="audio/test.wav",
                data=b"hello",
                content_type="audio/wav",
                bucket="test-bucket",
            )

        mock_s3_client.put_object.assert_called_once()

    def test_custom_backend(self, mock_s3_client):
        """Test AudioFetcher with a custom backend."""
        custom_backend = MagicMock()
        custom_backend.fetch.return_value = AudioFetchResult(
            key="audio/test.wav",
            data=b"custom data",
            content_type="audio/wav",
            size_bytes=11,
        )
        fetcher = AudioFetcher(backend=custom_backend)
        result = fetcher.fetch(key="audio/test.wav")
        assert result.data == b"custom data"

    def test_custom_default_bucket(self):
        """Test AudioFetcher with a custom default bucket."""
        custom_backend = MagicMock()
        custom_backend.fetch.return_value = AudioFetchResult(
            key="audio/test.wav",
            data=b"data",
            content_type="audio/wav",
            size_bytes=4,
        )
        fetcher = AudioFetcher(
            backend=custom_backend,
            default_bucket="custom-bucket",
        )
        # When bucket=None, should use default_bucket
        fetcher.fetch(key="audio/test.wav")
        custom_backend.fetch.assert_called_once_with(
            "audio/test.wav", bucket="custom-bucket"
        )
