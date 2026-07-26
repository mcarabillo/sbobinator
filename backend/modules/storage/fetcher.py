"""Audio file fetching from S3 / MinIO storage.

Uses ``boto3`` to interact with S3-compatible backends. Configuration is
read from ``utils.config.S3StorageConfig`` (accessible via ``settings().storage``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from utils.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioFetchResult:
    """Immutable result of a successful audio fetch."""

    key: str
    """S3 object key (path inside the bucket)."""
    data: bytes
    """Raw audio bytes."""
    content_type: str
    """Content-Type metadata from the S3 object."""
    size_bytes: int
    """Size of the fetched payload in bytes."""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class StorageBackend(Protocol):
    """Protocol for storage backends.

    Implementations must provide ``fetch`` which downloads audio data
    from a given S3 key and returns an ``AudioFetchResult``.
    """

    def fetch(self, key: str, bucket: str | None = None) -> AudioFetchResult: ...  # noqa: D102


# ---------------------------------------------------------------------------
# S3 / MinIO Backend
# ---------------------------------------------------------------------------


class S3StorageBackend:
    """Fetches audio files from S3 / MinIO using ``boto3``.

    Parameters
    ----------
    config:
        S3 configuration. Defaults to the global ``settings().storage``.
    session:
        Optional ``boto3.Session`` for custom credential management.
    """

    def __init__(
        self,
        config=None,
        session: boto3.Session | None = None,
    ) -> None:
        self.config = config or settings().storage

        if session is not None:
            self._session = session
        else:
            self._session = boto3.Session(
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                region_name=self.config.region,
            )

        self._client = self._session.client(
            "s3",
            endpoint_url=self.config.endpoint_url,
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
        )

    def fetch(self, key: str, bucket: str | None = None) -> AudioFetchResult:
        """Download an audio object from S3 / MinIO.

        Parameters
        ----------
        key:
            S3 object key (e.g. ``"audio/jobs/12345.mp3"``).
        bucket:
            Override the default bucket from config.

        Raises
        ------
        FetchError
            If the object cannot be retrieved (missing, permission denied, etc.).
        """
        bucket = bucket or self.config.bucket
        max_size = self.config.max_audio_size

        logger.info("Fetching audio from s3://%s/%s", bucket, key)

        try:
            response = self._client.get_object(Bucket=bucket, Key=key)

            data = response["Body"].read()

            # --- Size check ---
            if len(data) > max_size:
                raise FetchError(
                    f"Audio file too large: {len(data)} bytes "
                    f"(max {max_size:,})"
                )

            content_type = response.get("ContentType", "application/octet-stream")

            result = AudioFetchResult(
                key=key,
                data=data,
                content_type=content_type,
                size_bytes=len(data),
            )

            logger.info(
                "Fetched %d bytes (s3://%s/%s) [%s]",
                result.size_bytes,
                bucket,
                result.key,
                result.content_type,
            )
            return result

        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                raise FetchError(f"Object not found: s3://{bucket}/{key}") from exc
            if error_code in ("AccessDenied", "403"):
                raise FetchError(f"Access denied to s3://{bucket}/{key}") from exc
            raise FetchError(f"S3 error ({error_code}): {exc}") from exc
        except NoCredentialsError:
            raise FetchError(
                "AWS credentials not found. Check S3_STORAGE__ACCESS_KEY_ID "
                "and S3_STORAGE__SECRET_ACCESS_KEY."
            ) from None


# ---------------------------------------------------------------------------
# Pipeline-level fetcher
# ---------------------------------------------------------------------------


class AudioFetcher:
    """High-level audio fetcher backed by S3 / MinIO.

    Parameters
    ----------
    backend:
        S3 storage backend. Defaults to a new one built from config.
    default_bucket:
        Override the default bucket from config.
    """

    def __init__(
        self,
        backend: S3StorageBackend | None = None,
        default_bucket: str | None = None,
    ) -> None:
        self.backend = backend or S3StorageBackend()
        self.default_bucket = default_bucket or settings().storage.bucket

    def fetch(self, key: str, bucket: str | None = None) -> AudioFetchResult:
        """Fetch audio from S3 / MinIO.

        Parameters
        ----------
        key:
            S3 object key.
        bucket:
            Override the default bucket.
        """
        return self.backend.fetch(key, bucket=bucket or self.default_bucket)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FetchError(Exception):
    """Raised when an audio fetch operation fails."""
