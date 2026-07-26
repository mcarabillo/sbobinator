"""Storage module for fetching audio files from S3/MinIO backends."""

from .fetcher import AudioFetcher

__all__ = ["AudioFetcher"]
