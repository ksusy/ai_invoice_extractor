"""Ingestion module – byte-stream and local-path handling."""

from src.core.ingestion.base import BaseIngestor, IngestedDocument
from src.core.ingestion.service import IngestionService, create_ingestion_service

__all__ = [
    "BaseIngestor",
    "IngestedDocument",
    "IngestionService",
    "create_ingestion_service",
]
