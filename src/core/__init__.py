"""Core module – business logic and processing pipeline.

This module contains the main processing components:
    - Ingestion: File upload and storage
    - Classification: Native PDF vs scanned detection
    - OCR Engines: Text extraction from images
    - Extraction: Structured data extraction
    - Analysis: Data validation and enrichment
    - Analytics: SQL agent for data queries
"""

from src.core.main_pipeline import (
    ProcessingOrchestrator,
    ProcessingResult,
    ProcessingStatus,
    create_orchestrator,
)

__all__ = [
    "ProcessingOrchestrator",
    "ProcessingResult",
    "ProcessingStatus",
    "create_orchestrator",
]
