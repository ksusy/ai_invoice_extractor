"""Abstract base class for native-PDF vs scan classification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class DocumentKind(str, Enum):
    """Result of the native/scan classifier."""

    NATIVE_PDF = "native_pdf"   # Has a selectable text layer
    SCANNED = "scanned"         # Image-only or rasterised PDF
    HYBRID = "hybrid"           # Mixed pages


class BaseClassifier(ABC):
    """Strategy interface for detecting whether a PDF is native or scanned.

    Implementations may use heuristics (e.g. extractable text length),
    pdfminer structure analysis, or ML-based approaches.
    """

    @abstractmethod
    async def classify(self, file_bytes: bytes) -> DocumentKind:
        """Classify a document's type.

        Args:
            file_bytes: Raw bytes of the document (PDF or image).

        Returns:
            A ``DocumentKind`` enum value.
        """
        ...
