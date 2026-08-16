"""Abstract base class for OCR engines (Tesseract, PaddleOCR, EasyOCR …).

Abstraktní třída pro OCR enginy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class OCRResult:
    """Raw OCR output with per-page text and performance metrics.

    Attributes:
        full_text: Complete OCR text from all pages.
        pages: Per-page text list (for multi-page documents).
        confidence: Average confidence score (0.0-1.0).
        engine_name: Name of the OCR engine used.
        latency_ms: Execution time in milliseconds.
        cost_usd: Estimated cost (for cloud-based engines).
        raw_data: Engine-specific metadata.
        error_message: Error description if OCR failed.
    """

    full_text: str
    pages: list[str] = field(default_factory=list)
    confidence: float = 0.0
    engine_name: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw_data: dict | None = None
    error_message: str | None = None

    @property
    def is_successful(self) -> bool:
        """Check if OCR completed without errors."""
        return self.error_message is None and len(self.full_text) > 0

    @property
    def page_count(self) -> int:
        """Number of pages processed."""
        return len(self.pages) if self.pages else (1 if self.full_text else 0)


class BaseOCREngine(ABC):
    """Strategy interface for optical character recognition.

    Each concrete engine (Tesseract, PaddleOCR, EasyOCR) implements
    this interface so they can be swapped at runtime via configuration.

    Design pattern: Strategy
    Responsibility: Single - OCR text extraction only
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name (e.g. ``'tesseract'``)."""
        ...

    @abstractmethod
    async def recognize(self, image_bytes: bytes) -> OCRResult:
        """Run OCR on a single image / page.

        Args:
            image_bytes: Raw bytes of a pre-processed image.

        Returns:
            An ``OCRResult`` containing the recognised text.
        """
        ...

    @abstractmethod
    async def recognize_pdf(self, pdf_bytes: bytes) -> OCRResult:
        """Run OCR on a multi-page PDF.

        Converts each PDF page to image and runs OCR.

        Args:
            pdf_bytes: Raw bytes of the PDF document.

        Returns:
            An ``OCRResult`` with per-page text stored in ``pages``.
        """
        ...

    @abstractmethod
    async def extract_native_text(self, pdf_bytes: bytes) -> OCRResult:
        """Extract text from a native PDF (no OCR needed).

        Uses PDF text layer extraction instead of image-based OCR.

        Args:
            pdf_bytes: Raw bytes of the PDF document.

        Returns:
            An ``OCRResult`` with extracted text, engine_name='native'.
        """
        ...




