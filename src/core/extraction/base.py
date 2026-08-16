"""Abstract base class for extraction strategies.

Strategies include: Regex, LLM Text, Vision LLM, and Hybrid.

Abstraktní třída pro extrakční strategie.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.domain.entities import (
    CommodityType,
    ExtractionResult,
    InvoiceData,
)

if TYPE_CHECKING:
    pass


@dataclass
class ExtractionContext:
    """Context passed to extraction strategies.

    Contains all information needed for extraction including raw text,
    source file metadata, and optional image data for vision-based strategies.
    """

    raw_text: str
    source_filename: str
    commodity_hint: CommodityType | None = None
    image_bytes: bytes | None = None
    page_count: int = 1
    extra_metadata: dict = field(default_factory=dict)


class BaseExtractionStrategy(ABC):
    """Strategy interface for extracting structured data from OCR text.

    Concrete implementations:
    - **RegexExtraction** – rule-based regular expressions
    - **LLMTextExtraction** – send plain text to an LLM
    - **VisionLLMExtraction** – send images to a vision-capable LLM
    - **HybridExtraction** – combine multiple strategies and vote

    Design pattern: Strategy
    Responsibility: Single - Extract structured data from text/images
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @abstractmethod
    async def extract(self, context: ExtractionContext) -> ExtractionResult:
        """Extract structured invoice data from context.

        Args:
            context: ExtractionContext with raw text and metadata.

        Returns:
            ExtractionResult with parsed InvoiceData or errors.
        """
        ...

    @abstractmethod
    async def validate(self, invoice_data: InvoiceData) -> list[str]:
        """Run post-extraction validation checks.

        Args:
            invoice_data: Parsed invoice data to validate.

        Returns:
            List of validation warnings (empty = all OK).
        """
        ...

    def can_handle_commodity(self, commodity: CommodityType) -> bool:
        """Check if this strategy supports the given commodity type.

        By default, all strategies support all commodities.
        Override in subclasses to restrict.
        """
        return True
