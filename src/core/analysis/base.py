"""Abstract base class for invoice analysis / business-logic rules."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities import InvoiceMetadata


class AnalysisReport:
    """Summary produced by an analyser."""

    def __init__(
        self,
        is_correction: bool = False,
        is_transitional: bool = False,
        linked_invoice: str | None = None,
        cross_year: bool = False,
        warnings: list[str] | None = None,
    ) -> None:
        self.is_correction = is_correction
        self.is_transitional = is_transitional
        self.linked_invoice = linked_invoice
        self.cross_year = cross_year
        self.warnings = warnings or []


class BaseAnalyser(ABC):
    """Strategy interface for post-extraction business analysis.

    Responsibilities:
    - Detect *correction* invoices (opravná faktura).
    - Detect *transitional* invoices (přechodná faktura).
    - Flag cross-year billing periods.
    - Identify links to previous invoices.
    """

    @abstractmethod
    async def analyse(self, metadata: InvoiceMetadata) -> AnalysisReport:
        """Perform analysis on extracted metadata.

        Args:
            metadata: Validated invoice metadata.

        Returns:
            An ``AnalysisReport`` with flags and warnings.
        """
        ...
