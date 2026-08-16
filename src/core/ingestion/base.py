"""Abstract base class for document ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IngestedDocument:
    """Container for a successfully ingested document."""

    file_name: str
    file_bytes: bytes
    mime_type: str
    source_path: str | None = None


class BaseIngestor(ABC):
    """Strategy interface for ingesting invoices from various sources.

    Concrete implementations handle byte streams (API uploads),
    local file paths, or cloud storage locations.
    """

    @abstractmethod
    async def ingest(self, source: str | bytes | Path) -> IngestedDocument:
        """Read / download the document and return an ``IngestedDocument``.

        Args:
            source: A file path, URL, or raw bytes.

        Returns:
            An ``IngestedDocument`` ready for downstream processing.

        Raises:
            FileNotFoundError: If the source cannot be located.
            ValueError: If the file type is unsupported.
        """
        ...
