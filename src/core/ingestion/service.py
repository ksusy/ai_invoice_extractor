"""Ingestion service for handling file uploads and batch processing.

This service is responsible for:
    1. Accepting files (single via FastAPI, batch from directory)
    2. Saving original files to data/raw/
    3. Creating Transaction records with PENDING status

Služba pro příjem souborů faktur.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.core.ingestion.base import BaseIngestor, IngestedDocument
from src.domain.constants import SUPPORTED_FILE_EXTENSIONS
from src.infrastructure.db.database import get_session_context
from src.infrastructure.db.models import Transaction

logger = logging.getLogger(__name__)


class IngestionService(BaseIngestor):
    """Concrete implementation of the ingestion strategy.

    Handles both single file uploads (API) and batch directory processing (CLI).
    Files are saved to data/raw/ with a unique filename to avoid collisions.
    """

    def __init__(self, session: AsyncSession | None = None) -> None:
        """Initialize the ingestion service.

        Args:
            session: Optional async session for dependency injection.
                     If None, a new session will be created per operation.
        """
        self._session = session
        self._settings = get_settings()
        self._raw_dir = Path(self._settings.raw_data_dir)
        self._raw_dir.mkdir(parents=True, exist_ok=True)

    async def ingest(self, source: str | bytes | Path) -> IngestedDocument:
        """Ingest a single file from path or bytes.

        Args:
            source: File path (str/Path) or raw bytes.

        Returns:
            IngestedDocument with file metadata.

        Raises:
            FileNotFoundError: If source path doesn't exist.
            ValueError: If file extension is not supported.
        """
        if isinstance(source, bytes):
            # For raw bytes, we need a filename - generate one
            file_name = f"upload_{uuid.uuid4().hex[:8]}.pdf"
            file_bytes = source
            source_path = None
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Source file not found: {source}")

            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_FILE_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file type: {suffix}. "
                    f"Supported: {SUPPORTED_FILE_EXTENSIONS}"
                )

            file_name = path.name
            file_bytes = path.read_bytes()
            source_path = str(path.absolute())

        mime_type, _ = mimetypes.guess_type(file_name)
        mime_type = mime_type or "application/octet-stream"

        return IngestedDocument(
            file_name=file_name,
            file_bytes=file_bytes,
            mime_type=mime_type,
            source_path=source_path,
        )

    async def ingest_and_save(
        self,
        source: str | bytes | Path,
        original_filename: str | None = None,
    ) -> Transaction:
        """Ingest a file, save it to raw/, and create a DB record.

        Args:
            source: File path or raw bytes.
            original_filename: Original filename (required for bytes input).

        Returns:
            Created Transaction record with PENDING status.
        """
        # Ingest the document
        if isinstance(source, bytes):
            if not original_filename:
                raise ValueError("original_filename required for bytes input")
            doc = IngestedDocument(
                file_name=original_filename,
                file_bytes=source,
                mime_type=mimetypes.guess_type(original_filename)[0] or "application/octet-stream",
                source_path=None,
            )
        else:
            doc = await self.ingest(source)
            if original_filename:
                doc = IngestedDocument(
                    file_name=original_filename,
                    file_bytes=doc.file_bytes,
                    mime_type=doc.mime_type,
                    source_path=doc.source_path,
                )

        # Generate unique filename to avoid collisions
        file_hash = hashlib.md5(doc.file_bytes).hexdigest()[:8]
        unique_filename = f"{uuid.uuid4().hex[:8]}_{file_hash}_{doc.file_name}"
        save_path = self._raw_dir / unique_filename

        # Save the file
        save_path.write_bytes(doc.file_bytes)

        # Create transaction record
        transaction = Transaction(
            filename=doc.file_name,
            file_path=str(save_path),
            file_size_bytes=len(doc.file_bytes),
            mime_type=doc.mime_type,
            status="pending",
        )

        if self._session:
            self._session.add(transaction)
            await self._session.flush()
        else:
            async with get_session_context() as session:
                session.add(transaction)
                await session.flush()
                await session.refresh(transaction)

        return transaction

    async def ingest_from_upload_file(
        self,
        content: BinaryIO,
        filename: str,
        content_type: str | None = None,
    ) -> Transaction:
        """Ingest a file from FastAPI UploadFile.

        Args:
            content: File-like object with .read() method.
            filename: Original filename from the upload.
            content_type: MIME type from the upload.

        Returns:
            Created Transaction record.
        """
        file_bytes = content.read() if hasattr(content, "read") else content
        if isinstance(file_bytes, str):
            file_bytes = file_bytes.encode()

        # Validate extension
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_FILE_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {SUPPORTED_FILE_EXTENSIONS}"
            )

        return await self.ingest_and_save(
            source=file_bytes,
            original_filename=filename,
        )

    async def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = False,
    ) -> list[Transaction]:
        """Batch ingest all supported files from a directory.

        Args:
            directory: Path to the directory to scan.
            recursive: If True, scan subdirectories as well.

        Returns:
            List of created Transaction records.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Collect all supported files
        files: list[Path] = []
        pattern = "**/*" if recursive else "*"
        for ext in SUPPORTED_FILE_EXTENSIONS:
            files.extend(dir_path.glob(f"{pattern}{ext}"))
            files.extend(dir_path.glob(f"{pattern}{ext.upper()}"))

        # Remove duplicates and sort
        files = sorted(set(files))

        # Ingest each file
        transactions: list[Transaction] = []
        for file_path in files:
            try:
                tx = await self.ingest_and_save(source=file_path)
                transactions.append(tx)
            except Exception as e:
                # Log error but continue with other files
                logger.error("Error ingesting %s: %s", file_path, e)

        return transactions

    async def get_transaction(self, transaction_id: uuid.UUID) -> Transaction | None:
        """Retrieve a transaction by ID.

        Args:
            transaction_id: UUID of the transaction.

        Returns:
            Transaction if found, None otherwise.
        """
        if self._session:
            result = await self._session.execute(
                select(Transaction).where(Transaction.id == transaction_id)
            )
            return result.scalar_one_or_none()
        else:
            async with get_session_context() as session:
                result = await session.execute(
                    select(Transaction).where(Transaction.id == transaction_id)
                )
                return result.scalar_one_or_none()

    async def list_pending_transactions(self) -> Sequence[Transaction]:
        """List all transactions with PENDING status.

        Returns:
            List of pending Transaction records.
        """
        if self._session:
            result = await self._session.execute(
                select(Transaction).where(Transaction.status == "pending")
            )
            return result.scalars().all()
        else:
            async with get_session_context() as session:
                result = await session.execute(
                    select(Transaction).where(Transaction.status == "pending")
                )
                return result.scalars().all()


# Convenience factory function
def create_ingestion_service(session: AsyncSession | None = None) -> IngestionService:
    """Factory function to create an IngestionService instance."""
    return IngestionService(session=session)
