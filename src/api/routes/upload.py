"""Upload endpoints – receive invoices via API.

Koncové body pro nahrávání faktur přes API.
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.core import create_orchestrator
from src.core.classification import DocumentKind
from src.domain.constants import SUPPORTED_FILE_EXTENSIONS
from src.infrastructure.db.database import get_async_session

router = APIRouter(prefix="/upload", tags=["upload"])
logger = logging.getLogger(__name__)


# ── Response schemas ──────────────────────────────────────────────


class UploadResponse(BaseModel):
    """Response model for successful upload."""

    status: str
    transaction_id: UUID
    filename: str
    document_kind: str | None = None
    is_scan: bool | None = None


class BatchUploadResponse(BaseModel):
    """Response model for batch upload."""

    total: int
    successful: int
    failed: int
    results: list[UploadResponse]
    errors: list[dict] = []


# ── Helper functions ──────────────────────────────────────────────


def _validate_file_type(content_type: str | None, filename: str | None) -> None:
    """Validate that the uploaded file is a supported type."""
    # Check by content type
    allowed_content_types = {
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/tiff",
        "image/bmp",
    }

    if content_type and content_type in allowed_content_types:
        return

    # Check by file extension
    if filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in SUPPORTED_FILE_EXTENSIONS:
            return

    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail="Unsupported file type. Allowed: PDF, PNG, JPEG, TIFF, BMP",
    )


async def _validate_file_size(file: UploadFile) -> None:
    """Validate that the uploaded file does not exceed the size limit."""
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    await file.seek(0)  # Reset for downstream consumers
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(content) / 1024 / 1024:.1f} MB). Maximum: {settings.max_upload_size_mb} MB.",
        )


def _get_document_kind_label(kind: DocumentKind | None) -> str | None:
    """Convert DocumentKind enum to string label."""
    if kind is None:
        return None
    return {
        DocumentKind.NATIVE_PDF: "native_pdf",
        DocumentKind.SCANNED: "scanned",
        DocumentKind.HYBRID: "hybrid",
    }.get(kind, "unknown")


# ── Endpoints ─────────────────────────────────────────────────────


@router.post("/", status_code=status.HTTP_202_ACCEPTED, response_model=UploadResponse)
async def upload_invoice(
    file: Annotated[UploadFile, File(description="Invoice file to process")],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> UploadResponse:
    """Accept a single invoice file (PDF / image) for processing.

    The file is saved to data/raw/ and a processing task is created.
    Returns a transaction ID that can be used to poll for results.

    Nahrání jednotlivé faktury ke zpracování.
    """
    _validate_file_type(file.content_type, file.filename)
    await _validate_file_size(file)

    orchestrator = create_orchestrator(session=session)

    try:
        result = await orchestrator.process_upload(
            content=file.file,
            filename=file.filename or "unknown.pdf",
            content_type=file.content_type,
        )

        return UploadResponse(
            status=result.status.value,
            transaction_id=result.transaction_id,
            filename=file.filename or "unknown",
            document_kind=_get_document_kind_label(result.document_kind),
            is_scan=result.document_kind == DocumentKind.SCANNED if result.document_kind else None,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {e}",
        ) from e


@router.post("/batch", status_code=status.HTTP_202_ACCEPTED, response_model=BatchUploadResponse)
async def upload_batch(
    files: Annotated[list[UploadFile], File(description="Multiple invoice files to process")],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> BatchUploadResponse:
    """Accept multiple invoice files for batch processing.

    Returns a summary of processed files with their transaction IDs.

    Hromadné nahrání více faktur ke zpracování.
    """
    orchestrator = create_orchestrator(session=session)
    results: list[UploadResponse] = []
    errors: list[dict] = []
    successful = 0
    failed = 0

    for file in files:
        try:
            _validate_file_type(file.content_type, file.filename)
            await _validate_file_size(file)

            result = await orchestrator.process_upload(
                content=file.file,
                filename=file.filename or "unknown.pdf",
                content_type=file.content_type,
            )

            results.append(
                UploadResponse(
                    status=result.status.value,
                    transaction_id=result.transaction_id,
                    filename=file.filename or "unknown",
                    document_kind=_get_document_kind_label(result.document_kind),
                    is_scan=result.document_kind == DocumentKind.SCANNED if result.document_kind else None,
                )
            )
            successful += 1

        except HTTPException as e:
            failed += 1
            errors.append({"filename": file.filename or "unknown", "detail": e.detail})
        except Exception as e:
            logger.error("Batch upload failed for %s: %s", file.filename, e, exc_info=True)
            failed += 1
            errors.append({"filename": file.filename or "unknown", "detail": str(e)})

    return BatchUploadResponse(
        total=len(files),
        successful=successful,
        failed=failed,
        results=results,
        errors=errors,
    )


@router.get("/status/{transaction_id}", response_model=UploadResponse)
async def get_upload_status(
    transaction_id: UUID,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> UploadResponse:
    """Get the processing status of an uploaded file.

    Získání stavu zpracování nahrané faktury.
    """
    orchestrator = create_orchestrator(session=session)
    tx_status = await orchestrator.get_transaction_status(transaction_id)

    if tx_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction not found: {transaction_id}",
        )

    # Fetch transaction from DB for filename
    from sqlalchemy import select

    from src.infrastructure.db.models import Transaction

    result = await session.execute(
        select(Transaction).where(Transaction.id == transaction_id)
    )
    transaction = result.scalar_one_or_none()
    filename = transaction.filename if transaction else "unknown"

    return UploadResponse(
        status=tx_status.value,
        transaction_id=transaction_id,
        filename=filename,
    )

