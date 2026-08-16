"""Results endpoints – retrieve extraction results."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.infrastructure.db.database import get_async_session
from src.infrastructure.db.models import (
    DBExtractionResult,
    Invoice,
    Transaction,
)

router = APIRouter(prefix="/results", tags=["results"])


# ── Response schemas ──────────────────────────────────────────────


class TransactionStatusResponse(BaseModel):
    """Lightweight status check."""

    transaction_id: UUID
    status: str
    filename: str
    error_message: str | None = None


class ExtractionResultResponse(BaseModel):
    """Full extraction result payload."""

    transaction_id: UUID
    status: str
    filename: str
    strategy_name: str | None = None
    confidence: float | None = None
    extracted_data: dict | None = None
    invoice_id: UUID | None = None
    invoice_number: str | None = None
    commodity: str | None = None
    error_message: str | None = None


class TransactionListItem(BaseModel):
    """Summary item for transaction list endpoint."""

    transaction_id: UUID
    status: str
    filename: str
    commodity: str | None = None
    created_at: str | None = None
    has_extraction: bool = False


class PaginatedTransactionResponse(BaseModel):
    """Paginated list of transactions."""

    items: list[TransactionListItem]
    total: int
    page: int
    page_size: int
    total_pages: int


# ── Endpoints ─────────────────────────────────────────────────────


@router.get("/", response_model=PaginatedTransactionResponse)
async def list_results(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: str | None = Query(None, alias="status", description="Filter by status"),
    commodity: str | None = Query(None, description="Filter by commodity type"),
) -> PaginatedTransactionResponse:
    """List all transactions with pagination and optional filtering.

    Výpis všech transakcí s stránkováním a filtrováním.
    """
    query = select(Transaction)
    count_query = select(func.count()).select_from(Transaction)

    if status_filter:
        query = query.where(Transaction.status == status_filter)
        count_query = count_query.where(Transaction.status == status_filter)

    if commodity:
        query = query.where(Transaction.commodity == commodity)
        count_query = count_query.where(Transaction.commodity == commodity)

    # Get total count
    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    # Build response items with a single subquery for extraction existence
    from sqlalchemy import exists as sa_exists

    ext_exists_subq = (
        select(DBExtractionResult.transaction_id)
        .where(DBExtractionResult.transaction_id == Transaction.id)
        .exists()
        .correlate(Transaction)
    )

    # Get page of results with extraction existence in one query
    query = (
        select(Transaction, ext_exists_subq.label("has_extraction"))
    )

    if status_filter:
        query = query.where(Transaction.status == status_filter)

    if commodity:
        query = query.where(Transaction.commodity == commodity)

    offset = (page - 1) * page_size
    query = query.order_by(Transaction.created_at.desc()).offset(offset).limit(page_size)
    result = await session.execute(query)
    rows = result.all()

    # Build response items
    items = []
    for tx, has_ext in rows:
        items.append(
            TransactionListItem(
                transaction_id=tx.id,
                status=tx.status,
                filename=tx.filename,
                commodity=tx.commodity,
                created_at=tx.created_at.isoformat() if tx.created_at else None,
                has_extraction=has_ext,
            )
        )

    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    return PaginatedTransactionResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{task_id}", response_model=ExtractionResultResponse)
async def get_result(
    task_id: UUID,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> ExtractionResultResponse:
    """Return the extraction result for a given task (transaction).

    Returns 404 if the transaction does not exist.
    Returns partial data if processing is still in progress.
    """
    # Fetch transaction
    result = await session.execute(
        select(Transaction).where(Transaction.id == task_id)
    )
    transaction = result.scalar_one_or_none()

    if transaction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {task_id} not found.",
        )

    # Fetch latest extraction result
    ext_result = await session.execute(
        select(DBExtractionResult)
        .where(DBExtractionResult.transaction_id == task_id)
        .order_by(DBExtractionResult.created_at.desc())
        .limit(1)
    )
    extraction = ext_result.scalar_one_or_none()

    # Fetch associated invoice (if saved)
    inv_result = await session.execute(
        select(Invoice).where(Invoice.transaction_id == task_id).limit(1)
    )
    invoice = inv_result.scalar_one_or_none()

    return ExtractionResultResponse(
        transaction_id=transaction.id,
        status=transaction.status,
        filename=transaction.filename,
        strategy_name=extraction.strategy_name if extraction else None,
        confidence=extraction.confidence if extraction else None,
        extracted_data=extraction.extracted_json if extraction else None,
        invoice_id=invoice.id if invoice else None,
        invoice_number=invoice.invoice_number if invoice else None,
        commodity=invoice.commodity if invoice else None,
        error_message=transaction.error_message,
    )


@router.get("/{task_id}/status", response_model=TransactionStatusResponse)
async def get_result_status(
    task_id: UUID,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> TransactionStatusResponse:
    """Lightweight status check for a transaction."""
    result = await session.execute(
        select(Transaction).where(Transaction.id == task_id)
    )
    transaction = result.scalar_one_or_none()

    if transaction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {task_id} not found.",
        )

    return TransactionStatusResponse(
        transaction_id=transaction.id,
        status=transaction.status,
        filename=transaction.filename,
        error_message=transaction.error_message,
    )
