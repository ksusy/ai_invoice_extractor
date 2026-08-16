"""Response schemas returned by the API endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, Field

from src.domain.entities import InvoiceMetadata


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TaskAccepted(BaseModel):
    """Returned when an upload is accepted."""

    task_id: UUID
    filename: str
    status: str = "accepted"
    created_at: datetime = Field(default_factory=_utcnow)


class ExtractionResponse(BaseModel):
    """Full extraction result returned to the frontend."""

    task_id: UUID
    filename: str
    strategy_used: str
    confidence: float
    metadata: InvoiceMetadata | None = None
    warnings: list[str] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=_utcnow)


class ErrorResponse(BaseModel):
    """Standard error payload."""

    detail: str
    code: str = "UNKNOWN_ERROR"
