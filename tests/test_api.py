"""API endpoint integration tests.

These tests use an in-memory SQLite database and the FastAPI
test client (httpx.AsyncClient) to validate upload and result routes.
"""

from __future__ import annotations

import uuid

import pytest

# ── Health check ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check(client):
    """GET /health should return 200 with {"status": "ok"}."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Upload validation ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upload_unsupported_file_type(client):
    """Uploading a .txt file should return 415."""
    resp = await client.post(
        "/api/v1/upload/",
        files={"file": ("invoice.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415


@pytest.mark.asyncio
async def test_upload_no_file(client):
    """POST without a file should return 422."""
    resp = await client.post("/api/v1/upload/")
    assert resp.status_code == 422


# ── Results ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_result_not_found(client):
    """GET /results/{nonexistent} should return 404."""
    fake_id = uuid.uuid4()
    resp = await client.get(f"/api/v1/results/{fake_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_result_status_not_found(client):
    """GET /results/{nonexistent}/status should return 404."""
    fake_id = uuid.uuid4()
    resp = await client.get(f"/api/v1/results/{fake_id}/status")
    assert resp.status_code == 404


# ── DB persistence ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invoice_roundtrip(async_session):
    """Save an invoice to the DB and read it back."""
    from datetime import date as d

    from src.infrastructure.db.models import Invoice

    inv = Invoice(
        source_filename="test.pdf",
        invoice_number="FV-2025-001",
        commodity="elektrina_nn",
        period_from=d(2025, 1, 1),
        period_to=d(2025, 6, 30),
        issue_date=d(2025, 7, 1),
    )
    async_session.add(inv)
    await async_session.flush()

    from sqlalchemy import select

    result = await async_session.execute(
        select(Invoice).where(Invoice.invoice_number == "FV-2025-001")
    )
    fetched = result.scalar_one()
    assert fetched.source_filename == "test.pdf"
    assert fetched.commodity == "elektrina_nn"


@pytest.mark.asyncio
async def test_transaction_roundtrip(async_session):
    """Save a transaction and verify status field, created_at."""
    from src.infrastructure.db.models import Transaction

    tx = Transaction(
        filename="invoice.pdf",
        status="pending",
    )
    async_session.add(tx)
    await async_session.flush()

    from sqlalchemy import select

    result = await async_session.execute(
        select(Transaction).where(Transaction.filename == "invoice.pdf")
    )
    fetched = result.scalar_one()
    assert fetched.status == "pending"
    assert fetched.id is not None


@pytest.mark.asyncio
async def test_commodity_details_cascade(async_session):
    """ElectricityNNDetail should cascade-delete with its Invoice."""
    from datetime import date as d
    from decimal import Decimal

    from sqlalchemy import select

    from src.infrastructure.db.models import ElectricityNNDetail, Invoice

    inv = Invoice(
        source_filename="cascade_test.pdf",
        invoice_number="FV-CASCADE",
        commodity="elektrina_nn",
        period_from=d(2025, 1, 1),
        period_to=d(2025, 6, 30),
        issue_date=d(2025, 7, 1),
    )
    async_session.add(inv)
    await async_session.flush()

    detail = ElectricityNNDetail(
        invoice_id=inv.id,
        period_from=d(2025, 1, 1),
        period_to=d(2025, 6, 30),
        consumption_low_tariff=Decimal("1200.00"),
        consumption_high_tariff=Decimal("800.00"),
    )
    async_session.add(detail)
    await async_session.flush()

    # Verify detail exists
    result = await async_session.execute(
        select(ElectricityNNDetail).where(
            ElectricityNNDetail.invoice_id == inv.id
        )
    )
    assert result.scalar_one() is not None

    # Delete invoice — detail should cascade
    await async_session.delete(inv)
    await async_session.flush()

    result = await async_session.execute(
        select(ElectricityNNDetail).where(
            ElectricityNNDetail.invoice_id == inv.id
        )
    )
    assert result.scalar_one_or_none() is None
