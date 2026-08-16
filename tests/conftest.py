"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from datetime import date

import pytest

from src.domain.entities import (
    BillingPeriod,
    CommodityType,
    CorrectionInfo,
    ElectricityNNData,
    ExtractionResult,
    GasMOData,
    InvoiceData,
    InvoiceType,
    SupplyPoint,
)

# ── Database fixtures (SQLite async for fast tests) ──────────────


@pytest.fixture()
async def async_engine():
    """Create an in-memory SQLite async engine for testing."""
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )
    from src.infrastructure.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture()
async def async_session(async_engine):
    """Yield an async SQLAlchemy session bound to the test engine."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture()
def app(async_engine):
    """Create a FastAPI test app with the test DB wired in."""

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.infrastructure.db.database import get_async_session
    from src.main import create_app

    test_app = create_app()

    session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async def override_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    test_app.dependency_overrides[get_async_session] = override_session
    return test_app


@pytest.fixture()
async def client(app):
    """Async HTTP client for testing FastAPI endpoints."""
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Domain fixtures ──────────────────────────────────────────────


@pytest.fixture()
def sample_billing_period() -> BillingPeriod:
    """A simple same-year billing period."""
    return BillingPeriod(period_from=date(2025, 1, 1), period_to=date(2025, 6, 30))


@pytest.fixture()
def cross_year_billing_period() -> BillingPeriod:
    """A billing period that spans two calendar years."""
    return BillingPeriod(period_from=date(2024, 11, 1), period_to=date(2025, 1, 31))


@pytest.fixture()
def sample_invoice(sample_billing_period: BillingPeriod) -> InvoiceData:
    """A minimal valid InvoiceData instance for electricity NN."""
    return InvoiceData(
        invoice_number="FV-2025-001",
        variable_symbol="1234567890",
        commodity=CommodityType.ELEKTRINA_NN,
        period=sample_billing_period,
        issue_date=date(2025, 7, 1),
        due_date=date(2025, 7, 15),
        supply_point=SupplyPoint(ean_code="859182400000000001"),
        total_amount_ex_vat=10000.0,
        total_amount_inc_vat=12100.0,
        vat_amount=2100.0,
        amount_to_pay=12100.0,
        electricity_nn_details=[
            ElectricityNNData(
                period=sample_billing_period,
                consumption_low_tariff=1200.0,
                consumption_high_tariff=800.0,
                total_consumption=2000.0,
                distribution_tariff="D02d",
                circuit_breaker_value=25.0,
                amount_ex_vat=10000.0,
                amount_inc_vat=12100.0,
            )
        ],
    )


@pytest.fixture()
def correction_invoice(sample_billing_period: BillingPeriod) -> InvoiceData:
    """A correction (opravná) invoice linked to FV-2025-001."""
    return InvoiceData(
        invoice_number="FV-2025-002",
        commodity=CommodityType.ELEKTRINA_NN,
        period=sample_billing_period,
        issue_date=date(2025, 7, 10),
        is_correction=True,
        invoice_type=InvoiceType.CORRECTION,
        correction_info=CorrectionInfo(
            original_invoice_number="FV-2025-001",
        ),
        total_amount_ex_vat=-500.0,
        total_amount_inc_vat=-605.0,
        amount_to_pay=-605.0,
    )


@pytest.fixture()
def sample_gas_invoice(sample_billing_period: BillingPeriod) -> InvoiceData:
    """A minimal gas MO invoice."""
    return InvoiceData(
        invoice_number="FV-2025-G01",
        commodity=CommodityType.PLYN_MO,
        period=sample_billing_period,
        issue_date=date(2025, 7, 1),
        total_amount_inc_vat=8500.0,
        gas_mo_details=[
            GasMOData(
                period=sample_billing_period,
                consumption_m3=350.0,
                consumption_mwh=3.5,
                conversion_factor=10.55,
                amount_inc_vat=8500.0,
            )
        ],
    )


@pytest.fixture()
def sample_extraction_result(sample_invoice: InvoiceData) -> ExtractionResult:
    """ExtractionResult wrapping the sample invoice."""
    return ExtractionResult(
        source_file="test_invoice.pdf",
        strategy_name="regex",
        confidence=0.85,
        invoice_data=sample_invoice,
    )
