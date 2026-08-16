"""Tests for transitional and correction analysis."""

from __future__ import annotations

from datetime import date

import pytest

from src.core.analysis.correction import CorrectionAnalyser
from src.core.analysis.transitional import TransitionalAnalyser
from src.domain.entities import (
    BillingPeriod,
    CommodityType,
    InvoiceData,
    InvoiceType,
)

# ── Transitional Analyser ───────────────────────────────────────


class TestTransitionalAnalyser:
    @pytest.fixture()
    def analyser(self) -> TransitionalAnalyser:
        return TransitionalAnalyser()

    @pytest.mark.asyncio
    async def test_same_year_not_transitional(self, analyser: TransitionalAnalyser, sample_invoice: InvoiceData):
        report = await analyser.analyse(sample_invoice)
        assert report.is_transitional is False
        assert report.cross_year is False

    @pytest.mark.asyncio
    async def test_cross_year_is_transitional(self, analyser: TransitionalAnalyser):
        invoice = InvoiceData(
            invoice_number="FV-CROSS",
            commodity=CommodityType.ELEKTRINA_NN,
            period=BillingPeriod(
                period_from=date(2024, 11, 1),
                period_to=date(2025, 1, 31),
            ),
            issue_date=date(2025, 2, 15),
        )
        report = await analyser.analyse(invoice)
        assert report.is_transitional is True
        assert report.cross_year is True
        assert len(report.warnings) > 0

    def test_calculate_split(self, analyser: TransitionalAnalyser):
        invoice = InvoiceData(
            invoice_number="FV-SPLIT",
            commodity=CommodityType.PLYN_MO,
            period=BillingPeriod(
                period_from=date(2024, 11, 1),
                period_to=date(2025, 1, 31),
            ),
            issue_date=date(2025, 2, 1),
            total_amount_inc_vat=9200.0,
        )
        result = analyser.calculate_split(invoice)

        assert result.is_cross_year is True
        assert len(result.splits) == 2

        # 2024 split: Nov 1 - Dec 31 = 61 days
        # 2025 split: Jan 1 - Jan 31 = 31 days
        # Total: 92 days
        assert result.total_days == 92
        assert result.splits[0].year == 2024
        assert result.splits[0].days == 61
        assert result.splits[1].year == 2025
        assert result.splits[1].days == 31

        # Check fractions sum to 1
        total_fraction = sum(s.fraction for s in result.splits)
        assert total_fraction == pytest.approx(1.0, abs=0.01)

        # Check amounts sum to original
        total_amount = sum(s.amount_fraction for s in result.splits if s.amount_fraction)
        assert total_amount == pytest.approx(9200.0, abs=1.0)

    def test_non_cross_year_no_split(self, analyser: TransitionalAnalyser, sample_invoice: InvoiceData):
        result = analyser.calculate_split(sample_invoice)
        assert result.is_cross_year is False
        assert len(result.splits) == 0


# ── Correction Analyser ─────────────────────────────────────────


class TestCorrectionAnalyser:
    @pytest.fixture()
    def analyser(self) -> CorrectionAnalyser:
        return CorrectionAnalyser()

    @pytest.mark.asyncio
    async def test_regular_invoice_not_correction(
        self, analyser: CorrectionAnalyser, sample_invoice: InvoiceData
    ):
        report = await analyser.analyse(sample_invoice)
        assert report.is_correction is False

    @pytest.mark.asyncio
    async def test_correction_invoice_detected(
        self, analyser: CorrectionAnalyser, correction_invoice: InvoiceData
    ):
        report = await analyser.analyse(correction_invoice)
        assert report.is_correction is True
        assert report.linked_invoice == "FV-2025-001"

    def test_detect_dobropis(self, analyser: CorrectionAnalyser, correction_invoice: InvoiceData):
        result = analyser.detect_correction_type(correction_invoice)
        assert result.is_correction is True
        assert result.correction_type == "dobropis"
        assert result.total_delta is not None
        assert result.total_delta < 0

    def test_detect_vrubopis(self, analyser: CorrectionAnalyser):
        inv = InvoiceData(
            invoice_number="FV-VRUB",
            commodity=CommodityType.ELEKTRINA_NN,
            period=BillingPeriod(period_from=date(2025, 1, 1), period_to=date(2025, 6, 30)),
            issue_date=date(2025, 7, 1),
            is_correction=True,
            invoice_type=InvoiceType.CORRECTION,
            total_amount_inc_vat=500.0,
        )
        result = analyser.detect_correction_type(inv)
        assert result.correction_type == "vrubopis"

    def test_calculate_deltas(
        self,
        analyser: CorrectionAnalyser,
        sample_invoice: InvoiceData,
        correction_invoice: InvoiceData,
    ):
        result = analyser.calculate_deltas(correction_invoice, sample_invoice)
        assert result.is_correction is True
        assert len(result.deltas) > 0
        assert result.total_delta is not None

    def test_link_correction(
        self,
        analyser: CorrectionAnalyser,
        sample_invoice: InvoiceData,
        correction_invoice: InvoiceData,
    ):
        linked = analyser.link_correction_to_original(correction_invoice, sample_invoice)
        assert linked.correction_info is not None
        assert linked.correction_info.original_invoice_number == "FV-2025-001"
        assert linked.correction_info.original_invoice_id == sample_invoice.id

    @pytest.mark.asyncio
    async def test_correction_without_reference(self, analyser: CorrectionAnalyser):
        inv = InvoiceData(
            invoice_number="FV-ORPHAN",
            commodity=CommodityType.VODA,
            period=BillingPeriod(period_from=date(2025, 1, 1), period_to=date(2025, 6, 30)),
            issue_date=date(2025, 7, 1),
            is_correction=True,
            invoice_type=InvoiceType.CORRECTION,
        )
        report = await analyser.analyse(inv)
        assert report.is_correction is True
        assert report.linked_invoice is None
        # Should warn about missing reference
        assert any("bez reference" in w for w in report.warnings)
