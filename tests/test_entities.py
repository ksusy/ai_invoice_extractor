"""Unit tests for domain entities and OCR-string cleaning."""

from __future__ import annotations

from datetime import date

import pytest

from src.domain.entities import (
    BillingPeriod,
    ConsumptionRecord,
    MonetaryAmount,
    clean_czech_number,
    clean_czech_number_required,
)
from src.core.commodities.factory import CommoditySchemaFactory
from src.domain.constants import CommodityType


# ── clean_czech_number ───────────────────────────────────────────


class TestCleanCzechNumber:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("1 200,50", 1200.50),
            ("1.200,50", 1200.50),
            ("1200.50", 1200.50),
            ("0,99", 0.99),
            (42, 42.0),
            (3.14, 3.14),
            (None, None),
            ("", None),
        ],
    )
    def test_various_formats(self, raw, expected):
        result = clean_czech_number(raw)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)


class TestCleanNumericStringAlias:
    """Verify the backward-compatible alias works identically."""

    def test_alias_matches(self):
        assert clean_czech_number_required("1 200,50") == pytest.approx(1200.50)


# ── MonetaryAmount ───────────────────────────────────────────────


class TestMonetaryAmount:
    def test_dirty_string(self):
        m = MonetaryAmount(value="1 200,50")  # type: ignore[arg-type]
        assert m.value == pytest.approx(1200.50)
        assert m.currency == "CZK"


# ── BillingPeriod (ConsumptionRecord alias) ──────────────────────


class TestBillingPeriod:
    def test_basic_period(self):
        bp = BillingPeriod(
            period_from=date(2025, 1, 1),
            period_to=date(2025, 12, 31),
        )
        assert bp.period_from == date(2025, 1, 1)
        assert bp.period_to == date(2025, 12, 31)

    def test_cross_year(self):
        bp = BillingPeriod(
            period_from=date(2024, 11, 1),
            period_to=date(2025, 1, 31),
        )
        assert bp.is_cross_year() is True

    def test_same_year(self):
        bp = BillingPeriod(
            period_from=date(2025, 1, 1),
            period_to=date(2025, 6, 30),
        )
        assert bp.is_cross_year() is False

    def test_alias_identity(self):
        """ConsumptionRecord should be the same class as BillingPeriod."""
        assert ConsumptionRecord is BillingPeriod


# ── CommoditySchemaFactory ──────────────────────────────────────


class TestCommodityFactory:
    def test_all_commodities_registered(self):
        for ct in CommodityType:
            schema_cls = CommoditySchemaFactory.get_schema_class(ct)
            assert schema_cls is not None
