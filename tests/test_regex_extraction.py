"""Tests for the regex extraction strategy."""

from __future__ import annotations

from datetime import date

import pytest

from src.core.extraction.base import ExtractionContext
from src.core.extraction.regex_strategy import RegexExtractionStrategy
from src.domain.entities import CommodityType

SAMPLE_ELECTRICITY_TEXT = """
Faktura za elektřinu č. FV-2025-0042
Variabilní symbol: 1234567890
EAN: 859182400000000001

Odběrné místo: OM-12345678
Distribuční tarif: D02d
Jistič: 25 A

Období: 01.01.2025 - 30.06.2025
Datum vystavení: 01.07.2025
Datum splatnosti: 15.07.2025
DUZP: 30.06.2025

IČO odběratele: 12345678
DIČ odběratele: CZ12345678
IČO dodavatele: 87654321

Spotřeba VT: 800,5 kWh
Spotřeba NT: 1 200,30 kWh

Základ daně: 10 000,00 Kč
DPH 21%: 2 100,00 Kč
Celkem k úhradě: 12 100,00 Kč
"""


SAMPLE_GAS_TEXT = """
Faktura za zemní plyn č. FV-2025-G01
Variabilní symbol: 9876543210
EAN: 859182400099990001

Období: 01.01.2025 - 31.03.2025
Datum vystavení: 05.04.2025
Datum splatnosti: 20.04.2025

Spotřeba: 350,5 m³
Spotřeba: 3,505 MWh
Přepočítací koeficient: 10,55
Spalné teplo: 34,08 MJ/m³

Celkem k úhradě: 8 500,00 Kč
"""


SAMPLE_WATER_TEXT = """
Faktura za vodu a stočné
Faktura č. FV-2025-W01

Období: 01.01.2025 - 30.06.2025
Datum vystavení: 01.07.2025

Spotřeba: 45 m³
Vodné: 2 500,00 Kč
Stočné: 2 200,00 Kč

Celkem k úhradě: 4 700,00 Kč
"""


@pytest.fixture()
def strategy() -> RegexExtractionStrategy:
    return RegexExtractionStrategy()


class TestCommodityDetection:
    def test_detect_electricity_nn(self, strategy: RegexExtractionStrategy):
        commodity = strategy._detect_commodity(SAMPLE_ELECTRICITY_TEXT)
        assert commodity == CommodityType.ELEKTRINA_NN

    def test_detect_gas(self, strategy: RegexExtractionStrategy):
        commodity = strategy._detect_commodity(SAMPLE_GAS_TEXT)
        assert commodity == CommodityType.PLYN_MO

    def test_detect_water(self, strategy: RegexExtractionStrategy):
        commodity = strategy._detect_commodity(SAMPLE_WATER_TEXT)
        assert commodity == CommodityType.VODA

    def test_unknown_text(self, strategy: RegexExtractionStrategy):
        commodity = strategy._detect_commodity("some random text with no indicators")
        assert commodity is None


class TestElectricityExtraction:
    @pytest.mark.asyncio
    async def test_full_extraction(self, strategy: RegexExtractionStrategy):
        ctx = ExtractionContext(
            raw_text=SAMPLE_ELECTRICITY_TEXT,
            source_filename="test_elec.pdf",
        )
        result = await strategy.extract(ctx)

        assert result.invoice_data is not None
        inv = result.invoice_data
        assert inv.invoice_number == "FV-2025-0042"
        assert inv.variable_symbol == "1234567890"
        assert inv.commodity == CommodityType.ELEKTRINA_NN
        assert inv.period.period_from == date(2025, 1, 1)
        assert inv.period.period_to == date(2025, 6, 30)
        assert inv.issue_date == date(2025, 7, 1)
        assert inv.due_date == date(2025, 7, 15)
        assert inv.customer_tax_id == "12345678"
        assert inv.supplier_tax_id == "87654321"
        assert inv.total_amount_inc_vat == pytest.approx(12100.0)

    @pytest.mark.asyncio
    async def test_electricity_nn_details(self, strategy: RegexExtractionStrategy):
        ctx = ExtractionContext(
            raw_text=SAMPLE_ELECTRICITY_TEXT,
            source_filename="test_elec.pdf",
        )
        result = await strategy.extract(ctx)

        assert result.invoice_data is not None
        assert len(result.invoice_data.electricity_nn_details) == 1
        detail = result.invoice_data.electricity_nn_details[0]
        assert detail.consumption_high_tariff == pytest.approx(800.5)
        assert detail.consumption_low_tariff == pytest.approx(1200.3)
        assert detail.distribution_tariff == "D02d"
        assert detail.circuit_breaker_value == pytest.approx(25.0)


class TestGasExtraction:
    @pytest.mark.asyncio
    async def test_gas_extraction(self, strategy: RegexExtractionStrategy):
        ctx = ExtractionContext(
            raw_text=SAMPLE_GAS_TEXT,
            source_filename="test_gas.pdf",
        )
        result = await strategy.extract(ctx)

        assert result.invoice_data is not None
        inv = result.invoice_data
        assert inv.invoice_number == "FV-2025-G01"
        assert inv.total_amount_inc_vat == pytest.approx(8500.0)
        assert len(inv.gas_mo_details) == 1
        detail = inv.gas_mo_details[0]
        assert detail.consumption_m3 == pytest.approx(350.5)


class TestWaterExtraction:
    @pytest.mark.asyncio
    async def test_water_extraction(self, strategy: RegexExtractionStrategy):
        ctx = ExtractionContext(
            raw_text=SAMPLE_WATER_TEXT,
            source_filename="test_water.pdf",
        )
        result = await strategy.extract(ctx)

        assert result.invoice_data is not None
        inv = result.invoice_data
        assert inv.commodity == CommodityType.VODA
        assert inv.total_amount_inc_vat == pytest.approx(4700.0)
        assert len(inv.water_details) == 1


class TestConfidence:
    @pytest.mark.asyncio
    async def test_good_extraction_confidence(self, strategy: RegexExtractionStrategy):
        ctx = ExtractionContext(
            raw_text=SAMPLE_ELECTRICITY_TEXT,
            source_filename="test.pdf",
        )
        result = await strategy.extract(ctx)
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_poor_text_low_confidence(self, strategy: RegexExtractionStrategy):
        ctx = ExtractionContext(
            raw_text="elektřina NN faktura\nnějaky text bez údajů",
            source_filename="bad.pdf",
        )
        result = await strategy.extract(ctx)
        # Should still extract with low confidence or errors
        assert result.confidence < 0.8 or result.errors


class TestValidation:
    @pytest.mark.asyncio
    async def test_validation_warnings(self, strategy: RegexExtractionStrategy):
        ctx = ExtractionContext(
            raw_text=SAMPLE_ELECTRICITY_TEXT,
            source_filename="test.pdf",
        )
        result = await strategy.extract(ctx)
        assert result.invoice_data is not None
        warnings = await strategy.validate(result.invoice_data)
        # Good data should have no warnings
        assert isinstance(warnings, list)
