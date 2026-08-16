"""Transitional (cross-year) invoice analysis and splitting.

Handles invoices where the billing period spans December 31 -> January 1,
splitting consumption proportionally between years.

Přechodová faktura - logika pro rozdělení faktur přes přelom roku.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import TypeVar

from src.core.analysis.base import AnalysisReport, BaseAnalyser
from src.domain.entities import (
    BillingPeriod,
    CommodityType,
    ElectricityNNData,
    ElectricityVNData,
    GasMOData,
    GasVOData,
    HeatData,
    InvoiceData,
    InvoiceMetadata,
    WaterData,
)


logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class YearSplit:
    """Represents a single year's portion of a cross-year invoice.

    Attributes:
        year: The calendar year.
        period: Billing period for this year's portion.
        days: Number of days in this year's portion.
        fraction: Proportion of total period (0.0 to 1.0).
        amount_fraction: Proportional amount for this year (CZK).
    """

    year: int
    period: BillingPeriod
    days: int
    fraction: float
    amount_fraction: float | None = None


@dataclass
class TransitionalSplitResult:
    """Result of transitional invoice year splitting.

    Attributes:
        is_cross_year: Whether the invoice spans multiple years.
        original_period: Original billing period.
        splits: List of YearSplit objects, one per year.
        total_days: Total days in original period.
        warnings: Any warnings generated during splitting.
    """

    is_cross_year: bool
    original_period: BillingPeriod
    splits: list[YearSplit] = field(default_factory=list)
    total_days: int = 0
    warnings: list[str] = field(default_factory=list)


# Type variable for commodity detail classes
T = TypeVar("T", ElectricityNNData, ElectricityVNData, GasMOData, WaterData, HeatData)


# ════════════════════════════════════════════════════════════════════════════
# TRANSITIONAL ANALYSER
# ════════════════════════════════════════════════════════════════════════════


class TransitionalAnalyser(BaseAnalyser):
    """Analyser for transitional (cross-year) invoices.

    Detects invoices spanning year boundaries and calculates
    proportional splits for consumption and amounts.

    Business Rules:
        - Period spanning Dec 31 -> Jan 1 triggers split
        - Consumption is split proportionally by day count
        - Amounts are split proportionally by day count
        - Each year's split maintains original commodity type
    """

    async def analyse(self, metadata: InvoiceMetadata) -> AnalysisReport:
        """Analyse invoice for cross-year status.

        The analyser detects whether the billing period spans multiple
        calendar years (cross_year). However, it does NOT automatically
        flag the invoice as transitional — that relies on the LLM
        detecting "přechodová faktura" in the text. Regular annual
        invoices that happen to cross year boundaries are NOT transitional.

        Args:
            metadata: Validated invoice metadata (InvoiceData).

        Returns:
            AnalysisReport with cross_year flag. is_transitional is
            preserved from the LLM extraction result.
        """
        warnings: list[str] = []

        is_cross_year = metadata.is_cross_year()

        # Any invoice whose billing period crosses a year boundary
        # is automatically considered transitional (přechodová faktura).
        is_transitional = metadata.is_transitional or is_cross_year

        if is_cross_year and metadata.period:
            years = self._get_years_in_period(metadata.period)
            warnings.append(
                f"Faktura přes přelom roku "
                f"({metadata.period.period_from} - {metadata.period.period_to}), "
                f"roky: {', '.join(map(str, years))}"
            )

        return AnalysisReport(
            is_correction=metadata.is_correction,
            is_transitional=is_transitional,
            linked_invoice=None,
            cross_year=is_cross_year,
            warnings=warnings,
        )

    def _get_years_in_period(self, period: BillingPeriod) -> list[int]:
        """Get list of years covered by billing period.

        Args:
            period: Billing period to analyze.

        Returns:
            Sorted list of years.
        """
        if not period.period_from or not period.period_to:
            return []

        start_year = period.period_from.year
        end_year = period.period_to.year

        return list(range(start_year, end_year + 1))

    def calculate_split(self, invoice: InvoiceData) -> TransitionalSplitResult:
        """Calculate year splits for cross-year invoice.

        Args:
            invoice: Invoice to split.

        Returns:
            TransitionalSplitResult with proportional splits per year.
        """
        period = invoice.period
        warnings: list[str] = []

        if not period.period_from or not period.period_to:
            return TransitionalSplitResult(
                is_cross_year=False,
                original_period=period,
                warnings=["Billing period dates missing"],
            )

        if not invoice.is_cross_year():
            return TransitionalSplitResult(
                is_cross_year=False,
                original_period=period,
            )

        # Calculate total days
        total_days = (period.period_to - period.period_from).days + 1

        if total_days <= 0:
            warnings.append(f"Invalid period: {period.period_from} to {period.period_to}")
            return TransitionalSplitResult(
                is_cross_year=False,
                original_period=period,
                warnings=warnings,
            )

        # Generate splits for each year
        splits: list[YearSplit] = []
        years = self._get_years_in_period(period)

        for year in years:
            split = self._calculate_year_split(
                period=period,
                year=year,
                total_days=total_days,
                total_amount=invoice.total_amount_inc_vat,
            )
            splits.append(split)

        logger.info(
            f"Split invoice {invoice.invoice_number} across {len(splits)} years: "
            f"{[(s.year, s.fraction) for s in splits]}"
        )

        return TransitionalSplitResult(
            is_cross_year=True,
            original_period=period,
            splits=splits,
            total_days=total_days,
            warnings=warnings,
        )

    def _calculate_year_split(
        self,
        period: BillingPeriod,
        year: int,
        total_days: int,
        total_amount: float | None,
    ) -> YearSplit:
        """Calculate split for a single year.

        Args:
            period: Original billing period.
            year: Target year.
            total_days: Total days in original period.
            total_amount: Total amount to split.

        Returns:
            YearSplit for the specified year.
        """
        # Determine year boundaries within period
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)

        # Clamp to period boundaries
        split_start = max(period.period_from, year_start)
        split_end = min(period.period_to, year_end)

        # Calculate days in this year's portion
        days_in_year = (split_end - split_start).days + 1
        fraction = days_in_year / total_days

        # Calculate proportional amount
        amount_fraction = None
        if total_amount is not None:
            amount_fraction = round(total_amount * fraction, 2)

        return YearSplit(
            year=year,
            period=BillingPeriod(period_from=split_start, period_to=split_end),
            days=days_in_year,
            fraction=round(fraction, 4),
            amount_fraction=amount_fraction,
        )

    def split_commodity_details(
        self,
        invoice: InvoiceData,
        split_result: TransitionalSplitResult,
    ) -> dict[int, InvoiceData]:
        """Split invoice into separate records per year.

        Creates a new InvoiceData for each year with proportionally
        split consumption values.

        Args:
            invoice: Original invoice to split.
            split_result: Pre-calculated split fractions.

        Returns:
            Dictionary mapping year -> InvoiceData.
        """
        if not split_result.is_cross_year:
            year = invoice.period.period_from.year if invoice.period else 0
            return {year: invoice}

        result: dict[int, InvoiceData] = {}

        for year_split in split_result.splits:
            # Create copy with adjusted values
            year_invoice = self._create_year_invoice(
                invoice=invoice,
                year_split=year_split,
            )
            result[year_split.year] = year_invoice

        return result

    def _create_year_invoice(
        self,
        invoice: InvoiceData,
        year_split: YearSplit,
    ) -> InvoiceData:
        """Create invoice record for a single year's portion.

        Args:
            invoice: Original invoice.
            year_split: Year split information.

        Returns:
            New InvoiceData for the year portion.
        """
        from copy import deepcopy
        from uuid import uuid4

        # Deep copy to avoid mutating original
        year_invoice = deepcopy(invoice)

        # Update identifiers
        year_invoice.id = uuid4()
        year_invoice.period = year_split.period
        year_invoice.is_transitional = True

        # Scale amounts
        if year_invoice.total_amount_inc_vat is not None:
            year_invoice.total_amount_inc_vat = round(
                year_invoice.total_amount_inc_vat * year_split.fraction, 2
            )
        if year_invoice.total_amount_ex_vat is not None:
            year_invoice.total_amount_ex_vat = round(
                year_invoice.total_amount_ex_vat * year_split.fraction, 2
            )
        if year_invoice.vat_amount is not None:
            year_invoice.vat_amount = round(
                year_invoice.vat_amount * year_split.fraction, 2
            )
        if year_invoice.amount_to_pay is not None:
            year_invoice.amount_to_pay = round(
                year_invoice.amount_to_pay * year_split.fraction, 2
            )

        # Scale commodity-specific consumption
        self._scale_commodity_details(year_invoice, year_split)

        return year_invoice

    def _scale_commodity_details(
        self,
        invoice: InvoiceData,
        year_split: YearSplit,
    ) -> None:
        """Scale commodity-specific consumption values.

        Args:
            invoice: Invoice to modify (mutated in place).
            year_split: Year split with fraction.
        """
        fraction = year_split.fraction

        # Electricity NN
        for detail in invoice.electricity_nn_details:
            detail.period = year_split.period
            if detail.consumption_low_tariff is not None:
                detail.consumption_low_tariff = round(
                    detail.consumption_low_tariff * fraction, 2
                )
            if detail.consumption_high_tariff is not None:
                detail.consumption_high_tariff = round(
                    detail.consumption_high_tariff * fraction, 2
                )
            if detail.total_consumption is not None:
                detail.total_consumption = round(
                    detail.total_consumption * fraction, 2
                )
            if detail.amount_inc_vat is not None:
                detail.amount_inc_vat = round(detail.amount_inc_vat * fraction, 2)
            if detail.amount_ex_vat is not None:
                detail.amount_ex_vat = round(detail.amount_ex_vat * fraction, 2)

        # Electricity VN
        for detail in invoice.electricity_vn_details:
            detail.period = year_split.period
            if detail.supply_consumption is not None:
                detail.supply_consumption = round(
                    detail.supply_consumption * fraction, 3
                )
            if detail.peak_consumption is not None:
                detail.peak_consumption = round(detail.peak_consumption * fraction, 3)
            if detail.off_peak_consumption is not None:
                detail.off_peak_consumption = round(
                    detail.off_peak_consumption * fraction, 3
                )
            if detail.amount_inc_vat is not None:
                detail.amount_inc_vat = round(detail.amount_inc_vat * fraction, 2)

        # Gas MO
        for detail in invoice.gas_mo_details:
            detail.period = year_split.period
            if detail.consumption_m3 is not None:
                detail.consumption_m3 = round(detail.consumption_m3 * fraction, 2)
            if detail.consumption_mwh is not None:
                detail.consumption_mwh = round(detail.consumption_mwh * fraction, 3)
            if detail.amount_inc_vat is not None:
                detail.amount_inc_vat = round(detail.amount_inc_vat * fraction, 2)
            if detail.amount_ex_vat is not None:
                detail.amount_ex_vat = round(detail.amount_ex_vat * fraction, 2)

        # Gas VO
        for detail in invoice.gas_vo_details:
            detail.period = year_split.period
            if detail.consumption_m3 is not None:
                detail.consumption_m3 = round(detail.consumption_m3 * fraction, 2)
            if detail.consumption_mwh is not None:
                detail.consumption_mwh = round(detail.consumption_mwh * fraction, 3)
            if detail.other_supply_services_price is not None:
                detail.other_supply_services_price = round(detail.other_supply_services_price * fraction, 2)
            if detail.trade_reserved_capacity_price is not None:
                detail.trade_reserved_capacity_price = round(detail.trade_reserved_capacity_price * fraction, 2)
            if detail.distribution_service_price is not None:
                detail.distribution_service_price = round(detail.distribution_service_price * fraction, 2)
            if detail.distribution_reserved_capacity_price is not None:
                detail.distribution_reserved_capacity_price = round(detail.distribution_reserved_capacity_price * fraction, 2)
            if detail.market_operator_price is not None:
                detail.market_operator_price = round(detail.market_operator_price * fraction, 2)
            if detail.natural_gas_tax_total is not None:
                detail.natural_gas_tax_total = round(detail.natural_gas_tax_total * fraction, 2)
            if detail.amount_inc_vat is not None:
                detail.amount_inc_vat = round(detail.amount_inc_vat * fraction, 2)
            if detail.amount_ex_vat is not None:
                detail.amount_ex_vat = round(detail.amount_ex_vat * fraction, 2)

        # Water
        for detail in invoice.water_details:
            detail.period = year_split.period
            if detail.consumption_m3 is not None:
                detail.consumption_m3 = round(detail.consumption_m3 * fraction, 2)
            if detail.water_rate is not None:
                detail.water_rate = round(detail.water_rate * fraction, 2)
            if detail.sewage_rate is not None:
                detail.sewage_rate = round(detail.sewage_rate * fraction, 2)
            if detail.amount_inc_vat is not None:
                detail.amount_inc_vat = round(detail.amount_inc_vat * fraction, 2)
            if detail.amount_ex_vat is not None:
                detail.amount_ex_vat = round(detail.amount_ex_vat * fraction, 2)

        # Heat
        for detail in invoice.heat_details:
            detail.period = year_split.period
            if detail.consumption_gj is not None:
                detail.consumption_gj = round(detail.consumption_gj * fraction, 2)
            if detail.heat_consumption is not None:
                detail.heat_consumption = round(detail.heat_consumption * fraction, 2)
            if detail.hot_water_heating is not None:
                detail.hot_water_heating = round(detail.hot_water_heating * fraction, 2)
            if detail.total_heat_consumption is not None:
                detail.total_heat_consumption = round(detail.total_heat_consumption * fraction, 2)
            if detail.amount_inc_vat is not None:
                detail.amount_inc_vat = round(detail.amount_inc_vat * fraction, 2)
            if detail.amount_ex_vat is not None:
                detail.amount_ex_vat = round(detail.amount_ex_vat * fraction, 2)


def create_transitional_analyser() -> TransitionalAnalyser:
    """Factory function for TransitionalAnalyser."""
    return TransitionalAnalyser()
