"""Correction invoice (opravná faktura) analysis and linking.

Handles correction invoices, linking them to originals and
calculating delta differences between original and corrected values.

Opravná faktura - logika pro propojení s původní fakturou a výpočet rozdílů.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from uuid import UUID

from src.core.analysis.base import AnalysisReport, BaseAnalyser
from src.domain.entities import (
    CommodityType,
    CorrectionInfo,
    InvoiceData,
    InvoiceMetadata,
    InvoiceType,
)


logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class CorrectionDelta:
    """Represents differences between original and correction invoice.

    Attributes:
        field_name: Name of the changed field.
        original_value: Value from original invoice.
        corrected_value: Value from correction invoice.
        delta: Difference (corrected - original).
        delta_percent: Percentage change.
    """

    field_name: str
    original_value: Any
    corrected_value: Any
    delta: float | None = None
    delta_percent: float | None = None


@dataclass
class CorrectionAnalysisResult:
    """Result of correction invoice analysis.

    Attributes:
        is_correction: Whether this is a correction invoice.
        original_invoice_number: Invoice being corrected.
        original_invoice_id: UUID of original (if linked).
        correction_type: Type of correction (dobropis/vrubopis).
        deltas: List of field differences.
        total_delta: Net change in total amount.
        warnings: Any warnings generated.
    """

    is_correction: bool = False
    original_invoice_number: str | None = None
    original_invoice_id: UUID | None = None
    correction_type: str | None = None  # 'dobropis' (credit) or 'vrubopis' (debit)
    deltas: list[CorrectionDelta] = field(default_factory=list)
    total_delta: float | None = None
    warnings: list[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════════
# CORRECTION ANALYSER
# ════════════════════════════════════════════════════════════════════════════


class CorrectionAnalyser(BaseAnalyser):
    """Analyser for correction invoices (opravné faktury).

    Detects correction invoices, determines their type (credit/debit),
    and calculates deltas when linked to original invoice.

    Czech Terminology:
        - Opravná faktura: General correction invoice
        - Dobropis: Credit note (returns money to customer)
        - Vrubopis: Debit note (charges additional amount)
        - Storno: Full cancellation of original invoice

    Business Rules:
        - Corrections must reference an original invoice number
        - Dobropis typically has negative total_amount (refund)
        - Vrubopis typically has positive total_amount (charge)
        - Storno should have opposite sign to original
    """

    async def analyse(self, metadata: InvoiceMetadata) -> AnalysisReport:
        """Analyse invoice for correction status.

        Args:
            metadata: Validated invoice metadata (InvoiceData).

        Returns:
            AnalysisReport with is_correction flag and linked_invoice.
        """
        warnings: list[str] = []
        linked_invoice: str | None = None

        is_correction = metadata.is_correction or metadata.invoice_type == InvoiceType.CORRECTION

        if is_correction:
            # Try to get linked invoice reference
            if metadata.correction_info:
                linked_invoice = metadata.correction_info.original_invoice_number

            if not linked_invoice:
                warnings.append(
                    "Opravná faktura bez reference na původní fakturu - "
                    "nelze propojit s původním dokladem"
                )
            else:
                warnings.append(
                    f"Opravná faktura odkazující na: {linked_invoice}"
                )

        return AnalysisReport(
            is_correction=is_correction,
            is_transitional=metadata.is_transitional,
            linked_invoice=linked_invoice,
            cross_year=metadata.is_cross_year(),
            warnings=warnings,
        )

    def detect_correction_type(self, invoice: InvoiceData) -> CorrectionAnalysisResult:
        """Detect type of correction invoice and basic info.

        Args:
            invoice: Invoice to analyze.

        Returns:
            CorrectionAnalysisResult with type detection.
        """
        warnings: list[str] = []

        if not invoice.is_correction:
            return CorrectionAnalysisResult(
                is_correction=False,
            )

        # Determine correction type by amount sign
        correction_type: str | None = None
        total = invoice.total_amount_inc_vat or invoice.amount_to_pay

        if total is not None:
            if total < 0:
                correction_type = "dobropis"
                warnings.append(f"Dobropis (kredit): vratka {abs(total)} Kč")
            elif total > 0:
                correction_type = "vrubopis"
                warnings.append(f"Vrubopis (debet): doplatek {total} Kč")
            else:
                correction_type = "storno"
                warnings.append("Storno: nulová hodnota (zrušení původní faktury)")

        # Get original invoice reference
        original_number: str | None = None
        if invoice.correction_info:
            original_number = invoice.correction_info.original_invoice_number

        return CorrectionAnalysisResult(
            is_correction=True,
            original_invoice_number=original_number,
            correction_type=correction_type,
            total_delta=total,
            warnings=warnings,
        )

    def calculate_deltas(
        self,
        correction: InvoiceData,
        original: InvoiceData,
    ) -> CorrectionAnalysisResult:
        """Calculate differences between correction and original invoice.

        Args:
            correction: Correction invoice.
            original: Original invoice being corrected.

        Returns:
            CorrectionAnalysisResult with field-by-field deltas.
        """
        deltas: list[CorrectionDelta] = []
        warnings: list[str] = []

        # Verify this is actually a correction
        if not correction.is_correction:
            warnings.append("Invoice is not marked as correction")
            return CorrectionAnalysisResult(
                is_correction=False,
                warnings=warnings,
            )

        # Compare key monetary fields
        amount_fields = [
            ("total_amount_inc_vat", "Celková částka s DPH"),
            ("total_amount_ex_vat", "Celková částka bez DPH"),
            ("vat_amount", "DPH"),
            ("amount_to_pay", "K úhradě"),
        ]

        for field_name, czech_name in amount_fields:
            orig_val = getattr(original, field_name, None)
            corr_val = getattr(correction, field_name, None)

            if orig_val is not None or corr_val is not None:
                delta = self._calculate_field_delta(
                    field_name=czech_name,
                    original_value=orig_val,
                    corrected_value=corr_val,
                )
                if delta:
                    deltas.append(delta)

        # Compare commodity-specific consumption
        consumption_deltas = self._compare_consumption(correction, original)
        deltas.extend(consumption_deltas)

        # Calculate total delta
        total_delta: float | None = None
        if correction.total_amount_inc_vat is not None:
            if original.total_amount_inc_vat is not None:
                total_delta = correction.total_amount_inc_vat - original.total_amount_inc_vat
            else:
                total_delta = correction.total_amount_inc_vat

        # Determine correction type
        correction_type = None
        if total_delta is not None:
            if total_delta < 0:
                correction_type = "dobropis"
            elif total_delta > 0:
                correction_type = "vrubopis"
            else:
                correction_type = "neutrální"

        return CorrectionAnalysisResult(
            is_correction=True,
            original_invoice_number=original.invoice_number,
            original_invoice_id=original.id,
            correction_type=correction_type,
            deltas=deltas,
            total_delta=total_delta,
            warnings=warnings,
        )

    def _calculate_field_delta(
        self,
        field_name: str,
        original_value: float | None,
        corrected_value: float | None,
    ) -> CorrectionDelta | None:
        """Calculate delta for a single field.

        Args:
            field_name: Human-readable field name.
            original_value: Original invoice value.
            corrected_value: Correction invoice value.

        Returns:
            CorrectionDelta or None if both are None.
        """
        if original_value is None and corrected_value is None:
            return None

        orig = original_value or 0.0
        corr = corrected_value or 0.0
        delta = corr - orig

        # Calculate percentage change
        delta_percent: float | None = None
        if orig != 0:
            delta_percent = round((delta / abs(orig)) * 100, 2)

        return CorrectionDelta(
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            delta=round(delta, 2),
            delta_percent=delta_percent,
        )

    def _compare_consumption(
        self,
        correction: InvoiceData,
        original: InvoiceData,
    ) -> list[CorrectionDelta]:
        """Compare commodity-specific consumption values.

        Args:
            correction: Correction invoice.
            original: Original invoice.

        Returns:
            List of consumption deltas.
        """
        deltas: list[CorrectionDelta] = []

        # Electricity NN
        if correction.electricity_nn_details and original.electricity_nn_details:
            corr_detail = correction.electricity_nn_details[0]
            orig_detail = original.electricity_nn_details[0]

            if corr_detail.consumption_low_tariff or orig_detail.consumption_low_tariff:
                delta = self._calculate_field_delta(
                    "Spotřeba NT (kWh)",
                    orig_detail.consumption_low_tariff,
                    corr_detail.consumption_low_tariff,
                )
                if delta:
                    deltas.append(delta)

            if corr_detail.consumption_high_tariff or orig_detail.consumption_high_tariff:
                delta = self._calculate_field_delta(
                    "Spotřeba VT (kWh)",
                    orig_detail.consumption_high_tariff,
                    corr_detail.consumption_high_tariff,
                )
                if delta:
                    deltas.append(delta)

        # Electricity VN
        if correction.electricity_vn_details and original.electricity_vn_details:
            corr_detail = correction.electricity_vn_details[0]
            orig_detail = original.electricity_vn_details[0]

            if corr_detail.supply_consumption or orig_detail.supply_consumption:
                delta = self._calculate_field_delta(
                    "Spotřeba silové elektřiny (MWh)",
                    orig_detail.supply_consumption,
                    corr_detail.supply_consumption,
                )
                if delta:
                    deltas.append(delta)

        # Gas MO
        if correction.gas_mo_details and original.gas_mo_details:
            corr_detail = correction.gas_mo_details[0]
            orig_detail = original.gas_mo_details[0]

            if corr_detail.consumption_m3 or orig_detail.consumption_m3:
                delta = self._calculate_field_delta(
                    "Spotřeba plynu MO (m³)",
                    orig_detail.consumption_m3,
                    corr_detail.consumption_m3,
                )
                if delta:
                    deltas.append(delta)

            if corr_detail.consumption_mwh or orig_detail.consumption_mwh:
                delta = self._calculate_field_delta(
                    "Spotřeba plynu MO (MWh)",
                    orig_detail.consumption_mwh,
                    corr_detail.consumption_mwh,
                )
                if delta:
                    deltas.append(delta)

        # Gas VO
        if correction.gas_vo_details and original.gas_vo_details:
            corr_detail = correction.gas_vo_details[0]
            orig_detail = original.gas_vo_details[0]

            if corr_detail.consumption_m3 or orig_detail.consumption_m3:
                delta = self._calculate_field_delta(
                    "Spotřeba plynu VO (m³)",
                    orig_detail.consumption_m3,
                    corr_detail.consumption_m3,
                )
                if delta:
                    deltas.append(delta)

            if corr_detail.consumption_mwh or orig_detail.consumption_mwh:
                delta = self._calculate_field_delta(
                    "Spotřeba plynu VO (MWh)",
                    orig_detail.consumption_mwh,
                    corr_detail.consumption_mwh,
                )
                if delta:
                    deltas.append(delta)

        # Water
        if correction.water_details and original.water_details:
            corr_detail = correction.water_details[0]
            orig_detail = original.water_details[0]

            if corr_detail.consumption_m3 or orig_detail.consumption_m3:
                delta = self._calculate_field_delta(
                    "Spotřeba vody (m³)",
                    orig_detail.consumption_m3,
                    corr_detail.consumption_m3,
                )
                if delta:
                    deltas.append(delta)

        # Heat
        if correction.heat_details and original.heat_details:
            corr_detail = correction.heat_details[0]
            orig_detail = original.heat_details[0]

            if corr_detail.consumption_gj or orig_detail.consumption_gj:
                delta = self._calculate_field_delta(
                    "Spotřeba tepla (GJ)",
                    orig_detail.consumption_gj,
                    corr_detail.consumption_gj,
                )
                if delta:
                    deltas.append(delta)

            if corr_detail.heat_consumption or orig_detail.heat_consumption:
                delta = self._calculate_field_delta(
                    "Spotřeba tepla (kWh)",
                    orig_detail.heat_consumption,
                    corr_detail.heat_consumption,
                )
                if delta:
                    deltas.append(delta)

        return deltas

    def link_correction_to_original(
        self,
        correction: InvoiceData,
        original: InvoiceData,
    ) -> InvoiceData:
        """Link correction invoice to original and populate CorrectionInfo.

        Args:
            correction: Correction invoice to update.
            original: Original invoice to link to.

        Returns:
            Updated correction invoice with linked CorrectionInfo.
        """
        if correction.correction_info is None:
            correction.correction_info = CorrectionInfo(
                original_invoice_number=original.invoice_number,
            )
        else:
            correction.correction_info.original_invoice_number = original.invoice_number

        # Calculate and store deltas
        analysis = self.calculate_deltas(correction, original)

        # Update correction info with analysis results
        correction.correction_info.original_invoice_id = original.id
        correction.correction_info.correction_type = analysis.correction_type
        correction.correction_info.total_delta = analysis.total_delta

        return correction

    def format_delta_report(
        self,
        result: CorrectionAnalysisResult,
    ) -> str:
        """Format correction analysis as human-readable report.

        Args:
            result: Analysis result.

        Returns:
            Formatted string report.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("ANALÝZA OPRAVNÉ FAKTURY")
        lines.append("=" * 60)

        if result.original_invoice_number:
            lines.append(f"Původní faktura: {result.original_invoice_number}")

        if result.correction_type:
            type_names = {
                "dobropis": "DOBROPIS (kredit)",
                "vrubopis": "VRUBOPIS (debet)",
                "storno": "STORNO",
                "neutrální": "NEUTRÁLNÍ OPRAVA",
            }
            lines.append(f"Typ opravy: {type_names.get(result.correction_type, result.correction_type)}")

        if result.total_delta is not None:
            sign = "+" if result.total_delta >= 0 else ""
            lines.append(f"Celková změna: {sign}{result.total_delta:,.2f} Kč")

        if result.deltas:
            lines.append("")
            lines.append("Změny jednotlivých položek:")
            lines.append("-" * 40)

            for delta in result.deltas:
                orig_str = f"{delta.original_value:,.2f}" if delta.original_value else "N/A"
                corr_str = f"{delta.corrected_value:,.2f}" if delta.corrected_value else "N/A"

                delta_str = ""
                if delta.delta is not None:
                    sign = "+" if delta.delta >= 0 else ""
                    delta_str = f" ({sign}{delta.delta:,.2f}"
                    if delta.delta_percent is not None:
                        delta_str += f", {sign}{delta.delta_percent}%"
                    delta_str += ")"

                lines.append(f"  {delta.field_name}:")
                lines.append(f"    Původně: {orig_str}")
                lines.append(f"    Nově: {corr_str}{delta_str}")

        if result.warnings:
            lines.append("")
            lines.append("Upozornění:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")

        lines.append("=" * 60)
        return "\n".join(lines)


def create_correction_analyser() -> CorrectionAnalyser:
    """Factory function for CorrectionAnalyser."""
    return CorrectionAnalyser()
