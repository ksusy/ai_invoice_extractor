"""Regex-based extraction strategy for structured invoice data.

Uses rule-based regular expressions to extract fields from OCR text.
This is the baseline strategy for comparison with LLM-based approaches.

Extrakční strategie založená na regulárních výrazech.
"""

from __future__ import annotations

import re
from datetime import date

from src.core.extraction.base import BaseExtractionStrategy, ExtractionContext
from src.domain.entities import (
    BillingPeriod,
    CommodityType,
    ElectricityNNData,
    ElectricityVNData,
    ExtractionResult,
    GasMOData,
    GasVOData,
    HeatData,
    InvoiceData,
    InvoiceType,
    SupplyPoint,
    WaterData,
    clean_czech_number,
    parse_czech_date,
)

# ════════════════════════════════════════════════════════════════════════════
# REGEX PATTERNS FOR CZECH INVOICES
# ════════════════════════════════════════════════════════════════════════════


# Common patterns
PATTERNS = {
    # Invoice identification
    "invoice_number": [
        r"(?:Číslo\s+(?:faktury|dokladu)|Faktura(?:\s+\w+){0,3}\s+č\.?|Doklad\s+č\.?)"
        r"\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-/]*\d[A-Za-z0-9\-/]*)",
        r"(?:Invoice\s+(?:No|Number)|Invoice\s*#)"
        r"\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-/]*\d[A-Za-z0-9\-/]*)",
    ],
    "variable_symbol": [
        r"(?:Variabilní\s+symbol|VS|Var\.?\s*sym\.?)\s*[:\-]?\s*(\d{6,12})",
    ],

    # Supply point identifiers
    "ean_code": [
        r"(?:EAN|Kód\s+EAN)\s*[:\-]?\s*(\d{13,18})",
        r"(859\d{15})",  # Czech EAN pattern
    ],
    "eic_code": [
        r"(?:EIC|Kód\s+EIC)\s*[:\-]?\s*(27ZG[A-Z0-9]{12})",
    ],
    "consumption_point_code": [
        r"(?:Odběrné\s+místo|Kód\s+odběrného\s+místa)\s*[:\-]?\s*([A-Z0-9\-]{8,20})",
    ],

    # Dates (DD.MM.YYYY format)
    "issue_date": [
        r"(?:Datum\s+vystavení|Vystaveno)\s*[:\-]?\s*(\d{1,2}\.\d{1,2}\.\d{4})",
    ],
    "due_date": [
        r"(?:Datum\s+splatnosti|Splatnost)\s*[:\-]?\s*(\d{1,2}\.\d{1,2}\.\d{4})",
    ],
    "vat_date": [
        r"(?:Datum\s+(?:uskutečnění\s+)?zdanitelného\s+plnění|DUZP|Datum\s+UZP)\s*[:\-]?\s*(\d{1,2}\.\d{1,2}\.\d{4})",
    ],

    # Party identification
    "customer_tax_id": [
        r"(?:IČO?\s+odběratele|IČO?(?!\s+dodavatele))\s*[:\-]?\s*(\d{8})",
    ],
    "supplier_tax_id": [
        r"(?:IČO?\s+dodavatele)\s*[:\-]?\s*(\d{8})",
    ],
    "customer_vat_id": [
        r"(?:DIČ\s+odběratele|DIČ\s*[:\-]?\s*)(CZ\d{8,10})",
    ],
    "supplier_vat_id": [
        r"(?:DIČ\s+dodavatele)\s*[:\-]?\s*(CZ\d{8,10})",
    ],

    # Billing period
    "period": [
        r"(?:Období|Zúčtovací\s+období|Fakturační\s+období)\s*[:\-]?\s*(\d{1,2}\.\d{1,2}\.\d{4})\s*[-–]\s*(\d{1,2}\.\d{1,2}\.\d{4})",
        r"od\s+(\d{1,2}\.\d{1,2}\.\d{4})\s+do\s+(\d{1,2}\.\d{1,2}\.\d{4})",
    ],

    # Amounts (CZK)
    "total_amount_inc_vat": [
        r"(?:Celkem\s+(?:k\s+)?(?:úhradě|zaplacení)|K\s+úhradě|Celková\s+částka)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:Kč|CZK)?",
        r"(?:Částka\s+celkem)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:Kč|CZK)?",
    ],
    "total_amount_ex_vat": [
        r"(?:Základ\s+daně|Celkem\s+bez\s+DPH)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:Kč|CZK)?",
    ],
    "vat_amount": [
        r"(?:DPH\s+(?:\d+\s*%)?|Daň)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:Kč|CZK)?",
    ],

    # Electricity NN specific
    "consumption_low_tariff": [
        r"(?:Spotřeba\s+NT|Nízký\s+tarif|NT)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:kWh)?",
    ],
    "consumption_high_tariff": [
        r"(?:Spotřeba\s+VT|Vysoký\s+tarif|VT)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:kWh)?",
    ],
    "distribution_tariff": [
        r"(?:Distribuční\s+tarif|Sazba)\s*[:\-]?\s*(D\d{2}[a-z]?)",
    ],
    "circuit_breaker": [
        r"(?:Jistič|Hlavní\s+jistič)\s*[:\-]?\s*(\d+)\s*(?:A|Amp)?",
    ],

    # Gas specific
    "consumption_m3": [
        r"(?:Spotřeba|Odběr)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*m[³3]",
    ],
    "consumption_mwh": [
        r"(?:Spotřeba|Odběr)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*MWh",
    ],
    "conversion_factor": [
        r"(?:Přepočítací\s+koeficient|Koef\.?\s+přepočtu)\s*[:\-]?\s*([\d\s]+[,.]?\d*)",
    ],
    "combustion_heat": [
        r"(?:Spalné\s+teplo)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:MJ/m[³3])?",
    ],

    # Water specific
    "water_rate": [
        r"(?:Vodné)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:Kč|CZK)?",
    ],
    "sewage_rate": [
        r"(?:Stočné)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:Kč|CZK)?",
    ],

    # Heat specific
    "consumption_gj": [
        r"(?:Spotřeba|Odběr)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*GJ",
    ],
    "heated_area": [
        r"(?:Vytápěná\s+plocha)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*m[²2]",
    ],
}

# Commodity detection patterns
# Zkratky komodit (NN, VN, MO, VO) musí být ohraničené hranicí slova. Bez ní
# se „VO“ shoduje uvnitř slova „voda“ nebo „vodné“ a faktura za vodu je pak
# chybně klasifikována jako plyn velkoodběr.
COMMODITY_PATTERNS = {
    CommodityType.ELEKTRINA_NN: [
        r"(?i)elektřina|elektrina|elektrická\s+energie|silová\s+elektřina",
        r"(?i)nízké\s+napětí|nn|low\s+voltage",
        r"(?i)distribuční\s+tarif\s*[:\-]?\s*D\d{2}",
    ],
    CommodityType.ELEKTRINA_VN: [
        r"(?i)vysoké\s+napětí|vn|high\s+voltage",
        r"(?i)rezervovaná\s+kapacita",
    ],
    CommodityType.PLYN_MO: [
        r"(?i)plyn|zemní\s+plyn|natural\s+gas",
        r"(?i)maloodběr|MO",
    ],
    CommodityType.PLYN_VO: [
        r"(?i)velkoodběr|VO",
    ],
    CommodityType.TEPLO: [
        r"(?i)teplo|dálkové\s+vytápění|district\s+heat",
        r"(?i)GJ|gigajoul",
    ],
    CommodityType.VODA: [
        r"(?i)voda|vodné|stočné|water",
        r"(?i)vodovod|kanalizace",
    ],
}


class RegexExtractionStrategy(BaseExtractionStrategy):
    """Baseline extraction using regular expressions.

    This strategy provides a rule-based approach to field extraction.
    It serves as a baseline for comparison with ML/LLM approaches.

    Limitations:
        - Highly dependent on consistent invoice formatting
        - May miss fields with unusual layouts
        - Cannot handle completely novel formats
    """

    @property
    def name(self) -> str:
        """Return strategy name identifier."""
        return "regex"

    def _search_patterns(
        self,
        text: str,
        pattern_list: list[str],
        group: int = 1,
    ) -> str | None:
        """Search text using multiple regex patterns.

        Args:
            text: Text to search.
            pattern_list: List of regex patterns to try.
            group: Capture group to return (default: 1).

        Returns:
            First match found, or None.
        """
        for pattern in pattern_list:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return match.group(group).strip()
                except IndexError:
                    return match.group(0).strip()
        return None

    def _search_patterns_all(
        self,
        text: str,
        pattern_list: list[str],
    ) -> list[str]:
        """Find all matches for patterns.

        Returns list of all matches found.
        """
        results = []
        for pattern in pattern_list:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            results.extend(matches)
        return results

    def _detect_commodity(self, text: str) -> CommodityType | None:
        """Detect commodity type from invoice text.

        Uses pattern matching to identify the utility type.
        """
        scores: dict[CommodityType, int] = {}

        for commodity, patterns in COMMODITY_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1
            if score > 0:
                scores[commodity] = score

        if not scores:
            return None

        # Return commodity with highest score
        return max(scores, key=scores.get)

    def _extract_period(self, text: str) -> BillingPeriod | None:
        """Extract billing period from text."""
        for pattern in PATTERNS["period"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    date_from = parse_czech_date(match.group(1))
                    date_to = parse_czech_date(match.group(2))
                    if date_from and date_to:
                        return BillingPeriod(period_from=date_from, period_to=date_to)
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_electricity_nn(self, text: str) -> ElectricityNNData | None:
        """Extract Electricity NN specific fields."""
        consumption_nt = self._search_patterns(text, PATTERNS["consumption_low_tariff"])
        consumption_vt = self._search_patterns(text, PATTERNS["consumption_high_tariff"])
        tariff = self._search_patterns(text, PATTERNS["distribution_tariff"])
        breaker = self._search_patterns(text, PATTERNS["circuit_breaker"])

        # Only return if we found at least one field
        if any([consumption_nt, consumption_vt, tariff, breaker]):
            return ElectricityNNData(
                consumption_low_tariff=clean_czech_number(consumption_nt),
                consumption_high_tariff=clean_czech_number(consumption_vt),
                distribution_tariff=tariff,
                circuit_breaker_value=clean_czech_number(breaker),
                period=self._extract_period(text),
            )
        return None

    def _extract_electricity_vn(self, text: str) -> ElectricityVNData | None:
        """Extract Electricity VN specific fields.

        Extracts reserved capacity, reactive power, grid usage,
        and other high-voltage-specific fields.
        """
        # VN-specific regex patterns
        vn_patterns = {
            "supply_consumption": [
                r"(?:Spotřeba\s+(?:silové\s+)?(?:elektřiny|SE))\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:MWh)",
            ],
            "annual_reserved_capacity": [
                r"(?:Roční\s+rezervovaná\s+kapacita|RK\s+roční)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:MW|kW)",
            ],
            "monthly_reserved_capacity": [
                r"(?:Měsíční\s+rezervovaná\s+kapacita|RK\s+měsíční)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:MW|kW)",
            ],
            "reactive_power_quantity": [
                r"(?:Jalová\s+energie|Jalový\s+výkon|Množství\s+jalové)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:kVArh)",
            ],
            "power_factor": [
                r"(?:Účiník|tg\s*[φfi])\s*[:\-]?\s*([\d\s]+[,.]?\d*)",
            ],
            "quarter_hour_max": [
                r"(?:Čtvrthodinové\s+maximum|1/4\s*hod\.?\s*max)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:MW|kW)",
            ],
            "grid_usage_rate": [
                r"(?:Sazba\s+(?:za\s+)?použití\s+sítí?)\s*[:\-]?\s*([\d\s]+[,.]?\d*)",
            ],
            "service_price": [
                r"(?:Cena\s+(?:za\s+)?služb[yí])\s*[:\-]?\s*([\d\s]+[,.]?\d*)",
            ],
            "renewable_energy_fee": [
                r"(?:POZE|Podpora\s+(?:obnovitelných|OZE))\s*[:\-]?\s*([\d\s]+[,.]?\d*)",
            ],
        }

        fields: dict[str, float | None] = {}
        for field_name, patterns in vn_patterns.items():
            raw = self._search_patterns(text, patterns)
            fields[field_name] = clean_czech_number(raw)

        period = self._extract_period(text)

        # Only return if we found at least one VN-specific field
        if any(v is not None for v in fields.values()) or period:
            return ElectricityVNData(
                period=period,
                supply_consumption=fields.get("supply_consumption"),
                annual_reserved_capacity=fields.get("annual_reserved_capacity"),
                monthly_reserved_capacity=fields.get("monthly_reserved_capacity"),
                reactive_power_quantity=fields.get("reactive_power_quantity"),
                power_factor=fields.get("power_factor"),
                quarter_hour_max=fields.get("quarter_hour_max"),
                grid_usage_rate=fields.get("grid_usage_rate"),
                service_price=fields.get("service_price"),
                renewable_energy_fee=fields.get("renewable_energy_fee"),
            )
        return None

    def _extract_gas_mo(self, text: str) -> GasMOData | None:
        """Extract Gas MO specific fields."""
        consumption_m3 = self._search_patterns(text, PATTERNS["consumption_m3"])
        consumption_mwh = self._search_patterns(text, PATTERNS["consumption_mwh"])
        conversion = self._search_patterns(text, PATTERNS["conversion_factor"])
        combustion = self._search_patterns(text, PATTERNS["combustion_heat"])

        if any([consumption_m3, consumption_mwh, conversion, combustion]):
            return GasMOData(
                consumption_m3=clean_czech_number(consumption_m3),
                consumption_mwh=clean_czech_number(consumption_mwh),
                conversion_factor=clean_czech_number(conversion),
                combustion_heat=clean_czech_number(combustion),
                period=self._extract_period(text),
            )
        return None

    def _extract_gas_vo(self, text: str) -> GasVOData | None:
        """Extract Gas VO (large-scale consumer) specific fields."""
        consumption_m3 = self._search_patterns(text, PATTERNS["consumption_m3"])
        consumption_mwh = self._search_patterns(text, PATTERNS["consumption_mwh"])
        conversion = self._search_patterns(text, PATTERNS["conversion_factor"])
        combustion = self._search_patterns(text, PATTERNS["combustion_heat"])

        # VO-specific patterns
        vo_patterns = {
            "daily_reserved_capacity": [
                r"(?:Denní\s+rezervovaná\s+kapacita)\s*[:\-]?\s*([\d\s]+[,.]?\d*)\s*(?:m[³3]/den)?",
            ],
            "market_operator_price": [
                r"(?:Činnost\s+operátora\s+trhu|OTE)\s*[:\-]?\s*([\d\s]+[,.]?\d*)",
            ],
            "natural_gas_tax_total": [
                r"(?:Daň\s+(?:ze\s+)?zemního\s+plynu)\s*[:\-]?\s*([\d\s]+[,.]?\d*)",
            ],
        }

        vo_fields: dict[str, float | None] = {}
        for field_name, patterns in vo_patterns.items():
            raw = self._search_patterns(text, patterns)
            vo_fields[field_name] = clean_czech_number(raw)

        if any([consumption_m3, consumption_mwh, conversion, combustion]) or any(
            v is not None for v in vo_fields.values()
        ):
            return GasVOData(
                consumption_m3=clean_czech_number(consumption_m3),
                consumption_mwh=clean_czech_number(consumption_mwh),
                conversion_factor=clean_czech_number(conversion),
                combustion_heat=clean_czech_number(combustion),
                daily_reserved_capacity=vo_fields.get("daily_reserved_capacity"),
                market_operator_price=vo_fields.get("market_operator_price"),
                natural_gas_tax_total=vo_fields.get("natural_gas_tax_total"),
                period=self._extract_period(text),
            )
        return None

    def _extract_water(self, text: str) -> WaterData | None:
        """Extract Water specific fields."""
        consumption = self._search_patterns(text, PATTERNS["consumption_m3"])
        water_rate = self._search_patterns(text, PATTERNS["water_rate"])
        sewage_rate = self._search_patterns(text, PATTERNS["sewage_rate"])

        if any([consumption, water_rate, sewage_rate]):
            return WaterData(
                consumption_m3=clean_czech_number(consumption),
                water_rate=clean_czech_number(water_rate),
                sewage_rate=clean_czech_number(sewage_rate),
                period=self._extract_period(text),
            )
        return None

    def _extract_heat(self, text: str) -> HeatData | None:
        """Extract Heat specific fields."""
        consumption = self._search_patterns(text, PATTERNS["consumption_gj"])
        heated_area = self._search_patterns(text, PATTERNS["heated_area"])

        if any([consumption, heated_area]):
            return HeatData(
                consumption_gj=clean_czech_number(consumption),
                heated_area=clean_czech_number(heated_area),
                period=self._extract_period(text),
            )
        return None

    async def extract(self, context: ExtractionContext) -> ExtractionResult:
        """Extract structured data from OCR text using regex patterns.

        Args:
            context: ExtractionContext with raw text and metadata.

        Returns:
            ExtractionResult with parsed InvoiceData.
        """
        text = context.raw_text
        errors: list[str] = []
        warnings: list[str] = []

        # Detect commodity type
        commodity = context.commodity_hint or self._detect_commodity(text)
        if not commodity:
            errors.append("Could not detect commodity type from invoice text")
            return ExtractionResult(
                source_file=context.source_filename,
                strategy_name=self.name,
                raw_text=text,
                errors=errors,
            )

        # Extract common fields
        invoice_number = self._search_patterns(text, PATTERNS["invoice_number"])
        if not invoice_number:
            errors.append("Missing required field: invoice_number")

        variable_symbol = self._search_patterns(text, PATTERNS["variable_symbol"])

        # Extract period
        period = self._extract_period(text)
        if not period:
            # Use default period if not found
            warnings.append("Could not extract billing period, using default")
            period = BillingPeriod(
                period_from=date.today().replace(day=1),
                period_to=date.today(),
            )

        # Extract dates
        issue_date = parse_czech_date(
            self._search_patterns(text, PATTERNS["issue_date"])
        )
        if not issue_date:
            warnings.append("Could not extract issue_date, using today")
            issue_date = date.today()

        due_date = parse_czech_date(
            self._search_patterns(text, PATTERNS["due_date"])
        )
        vat_date = parse_czech_date(
            self._search_patterns(text, PATTERNS["vat_date"])
        )

        # Extract supply point
        supply_point = SupplyPoint(
            ean_code=self._search_patterns(text, PATTERNS["ean_code"]) or "",
            eic_code=self._search_patterns(text, PATTERNS["eic_code"]) or "",
            consumption_point_code=self._search_patterns(
                text, PATTERNS["consumption_point_code"]
            ) or "",
        )

        # Extract amounts
        total_inc_vat = clean_czech_number(
            self._search_patterns(text, PATTERNS["total_amount_inc_vat"])
        )
        total_ex_vat = clean_czech_number(
            self._search_patterns(text, PATTERNS["total_amount_ex_vat"])
        )
        vat_amount = clean_czech_number(
            self._search_patterns(text, PATTERNS["vat_amount"])
        )

        # Extract party information
        customer_tax_id = self._search_patterns(text, PATTERNS["customer_tax_id"])
        supplier_tax_id = self._search_patterns(text, PATTERNS["supplier_tax_id"])
        customer_vat_id = self._search_patterns(text, PATTERNS["customer_vat_id"])
        supplier_vat_id = self._search_patterns(text, PATTERNS["supplier_vat_id"])

        # Build invoice data
        try:
            invoice_data = InvoiceData(
                source_filename=context.source_filename,
                invoice_number=invoice_number or "UNKNOWN",
                variable_symbol=variable_symbol,
                commodity=commodity,
                invoice_type=InvoiceType.REGULAR,
                supply_point=supply_point,
                period=period,
                issue_date=issue_date,
                due_date=due_date,
                vat_date=vat_date,
                customer_tax_id=customer_tax_id,
                customer_vat_id=customer_vat_id,
                supplier_tax_id=supplier_tax_id,
                supplier_vat_id=supplier_vat_id,
                total_amount_inc_vat=total_inc_vat,
                total_amount_ex_vat=total_ex_vat,
                vat_amount=vat_amount,
            )

            # Extract commodity-specific details
            if commodity == CommodityType.ELEKTRINA_NN:
                detail = self._extract_electricity_nn(text)
                if detail:
                    invoice_data.electricity_nn_details = [detail]
            elif commodity == CommodityType.ELEKTRINA_VN:
                detail = self._extract_electricity_vn(text)
                if detail:
                    invoice_data.electricity_vn_details = [detail]
            elif commodity == CommodityType.PLYN_MO:
                detail = self._extract_gas_mo(text)
                if detail:
                    invoice_data.gas_mo_details = [detail]
            elif commodity == CommodityType.PLYN_VO:
                detail = self._extract_gas_vo(text)
                if detail:
                    invoice_data.gas_vo_details = [detail]
            elif commodity == CommodityType.VODA:
                detail = self._extract_water(text)
                if detail:
                    invoice_data.water_details = [detail]
            elif commodity == CommodityType.TEPLO:
                detail = self._extract_heat(text)
                if detail:
                    invoice_data.heat_details = [detail]

            # Calculate confidence based on fields found
            confidence = self._calculate_confidence(invoice_data, errors, warnings)

            return ExtractionResult(
                source_file=context.source_filename,
                strategy_name=self.name,
                confidence=confidence,
                raw_text=text,
                invoice_data=invoice_data,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(f"Failed to build InvoiceData: {str(e)}")
            return ExtractionResult(
                source_file=context.source_filename,
                strategy_name=self.name,
                raw_text=text,
                errors=errors,
                warnings=warnings,
            )

    def _calculate_confidence(
        self,
        invoice_data: InvoiceData,
        errors: list[str],
        warnings: list[str],
    ) -> float:
        """Calculate extraction confidence score.

        Based on:
        - Number of required fields found
        - Number of errors/warnings
        - Completeness of commodity-specific details
        """
        score = 1.0

        # Deduct for errors
        score -= len(errors) * 0.2

        # Deduct for warnings (less severe)
        score -= len(warnings) * 0.05

        # Check required fields
        if invoice_data.invoice_number == "UNKNOWN":
            score -= 0.15

        if not invoice_data.supply_point.ean_code and not invoice_data.supply_point.eic_code:
            score -= 0.1

        if not invoice_data.total_amount_inc_vat:
            score -= 0.15

        # Cap between 0 and 1
        return max(0.0, min(1.0, score))

    async def validate(self, invoice_data: InvoiceData) -> list[str]:
        """Validate extracted invoice data.

        Checks for:
        - Required fields present
        - Valid date ranges
        - Reasonable monetary amounts
        """
        warnings: list[str] = []

        # Check required identifiers
        if not invoice_data.invoice_number or invoice_data.invoice_number == "UNKNOWN":
            warnings.append("Missing invoice number")

        if (
            not invoice_data.supply_point.ean_code
            and not invoice_data.supply_point.eic_code
            and not invoice_data.supply_point.consumption_point_code
        ):
            warnings.append("No supply point identifier found (EAN/EIC/code)")

        # Validate dates
        if (invoice_data.due_date and invoice_data.issue_date
                and invoice_data.due_date < invoice_data.issue_date):
            warnings.append("Due date is before issue date")

        # Validate amounts
        if invoice_data.total_amount_inc_vat is not None:
            if invoice_data.total_amount_inc_vat < 0 and not invoice_data.is_correction:
                warnings.append("Total amount is negative (not a correction invoice)")
            if invoice_data.total_amount_inc_vat > 10_000_000:
                warnings.append("Total amount seems unusually high (>10M CZK)")

        return warnings


def create_regex_strategy() -> RegexExtractionStrategy:
    """Factory function to create a RegexExtractionStrategy instance."""
    return RegexExtractionStrategy()
