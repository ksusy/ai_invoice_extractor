"""Pydantic V2 domain models with strict field validators.

These models represent the structured data extracted from utility
invoices.  ``@field_validator`` methods handle "dirty" OCR strings
(e.g. ``"1 200,50"`` → ``1200.50``) commonly produced by Czech-locale OCR.

Czech field naming convention mapped to English:
    See docs/domain_mapping_cz_en.md for complete mapping.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from enum import Enum

from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════


def clean_czech_number(raw: str | float | int | Decimal | None) -> float | None:
    """Normalise a Czech-formatted OCR number string into a Python float.

    Handles patterns like:
        - "1 200,50"   -> 1200.50  (space as thousands separator, comma decimal)
        - "1.200,50"   -> 1200.50  (dot as thousands separator, comma decimal)
        - "1200.50"    -> 1200.50  (standard format)
        - "-1 200,50"  -> -1200.50 (negative numbers)
        - "1 200"      -> 1200.0   (whole numbers with spaces)

    Returns None if input is None or empty string.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float, Decimal)):
        return float(raw)

    cleaned = str(raw).strip()
    if not cleaned:
        return None

    # Remove all whitespace (including non-breaking spaces)
    cleaned = re.sub(r"[\s\u00a0]+", "", cleaned)

    # Handle Czech format: comma as decimal separator and dot/space as thousands separator
    if "," in cleaned:
        # Remove dots used as thousands separators (before the comma)
        cleaned = cleaned.replace(".", "")
        # Replace comma with dot for decimal
        cleaned = cleaned.replace(",", ".")
    elif "." in cleaned:
        # No comma present — check if dot is a Czech thousands separator.
        # Pattern: digits.3digits(.3digits)* with NO trailing fractional part
        # e.g. "1.200" → 1200, "1.200.000" → 1200000
        # but  "1.20"  → 1.20  (normal decimal)
        if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", cleaned):
            cleaned = cleaned.replace(".", "")

    # Safety: keep only digits, single dot, and leading minus
    if not re.fullmatch(r"-?\d+\.?\d*", cleaned):
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None

def clean_czech_number_required(raw: str | float | int | Decimal) -> float:
    """Same as clean_czech_number but raises on None/empty."""
    result = clean_czech_number(raw)
    if result is None:
        raise ValueError(f"Cannot parse number from: {raw!r}")
    return result


# NOTE: legacy alias `_clean_numeric_string` removed — use `clean_czech_number_required` directly


def parse_czech_date(raw: str | date | None) -> date | None:
    """Parse a Czech-formatted date string (DD.MM.YYYY).

    Returns None if input is None or cannot be parsed.
    """
    if raw is None:
        return None
    if isinstance(raw, date):
        return raw

    cleaned = str(raw).strip()
    if not cleaned:
        return None

    # Normalise separators: replace / and - with dots for uniform handling
    cleaned = cleaned.replace("/", ".").replace("-", ".")
    # Remove optional spaces after dots (Czech style: "24. 12. 2023")
    cleaned = re.sub(r"\.\s+", ".", cleaned)

    # Try DD.MM.YYYY format
    match = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})$", cleaned)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            return None

    # Try DD.MM.YY (short year — OCR artefact)
    match = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{2})$", cleaned)
    if match:
        day, month = int(match.group(1)), int(match.group(2))
        short_year = int(match.group(3))
        year = 2000 + short_year if short_year < 80 else 1900 + short_year
        try:
            return date(year, month, day)
        except ValueError:
            return None

    # Try YYYY-MM-DD / YYYY.MM.DD (ISO format)
    match = re.match(r"(\d{4})\.(\d{1,2})\.(\d{1,2})$", cleaned)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            return None

    return None


def clean_consumption_point_code(raw: str | None) -> str | None:
    """Normalise consumption point code (kod_odberne_misto).

    Preserves alphanumeric characters to support different code formats:
    - Electricity EAN: 18 digits starting with 859
    - Gas EIC: 16 alphanumeric chars starting with 27ZG
    - Water/Heat: provider-specific (digits, sometimes with letters)

    Strategy:
    - Strip leading/trailing whitespace.
    - Remove everything except [A-Za-z0-9].
    - Return cleaned string or None if empty.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Keep alphanumeric characters (preserves EIC codes like 27ZG...)
    cleaned = re.sub(r"[^A-Za-z0-9]", "", s)
    return cleaned or None

## варто зробити для кожної комодити окремо, - тому що комжна комодита має свій специфічний формат й не хочемо компроміси 
# Електрина - містить 18 знаків, тільки цифри й починається на 859 (код Чехії)
# Плин - 16 znaků (kombinace čísel a písmen), v ČR vždy začíná 27ZG.
# вода: Formát: Obvykle se jedná o kombinaci čísel, někdy i písmen, v závislosti na konkrétním dodavateli vody.
# тепло = не знаю формат 

def clean_tax_id(raw: str | int | None) -> str | None:
    """Normalise Czech IČO (identification number).

    - Remove non-digit characters.
    - Reject if more than 8 digits (likely a phone number / bank account).
    - Return zero-padded 8-digit string to preserve leading zeros.
      e.g. "1234567" → "01234567" (IČO always displayed as 8 digits in Czech law).
    - Return None for empty / unparsable inputs.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    digits = re.sub(r"\D+", "", s)
    if not digits:
        return None
    if len(digits) > 8:
        return None  # too many digits — not a valid IČO
    return digits.zfill(8)  # zero-pad to 8 chars, preserving leading zeros


def validate_ean_electricity(ean: str | None) -> str | None:
    """Validate Czech electricity EAN supply point code.

    Czech EAN: exactly 18 digits, starts with 859 (Czech country prefix).
    Returns cleaned digits string or the original if validation fails
    (to avoid discarding values that might be valid but unusual).
    """
    if ean is None:
        return None
    digits = re.sub(r"\D", "", ean)
    if len(digits) == 18 and digits.startswith("859"):
        return digits
    return ean.strip() or None


def validate_eic_gas(eic: str | None) -> str | None:
    """Validate Czech gas EIC supply point code.

    Czech EIC: 16 alphanumeric characters, starts with 27ZG.
    Returns normalised uppercase string or the original if unusual.
    """
    if eic is None:
        return None
    cleaned = re.sub(r"\s", "", eic).upper()
    if len(cleaned) == 16 and cleaned.startswith("27ZG"):
        return cleaned
    return eic.strip() or None


# ════════════════════════════════════════════════════════════════════════════
# ENUMS
# ════════════════════════════════════════════════════════════════════════════


class CommodityType(str, Enum):
    """Supported commodity (utility) types."""

    ELEKTRINA_NN = "elektrina_nn"  # Electricity – low voltage
    ELEKTRINA_VN = "elektrina_vn"  # Electricity – high voltage
    PLYN_MO = "plyn_MO"           # Gas – small-scale consumer
    PLYN_VO = "plyn_VO"           # Gas – large-scale consumer
    TEPLO = "teplo"               # Heat
    VODA = "voda"                 # Water


class InvoiceType(str, Enum):
    """Classification of invoice purpose."""

    REGULAR = "regular"
    CORRECTION = "correction"      # opravná faktura
    TRANSITIONAL = "transitional"  # přechodová faktura — only for elektrina_nn and plyn_MO


# ════════════════════════════════════════════════════════════════════════════
# BASE MODELS
# ════════════════════════════════════════════════════════════════════════════


class CzechNumericModel(BaseModel):
    """Base model with Czech number cleaning for all float fields."""

    model_config = ConfigDict(
        strict=False,
        str_strip_whitespace=True,
        validate_default=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# VALUE OBJECTS
# ════════════════════════════════════════════════════════════════════════════


class MonetaryAmount(CzechNumericModel):
    """Financial amount with currency (částka).

    Czech: castka_bez_dph, castka_s_dph
    """

    value: float
    currency: str = "CZK"

    @field_validator("value", mode="before")
    @classmethod
    def clean_value(cls, v: str | float | int | Decimal) -> float:
        return clean_czech_number_required(v)


class MeterReading(CzechNumericModel):
    """Single meter reading record (stav měřidla)."""

    reading_date: date
    value: float
    unit: str = "kWh"

    @field_validator("value", mode="before")
    @classmethod
    def clean_value(cls, v: str | float | int | Decimal) -> float:
        return clean_czech_number_required(v)

    @field_validator("reading_date", mode="before")
    @classmethod
    def parse_date(cls, v: str | date) -> date:
        result = parse_czech_date(v)
        if result is None:
            raise ValueError(f"Cannot parse date: {v!r}")
        return result


# DEPRECATED: MeterReading — not used in current extraction pipeline;
# kept for backward compatibility. Use consumption fields in commodity detail models.
class AddressInfo(BaseModel):
    """Postal / supply-point address (adresa)."""

    street: str = ""
    city: str = ""
    postal_code: str = ""
    country: str = "CZ"


# DEPRECATED: AddressInfo — not used for invoice extraction;
# kept for backward compatibility.
class SupplyPoint(BaseModel):
    """Identification of a supply / metering point.

    Czech: kod_odberne_misto, EAN, EIC
    """

    consumption_point_code: str = ""  # kod_odberne_misto
    ean_code: str = ""   # DEPRECATED: EAN — everything uses consumption_point_code
    eic_code: str = ""   # DEPRECATED: EIC — everything uses consumption_point_code
    address: AddressInfo = Field(default_factory=AddressInfo)  # DEPRECATED

    @field_validator("consumption_point_code", mode="before")
    @classmethod
    def _clean_consumption_point_code(cls, v: str | None) -> str | None:
        if v is None:
            return None
        result = clean_consumption_point_code(v)
        # Return empty string instead of None to keep model type consistent
        return result or ""

# все буде - код одьерне місто, не обов'язкво вказувати еан еіц


class BillingPeriod(CzechNumericModel):
    """Billing period information (fakturační období).

    Czech: obdobi_od, obdobi_do
    """

    period_from: date  # obdobi_od
    period_to: date    # obdobi_do

    @field_validator("period_from", "period_to", mode="before")
    @classmethod
    def parse_date(cls, v: str | date) -> date:
        result = parse_czech_date(v)
        if result is None:
            raise ValueError(f"Cannot parse date: {v!r}")
        return result

    @model_validator(mode="after")
    def validate_period(self) -> "BillingPeriod":
        """Ensure period_from is before or equal to period_to."""
        if self.period_from > self.period_to:
            raise ValueError(
                f"period_from ({self.period_from}) must be <= period_to ({self.period_to})"
            )
        return self

    def is_cross_year(self) -> bool:
        """Check if the period spans multiple calendar years."""
        return self.period_from.year != self.period_to.year

# Важливо: для преходових фактур period визначається на етапі класифікації,
# потім розбивається на два періоди в detail-таблицях.


class CorrectionInfo(BaseModel):
    """Metadata for correction invoices (opravná faktura).

    Czech: opravna, kterou_fakturu_opravuje
    """

    original_invoice_number: str  # kterou_fakturu_opravuje
    original_invoice_id: UUID | None = None
    correction_reason: str = ""
    correction_type: str | None = None  # dobropis / vrubopis / storno
    total_delta: float | None = None  # net monetary change (CZK)
    difference_amount: MonetaryAmount | None = None

# TODO: CorrectionInfo — перевірити повноту логіки (dobropis / vrubopis / storno).

# ════════════════════════════════════════════════════════════════════════════
# COMMODITY DETAIL MODELS
# ════════════════════════════════════════════════════════════════════════════


class ElectricityNNData(CzechNumericModel):
    """Electricity low voltage detail (elektřina nízké napětí).

    Czech field mapping:
        - spotreba_nt -> consumption_low_tariff
        - spotreba_vt -> consumption_high_tariff
        - jistic -> circuit_breaker_value - не потріьно, видалити, за потреби будемо розширювати потім б чи можна зберігати, але не буде надсилатись потім на ендпоінт 
        - distribucni_tarif -> distribution_tariff - те саме що й вище, видалити, за потреби будемо розширювати потімб чи можна зберігати, але не буде надсилатись потім на ендпоінт
    """

    period: BillingPeriod | None = None

    # Consumption (kWh)
    consumption_low_tariff: float | None = None   # spotreba_nt
    consumption_high_tariff: float | None = None  # spotreba_vt
    total_consumption: float | None = None # не потрібне

    # Meter readings
    meter_reading_start: float | None = None # не потрібне
    meter_reading_end: float | None = None # не потрібне

    # Technical
    distribution_tariff: str | None = None  # distribucni_tarif (D01d, D02d, etc.) не потрібне 
    circuit_breaker_value: float | None = None  # jistic (A) не потрібне

    # Amounts (CZK)
    amount_ex_vat: float | None = None   # castka_bez_dph - у преходових фактур важлива щоб тут була частка за окремі періоди, будуть два записи в базі даних, на два періоди й ця частка має бути на два різних періоди
    amount_inc_vat: float | None = None  # castka_s_dph - у преходових фактур важлива щоб тут була частка за окремі періоди, будуть два записи в базі даних, на два періоди й ця частка має бути на два різних періоди

    # Component breakdown - не потрібно для нн 
    supply_charge: float | None = None        # silová elektřina
    distribution_charge: float | None = None  # distribuce
    system_services: float | None = None      # systémové služby
    renewable_energy_fee: float | None = None # POZE

    # а тут не вистачає doklad_cislo, kod_odberne_misto, ico_odberatel, ico_dodavatel, flag opravna, prechodova , datum_uzp	datetime, datum_vystaveni	datetime,datum_splatnosti	datetime, nazev_souboru	varchar(255)б чи це тільки для специфічних полів тут? 

    @field_validator(
        "consumption_low_tariff",
        "consumption_high_tariff",
        "total_consumption",
        "meter_reading_start",
        "meter_reading_end",
        "circuit_breaker_value",
        "amount_ex_vat",
        "amount_inc_vat",
        "supply_charge",
        "distribution_charge",
        "system_services",
        "renewable_energy_fee",
        mode="before",
    )
    @classmethod
    def clean_numbers(cls, v: str | float | int | Decimal | None) -> float | None:
        return clean_czech_number(v)


class ElectricityVNData(CzechNumericModel):
    """Electricity high voltage detail (elektřina vysoké napětí).

    Czech field mapping:
        - spotreba_se -> supply_consumption
        - castka_se -> supply_charge
        - castka_dan_se -> supply_tax_charge
        - ctvrt_hod_max -> quarter_hour_max
        - sazba_eru -> eru_rate
        - rk_rocni -> annual_reserved_capacity
        - castka_rk_rocni -> annual_reserved_capacity_charge
        - rk_mesicni -> monthly_reserved_capacity
        - castka_rk_mesicni -> monthly_reserved_capacity_charge
        - sazba_pouziti_siti -> grid_usage_rate
        - castka_pouziti_siti -> grid_usage_charge
        - prekroceni_rk -> reserved_capacity_excess
        - sazba_prekroceni_rk -> reserved_capacity_excess_rate
        - tg_fi_vn -> power_factor
        - mnozstvi_jalova -> reactive_power_quantity
        - sazba_jalova -> reactive_power_rate
        - castka_jalova -> reactive_power_charge
        - cena_sluzby -> service_price
        - cena_provoz -> operating_price
        - poze -> renewable_energy_fee
    """

    period: BillingPeriod | None = None

    # Consumption (MWh)
    supply_consumption: float | None = None       # spotreba_se
    peak_consumption: float | None = None
    off_peak_consumption: float | None = None

    # Supply charges
    supply_charge: float | None = None            # castka_se
    supply_tax_charge: float | None = None        # castka_dan_se

    # Quarter-hour maximum
    quarter_hour_max: float | None = None         # ctvrt_hod_max
    eru_rate: float | None = None                 # sazba_eru

    # Reserved capacity (MW)
    annual_reserved_capacity: float | None = None  # rk_rocni
    annual_reserved_capacity_charge: float | None = None  # castka_rk_rocni
    monthly_reserved_capacity: float | None = None  # rk_mesicni
    monthly_reserved_capacity_charge: float | None = None  # castka_rk_mesicni

    # Grid usage
    grid_usage_rate: float | None = None          # sazba_pouziti_siti
    grid_usage_charge: float | None = None        # castka_pouziti_siti

    # Capacity excess
    reserved_capacity_excess: float | None = None      # prekroceni_rk
    reserved_capacity_excess_rate: float | None = None  # sazba_prekroceni_rk
    reserved_capacity_excess_charge: float | None = None  # castka_prekroceni_rk

    # Reactive power (kVArh)
    power_factor: float | None = None             # tg_fi_vn
    reactive_power_quantity: float | None = None  # mnozstvi_jalova
    reactive_power_rate: float | None = None      # sazba_jalova
    reactive_power_charge: float | None = None    # castka_jalova

    # Fees / services
    service_price: float | None = None            # cena_sluzby
    operating_price: float | None = None          # cena_provoz
    renewable_energy_fee: float | None = None     # poze

    # Amounts (CZK)
    amount_ex_vat: float | None = None
    amount_inc_vat: float | None = None

    @field_validator(
        "supply_consumption", "peak_consumption", "off_peak_consumption",
        "supply_charge", "supply_tax_charge",
        "quarter_hour_max", "eru_rate",
        "annual_reserved_capacity", "annual_reserved_capacity_charge",
        "monthly_reserved_capacity", "monthly_reserved_capacity_charge",
        "grid_usage_rate", "grid_usage_charge",
        "reserved_capacity_excess", "reserved_capacity_excess_rate",
        "reserved_capacity_excess_charge",
        "power_factor", "reactive_power_quantity",
        "reactive_power_rate", "reactive_power_charge",
        "service_price", "operating_price", "renewable_energy_fee",
        "amount_ex_vat", "amount_inc_vat",
        mode="before",
    )
    @classmethod
    def clean_numbers(cls, v: str | float | int | Decimal | None) -> float | None:
        return clean_czech_number(v)


class GasMOData(CzechNumericModel):
    """Gas small-scale consumer detail (plyn maloodběr).

    Czech field mapping:
        - koef_prepoctu -> conversion_factor
        - spalne_teplo -> combustion_heat
        - spotreba_mwh -> consumption_mwh
        - spotreba_m3 -> consumption_m3
        - obdobi_mesice -> period_months
        - dan_zemni_plyn_celkem -> natural_gas_tax_total
        - jednotkova_cena_za_komoditni_slozku_ceny -> commodity_unit_price
        - cena_za_komoditni_slozku_ceny -> commodity_total_price
        - jednotkova_cena_za_staly_mesicni_plat -> fixed_monthly_fee_unit_price
        - cena_za_staly_mesicni_plat -> fixed_monthly_fee
        - jedn_cena_distr_plynu -> distribution_unit_price
        - pevna_cena_za_distribuci_plynu -> distribution_fixed_price
        - jedn_cena_pristavena_kapacita -> reserved_capacity_unit_price
        - cena_za_pristavenou_kapacitu -> reserved_capacity_price
        - cena_za_cinnost_operatora_trhu -> market_operator_price
    """

    period: BillingPeriod | None = None
    period_months: int | None = None  # obdobi_mesice

    # Consumption
    consumption_m3: float | None = None   # spotreba_m3
    consumption_mwh: float | None = None  # spotreba_mwh

    # Conversion parameters
    conversion_factor: float | None = None  # koef_prepoctu
    combustion_heat: float | None = None    # spalne_teplo (MJ/m³)

    # DEPRECATED: meter readings
    meter_reading_start: float | None = None  # DEPRECATED
    meter_reading_end: float | None = None    # DEPRECATED

    # Commodity pricing
    commodity_unit_price: float | None = None    # jednotkova_cena_za_komoditni_slozku_ceny
    commodity_total_price: float | None = None   # cena_za_komoditni_slozku_ceny
    commodity_charge: float | None = None        # komodita (legacy alias)

    # Fixed monthly fee
    fixed_monthly_fee_unit_price: float | None = None  # jednotkova_cena_za_staly_mesicni_plat
    fixed_monthly_fee: float | None = None             # cena_za_staly_mesicni_plat / stálý plat

    # Distribution
    distribution_unit_price: float | None = None   # jedn_cena_distr_plynu
    distribution_fixed_price: float | None = None  # pevna_cena_za_distribuci_plynu
    distribution_charge: float | None = None       # distribuce (legacy alias)

    # Reserved capacity
    reserved_capacity_unit_price: float | None = None  # jedn_cena_pristavena_kapacita
    reserved_capacity_price: float | None = None       # cena_za_pristavenou_kapacitu

    # Other fees
    market_operator_price: float | None = None  # cena_za_cinnost_operatora_trhu
    natural_gas_tax_total: float | None = None  # dan_zemni_plyn_celkem

    # Amounts (CZK)
    amount_ex_vat: float | None = None
    amount_inc_vat: float | None = None

    @field_validator(
        "consumption_m3", "consumption_mwh",
        "conversion_factor", "combustion_heat",
        "meter_reading_start", "meter_reading_end",
        "commodity_unit_price", "commodity_total_price", "commodity_charge",
        "fixed_monthly_fee_unit_price", "fixed_monthly_fee",
        "distribution_unit_price", "distribution_fixed_price", "distribution_charge",
        "reserved_capacity_unit_price", "reserved_capacity_price",
        "market_operator_price", "natural_gas_tax_total",
        "amount_ex_vat", "amount_inc_vat",
        mode="before",
    )
    @classmethod
    def clean_numbers(cls, v: str | float | int | Decimal | None) -> float | None:
        return clean_czech_number(v)


class WaterData(CzechNumericModel):
    """Water supply and sewage detail (voda).

    Czech field mapping:
        - vodne -> water_rate
        - stocne -> sewage_rate
        - srazkove -> precipitation_water
        - odpadni -> wastewater_charge
    """

    period: BillingPeriod | None = None

    # Consumption (m³)
    consumption_m3: float | None = None

    # DEPRECATED: meter readings
    meter_reading_start: float | None = None  # DEPRECATED
    meter_reading_end: float | None = None    # DEPRECATED

    # Rate components (CZK)
    water_rate: float | None = None            # vodne
    sewage_rate: float | None = None           # stocne
    precipitation_water: float | None = None   # srazkove
    wastewater_charge: float | None = None     # odpadni

    # Amounts (CZK)
    amount_ex_vat: float | None = None
    amount_inc_vat: float | None = None

    @field_validator(
        "consumption_m3",
        "meter_reading_start", "meter_reading_end",
        "water_rate", "sewage_rate",
        "precipitation_water", "wastewater_charge",
        "amount_ex_vat", "amount_inc_vat",
        mode="before",
    )
    @classmethod
    def clean_numbers(cls, v: str | float | int | Decimal | None) -> float | None:
        return clean_czech_number(v)


class HeatData(CzechNumericModel):
    """District heating detail (teplo).

    Czech field mapping:
        - spotreba_tepla -> heat_consumption
        - spotreba_ohrev_tv -> hot_water_heating
        - studena_voda -> cold_water
        - rez_kapacita -> reserved_capacity
        - doplnovaci_voda -> supplementary_water
        - celkova_spotreba_tepla -> total_heat_consumption
    """

    period: BillingPeriod | None = None

    # Consumption
    consumption_gj: float | None = None            # spotreba_gj (legacy)
    heat_consumption: float | None = None          # spotreba_tepla
    hot_water_heating: float | None = None         # spotreba_ohrev_tv
    cold_water: float | None = None                # studena_voda
    total_heat_consumption: float | None = None    # celkova_spotreba_tepla

    # Capacity / water
    reserved_capacity: float | None = None         # rez_kapacita
    supplementary_water: float | None = None       # doplnovaci_voda

    # DEPRECATED: property info
    heated_area: float | None = None               # DEPRECATED: vytapena_plocha

    # Charges (CZK)
    fixed_monthly_fee: float | None = None         # staly_mesicni_plat
    variable_charge: float | None = None

    # Amounts (CZK)
    amount_ex_vat: float | None = None
    amount_inc_vat: float | None = None

    @field_validator(
        "consumption_gj", "heat_consumption",
        "hot_water_heating", "cold_water", "total_heat_consumption",
        "reserved_capacity", "supplementary_water",
        "heated_area",
        "fixed_monthly_fee", "variable_charge",
        "amount_ex_vat", "amount_inc_vat",
        mode="before",
    )
    @classmethod
    def clean_numbers(cls, v: str | float | int | Decimal | None) -> float | None:
        return clean_czech_number(v)


class GasVOData(CzechNumericModel):
    """Gas large-scale consumer detail (plyn velkoodběr).

    Czech field mapping:
        - koef_prepoctu -> conversion_factor
        - spalne_teplo -> combustion_heat
        - spotreba_mwh -> consumption_mwh
        - spotreba_m -> consumption_m3
        - mnozstvi_denni_rez_kapacity -> daily_reserved_capacity
        - dan_zemni_plyn_celkem -> natural_gas_tax_total
        - cena_ostatni_sluzby_dodavky -> other_supply_services_price
        - jednotkova_cena_za_obchod_rez_kapacity -> trade_reserved_capacity_unit_price
        - cena_za_obchod_rez_kapacity -> trade_reserved_capacity_price
        - cena_sluzby_distribuce -> distribution_service_price
        - jednotkova_cena_za_sluzby_distr_soustavy -> distribution_system_unit_price
        - jednotkova_cena_za_distribuci_rez_kapacity -> distribution_reserved_capacity_unit_price
        - cena_distribuce_rezervovane_kapacity -> distribution_reserved_capacity_price
        - cena_cinnost_operatora_trhu -> market_operator_price
    """

    period: BillingPeriod | None = None

    # Consumption
    consumption_m3: float | None = None   # spotreba_m
    consumption_mwh: float | None = None  # spotreba_mwh

    # Conversion parameters
    conversion_factor: float | None = None  # koef_prepoctu
    combustion_heat: float | None = None    # spalne_teplo (MJ/m³)

    # Capacity
    daily_reserved_capacity: float | None = None  # mnozstvi_denni_rez_kapacity

    # Commodity / trade pricing
    other_supply_services_price: float | None = None         # cena_ostatni_sluzby_dodavky
    trade_reserved_capacity_unit_price: float | None = None  # jednotkova_cena_za_obchod_rez_kapacity
    trade_reserved_capacity_price: float | None = None       # cena_za_obchod_rez_kapacity

    # Distribution
    distribution_service_price: float | None = None               # cena_sluzby_distribuce
    distribution_system_unit_price: float | None = None           # jednotkova_cena_za_sluzby_distr_soustavy
    distribution_reserved_capacity_unit_price: float | None = None  # jednotkova_cena_za_distribuci_rez_kapacity
    distribution_reserved_capacity_price: float | None = None     # cena_distribuce_rezervovane_kapacity

    # Other fees
    market_operator_price: float | None = None   # cena_cinnost_operatora_trhu
    natural_gas_tax_total: float | None = None   # dan_zemni_plyn_celkem

    # Amounts (CZK)
    amount_ex_vat: float | None = None
    amount_inc_vat: float | None = None

    @field_validator(
        "consumption_m3", "consumption_mwh",
        "conversion_factor", "combustion_heat",
        "daily_reserved_capacity",
        "other_supply_services_price",
        "trade_reserved_capacity_unit_price", "trade_reserved_capacity_price",
        "distribution_service_price", "distribution_system_unit_price",
        "distribution_reserved_capacity_unit_price", "distribution_reserved_capacity_price",
        "market_operator_price", "natural_gas_tax_total",
        "amount_ex_vat", "amount_inc_vat",
        mode="before",
    )
    @classmethod
    def clean_numbers(cls, v: str | float | int | Decimal | None) -> float | None:
        return clean_czech_number(v)

# ════════════════════════════════════════════════════════════════════════════
# MAIN INVOICE MODEL
# ════════════════════════════════════════════════════════════════════════════


class InvoiceData(CzechNumericModel):
    """Complete invoice data model (faktura / doklad).

    Czech field mapping:
        - id_doklad -> id
        - doklad_cislo -> invoice_number
        - kod_odberne_misto -> consumption_point_code (in supply_point)
        - obdobi_od/do -> period (BillingPeriod)
          NOTE: transitional invoices may have multiple periods — stored in commodity detail tables
        - ico_odberatel -> customer_tax_id
        - ico_dodavatel -> supplier_tax_id
        - opravna -> is_correction / invoice_type
        - kterou_fakturu_opravuje -> correction_info
        - datum_uzp -> vat_date
        - datum_vystaveni -> issue_date
        - datum_splatnosti -> due_date
        - prechodova_faktura -> is_transitional / invoice_type
    """

    id: UUID = Field(default_factory=uuid4)
    source_filename: str = ""

    # Invoice identification
    invoice_number: str  # doklad_cislo
    variable_symbol: str | None = None  # DEPRECATED: not required for MVP

    # Invoice type
    commodity: CommodityType
    invoice_type: InvoiceType = InvoiceType.REGULAR
    is_correction: bool = False   # opravna
    is_transitional: bool = False  # prechodova_faktura

    # Supply point
    supply_point: SupplyPoint = Field(default_factory=SupplyPoint) 

    # Billing period (main period)
    period: BillingPeriod | None = None

    # Key dates
    issue_date: date | None = None  # datum_vystaveni
    due_date: date | None = None  # datum_splatnosti
    vat_date: date | None = None  # datum_uzp

    # Party identification
    customer_tax_id: str | None = None   # ico_odberatel (8-digit zero-padded string)
    customer_vat_id: str | None = None   # dic_odberatel (CZ + 8-10 digits)
    customer_name: str | None = None
    supplier_tax_id: str | None = None   # ico_dodavatele (8-digit zero-padded string)
    supplier_vat_id: str | None = None   # dic_dodavatel (CZ + 8-10 digits)
    supplier_name: str | None = None

    # Monetary amounts (CZK)
    total_amount_ex_vat: float | None = None   # castka_bez_dph (per-period for transitional)
    total_amount_inc_vat: float | None = None  # castka_s_dph  (per-period for transitional)
    vat_amount: float | None = None            # dan_zakladni (DPH částka — useful for validation)
    vat_rate: float | None = None             # sazba_dph — 21.0, 12.0, 0.0 (%)
    advance_payment: float | None = None       # zaplacene_zalohy
    amount_to_pay: float | None = None         # k_uhrade (= total_inc_vat - advance_payment)
    # Correction info (if is_correction=True)
    correction_info: CorrectionInfo | None = None 

    # Commodity-specific details (One-to-Many for transitional invoices)
    electricity_nn_details: list[ElectricityNNData] = Field(default_factory=list)
    electricity_vn_details: list[ElectricityVNData] = Field(default_factory=list)
    gas_mo_details: list[GasMOData] = Field(default_factory=list)
    gas_vo_details: list[GasVOData] = Field(default_factory=list)
    water_details: list[WaterData] = Field(default_factory=list)
    heat_details: list[HeatData] = Field(default_factory=list)

    @field_validator("issue_date", "due_date", "vat_date", mode="before")
    @classmethod
    def parse_dates(cls, v: str | date | None) -> date | None:
        if v is None:
            return None
        result = parse_czech_date(v)
        if result is None and v:
            raise ValueError(f"Cannot parse date: {v!r}")
        return result

    @field_validator("invoice_number", mode="before")
    @classmethod
    def _clean_invoice_number(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return str(v).strip()

    @field_validator("customer_tax_id", "supplier_tax_id", mode="before")
    @classmethod
    def _clean_tax_ids(cls, v: str | None) -> str | None:
        return clean_tax_id(v)

    @field_validator("is_correction", mode="before")
    @classmethod
    def _parse_is_correction(cls, v: bool | str | None) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        s = str(v).strip().lower()
        if not s:
            return False
        if s in ("true", "1", "yes", "y"):
            return True
        # Czech keywords that indicate correction
        if "oprav" in s or "oprava" in s or "opravn" in s:
            return True
        return False

    @field_validator(
        "total_amount_ex_vat",
        "total_amount_inc_vat",
        "vat_amount",
        "vat_rate",
        "advance_payment",
        "amount_to_pay",
        mode="before",
    )
    @classmethod
    def clean_amounts(cls, v: str | float | int | Decimal | None) -> float | None:
        return clean_czech_number(v)

    @model_validator(mode="after")
    def set_invoice_type_flags(self) -> "InvoiceData":
        """Synchronize invoice_type with boolean flags."""
        if self.is_correction:
            self.invoice_type = InvoiceType.CORRECTION
        elif self.is_transitional:
            self.invoice_type = InvoiceType.TRANSITIONAL
        return self

    def is_cross_year(self) -> bool:
        """Check if the main billing period spans multiple years."""
        if self.period is None:
            return False
        return self.period.is_cross_year()


# NOTE: plynVO, ElektrinaVN, teplo are always billed monthly;
#       transitional check is not strictly necessary but harmless.

# ════════════════════════════════════════════════════════════════════════════
# EXTRACTION RESULT WRAPPER
# ════════════════════════════════════════════════════════════════════════════


class ExtractionResult(BaseModel):
    """Wrapper returned by every extraction strategy."""

    id: UUID = Field(default_factory=uuid4)
    source_file: str
    strategy_name: str
    model_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    # Per-field confidence scores (0.0–1.0); key = field name, value = confidence.
    # Populated by LLM strategies that return per-field certainty.
    field_confidences: dict[str, float] = Field(default_factory=dict)
    raw_text: str = ""
    invoice_data: InvoiceData | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    token_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


# ════════════════════════════════════════════════════════════════════════════
# LEGACY MODELS (backward compatibility)
# ════════════════════════════════════════════════════════════════════════════

# Re-export for backward compatibility
ConsumptionRecord = BillingPeriod


class InvoiceMetadata(InvoiceData):
    """Alias for backward compatibility."""

    pass


# TODO: ensure errors always include context (commodity, field name, raw value).