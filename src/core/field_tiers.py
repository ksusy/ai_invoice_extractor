"""Mandatory vs. optional field definitions for utility invoice extraction.

Tier 1 (mandatory): fields required for every invoice, measurable against all DS2 invoices.
Tier 2 (optional/extended): commodity-specific fields, reported separately.

Field names match InvoiceData / commodity Detail model attribute names.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tier 1 — mandatory for all commodities
# ---------------------------------------------------------------------------
MANDATORY_BASE = [
    "invoice_number",      # cislo_dokladu
    "period_from",         # obdobi_od  (from InvoiceData.period.period_from)
    "period_to",           # obdobi_do
    "total_amount_ex_vat", # castka_bez_dph
    "total_amount_inc_vat",# castka_s_dph
]

# ---------------------------------------------------------------------------
# Tier 1 — mandatory commodity-specific fields
# ---------------------------------------------------------------------------
MANDATORY_COMMODITY: dict[str, list[str]] = {
    "elektrina_nn": [
        "consumption_low_tariff",   # spotreba_nt
        "consumption_high_tariff",  # spotreba_vt
    ],
    "elektrina_vn": [
        "supply_consumption",       # spotreba_se (MWh)
    ],
    "plyn_mo": [
        "consumption_m3",           # spotreba_m3
        "combustion_heat",          # spalne_teplo (MJ/m³)
        "consumption_mwh",          # spotreba_mwh
    ],
    "plyn_vo": [
        "consumption_m3",           # spotreba_m3
        "combustion_heat",          # spalne_teplo
        "consumption_mwh",          # spotreba_mwh
    ],
    "voda": [
        "water_rate",               # vodne
        "sewage_rate",              # stocne
    ],
    "teplo": [
        "heat_consumption",         # spotreba_tepla (GJ)
        "variable_charge",          # castka_se (variabilní složka)
    ],
}

# plyn_MO/plyn_VO aliases (entity CommodityType uses uppercase MO/VO)
MANDATORY_COMMODITY["plyn_MO"] = MANDATORY_COMMODITY["plyn_mo"]
MANDATORY_COMMODITY["plyn_VO"] = MANDATORY_COMMODITY["plyn_vo"]


def get_mandatory_fields(commodity: str) -> list[str]:
    """Return all mandatory field names (base + commodity-specific) for a commodity."""
    return MANDATORY_BASE + MANDATORY_COMMODITY.get(commodity, [])


# ---------------------------------------------------------------------------
# Tier 2 — optional (extended) fields per commodity
# ---------------------------------------------------------------------------
OPTIONAL_COMMODITY: dict[str, list[str]] = {
    "elektrina_nn": [
        "total_consumption",
        "supply_charge",
        "distribution_charge",
        "system_services",
        "renewable_energy_fee",
    ],
    "elektrina_vn": [
        "supply_charge",
        "supply_tax_charge",
        "quarter_hour_max",
        "eru_rate",
        "annual_reserved_capacity",
        "annual_reserved_capacity_charge",
        "monthly_reserved_capacity",
        "monthly_reserved_capacity_charge",
        "grid_usage_rate",
        "grid_usage_charge",
        "reserved_capacity_excess",
        "reserved_capacity_excess_rate",
        "reserved_capacity_excess_charge",
        "power_factor",
        "reactive_power_quantity",
        "reactive_power_rate",
        "reactive_power_charge",
        "service_price",
        "operating_price",
        "renewable_energy_fee",
    ],
    "plyn_mo": [
        "period_months",
        "conversion_factor",
        "commodity_unit_price",
        "commodity_total_price",
        "fixed_monthly_fee_unit_price",
        "fixed_monthly_fee",
        "distribution_unit_price",
        "distribution_fixed_price",
        "reserved_capacity_unit_price",
        "reserved_capacity_price",
        "market_operator_price",
        "natural_gas_tax_total",
    ],
    "plyn_vo": [
        "conversion_factor",
        "daily_reserved_capacity",
        "other_supply_services_price",
        "trade_reserved_capacity_unit_price",
        "trade_reserved_capacity_price",
        "distribution_service_price",
        "distribution_system_unit_price",
        "distribution_reserved_capacity_unit_price",
        "distribution_reserved_capacity_price",
        "market_operator_price",
        "natural_gas_tax_total",
    ],
    "voda": [
        "consumption_m3",
        "precipitation_water",
        "wastewater_charge",
    ],
    "teplo": [
        "consumption_gj",
        "hot_water_heating",
        "cold_water",
        "total_heat_consumption",
        "reserved_capacity",
        "supplementary_water",
        "fixed_monthly_fee",
    ],
}

OPTIONAL_COMMODITY["plyn_MO"] = OPTIONAL_COMMODITY["plyn_mo"]
OPTIONAL_COMMODITY["plyn_VO"] = OPTIONAL_COMMODITY["plyn_vo"]


def get_optional_fields(commodity: str) -> list[str]:
    """Return optional (Tier 2) field names for a commodity."""
    return OPTIONAL_COMMODITY.get(commodity, [])


# ---------------------------------------------------------------------------
# GT field alias map — maps GT CSV column names → entity field names
# (for notebooks that compare extracted fields against ground truth)
# ---------------------------------------------------------------------------
GT_FIELD_MAP: dict[str, str] = {
    "cislo_dokladu": "invoice_number",
    "obdobi_od": "period_from",
    "obdobi_do": "period_to",
    "castka_bez_dph": "total_amount_ex_vat",
    "castka_s_dph": "total_amount_inc_vat",
    # elektrina_nn
    "spotreba_nt": "consumption_low_tariff",
    "spotreba_vt": "consumption_high_tariff",
    # elektrina_vn
    "spotreba_se": "supply_consumption",
    # plyn
    "spotreba_m3": "consumption_m3",
    "spalne_teplo": "combustion_heat",
    "spotreba_mwh": "consumption_mwh",
    # voda
    "vodne": "water_rate",
    "stocne": "sewage_rate",
    # teplo
    "spotreba_tepla": "heat_consumption",
    "spotreba_gj": "consumption_gj",
}
