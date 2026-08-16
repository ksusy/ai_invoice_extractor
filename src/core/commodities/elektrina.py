"""Pydantic schemas for electricity invoices (NN & VN)."""

from __future__ import annotations

from pydantic import field_validator

from src.core.commodities.base import BaseCommoditySchema
from src.domain.entities import clean_czech_number_required


class ElektrinaNN(BaseCommoditySchema):
    """Electricity – low voltage (nízké napětí)."""

    unit: str = "kWh"
    distribution_tariff: str = ""
    peak_consumption: float = 0.0
    off_peak_consumption: float = 0.0
    circuit_breaker_value: float = 0.0  # jistič (A)

    @field_validator("peak_consumption", "off_peak_consumption", "circuit_breaker_value", mode="before")
    @classmethod
    def _clean(cls, v: str | float | int) -> float:
        return clean_czech_number_required(v)

    def commodity_label(self) -> str:
        return "Elektřina – nízké napětí"


class ElektrinaVN(BaseCommoditySchema):
    """Electricity – high voltage (vysoké napětí)."""

    unit: str = "MWh"
    reserved_capacity: float = 0.0
    peak_demand: float = 0.0

    @field_validator("reserved_capacity", "peak_demand", mode="before")
    @classmethod
    def _clean(cls, v: str | float | int) -> float:
        return clean_czech_number_required(v)

    def commodity_label(self) -> str:
        return "Elektřina – vysoké napětí"
