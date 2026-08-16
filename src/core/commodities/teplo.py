"""Pydantic schema for heat (teplo) invoices."""

from __future__ import annotations

from pydantic import field_validator

from src.core.commodities.base import BaseCommoditySchema
from src.domain.entities import clean_czech_number_required


class Teplo(BaseCommoditySchema):
    """District heating (teplo)."""

    unit: str = "GJ"
    heating_area: float = 0.0       # vytápěná plocha (m²)
    fixed_monthly_fee: float = 0.0  # stálý měsíční poplatek

    @field_validator("heating_area", "fixed_monthly_fee", mode="before")
    @classmethod
    def _clean(cls, v: str | float | int) -> float:
        return clean_czech_number_required(v)

    def commodity_label(self) -> str:
        return "Teplo"
