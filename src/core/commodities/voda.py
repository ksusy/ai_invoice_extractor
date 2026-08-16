"""Pydantic schema for water (voda) invoices."""

from __future__ import annotations

from pydantic import field_validator

from src.core.commodities.base import BaseCommoditySchema
from src.domain.entities import clean_czech_number_required


class Voda(BaseCommoditySchema):
    """Water supply and sewage (voda)."""

    unit: str = "m³"
    water_supply: float = 0.0   # vodné
    sewage: float = 0.0         # stočné

    @field_validator("water_supply", "sewage", mode="before")
    @classmethod
    def _clean(cls, v: str | float | int) -> float:
        return clean_czech_number_required(v)

    def commodity_label(self) -> str:
        return "Voda"
