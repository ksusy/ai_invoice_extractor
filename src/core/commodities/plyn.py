"""Pydantic schemas for gas invoices (MO & VO)."""

from __future__ import annotations

from pydantic import field_validator

from src.core.commodities.base import BaseCommoditySchema
from src.domain.entities import clean_czech_number_required


class PlynMO(BaseCommoditySchema):
    """Gas – small-scale consumer (maloodběr)."""

    unit: str = "m³"
    energy_equivalent: float = 0.0  # MWh converted
    calorific_value: float = 0.0     # spalné teplo (MJ/m³)

    @field_validator("energy_equivalent", "calorific_value", mode="before")
    @classmethod
    def _clean(cls, v: str | float | int) -> float:
        return clean_czech_number_required(v)

    def commodity_label(self) -> str:
        return "Plyn – maloodběr"


class PlynVO(BaseCommoditySchema):
    """Gas – large-scale consumer (velkoodběr)."""

    unit: str = "MWh"
    contracted_capacity: float = 0.0
    daily_max: float = 0.0

    @field_validator("contracted_capacity", "daily_max", mode="before")
    @classmethod
    def _clean(cls, v: str | float | int) -> float:
        return clean_czech_number_required(v)

    def commodity_label(self) -> str:
        return "Plyn – velkoodběr"
