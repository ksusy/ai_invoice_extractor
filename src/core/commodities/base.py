"""Base commodity schema with shared OCR-string cleaning validators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, field_validator

from src.domain.entities import clean_czech_number_required


class BaseCommoditySchema(BaseModel, ABC):
    """Abstract base for commodity-specific invoice line schemas.

    Subclasses define the exact fields for each utility type while
    inheriting the shared OCR-cleaning validators.
    """

    model_config = ConfigDict(strict=False)

    # -- Common fields present on every commodity invoice --
    total_consumption: float = 0.0
    unit: str = ""
    total_price: float = 0.0
    currency: str = "CZK"

    @field_validator("total_consumption", "total_price", mode="before")
    @classmethod
    def clean_numeric(cls, v: str | float | int) -> float:
        """Normalise OCR-produced number strings like ``'1 200,50'``."""
        return clean_czech_number_required(v)

    @abstractmethod
    def commodity_label(self) -> str:
        """Return a human-readable label for this commodity."""
        ...
