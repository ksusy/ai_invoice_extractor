"""Factory that maps ``CommodityType`` → concrete Pydantic schema class."""

from __future__ import annotations

from src.core.commodities.base import BaseCommoditySchema
from src.core.commodities.elektrina import ElektrinaNN, ElektrinaVN
from src.core.commodities.plyn import PlynMO, PlynVO
from src.core.commodities.teplo import Teplo
from src.core.commodities.voda import Voda
from src.domain.constants import CommodityType

_REGISTRY: dict[CommodityType, type[BaseCommoditySchema]] = {
    CommodityType.ELEKTRINA_NN: ElektrinaNN,
    CommodityType.ELEKTRINA_VN: ElektrinaVN,
    CommodityType.PLYN_MO: PlynMO,
    CommodityType.PLYN_VO: PlynVO,
    CommodityType.TEPLO: Teplo,
    CommodityType.VODA: Voda,
}


class CommoditySchemaFactory:
    """Factory for obtaining the correct commodity schema at runtime."""

    @staticmethod
    def get_schema_class(commodity: CommodityType) -> type[BaseCommoditySchema]:
        """Return the Pydantic model class for the given commodity.

        Raises:
            KeyError: If no schema is registered for ``commodity``.
        """
        try:
            return _REGISTRY[commodity]
        except KeyError:
            raise KeyError(
                f"No commodity schema registered for {commodity!r}. "
                f"Available: {list(_REGISTRY.keys())}"
            ) from None

    @staticmethod
    def list_commodities() -> list[CommodityType]:
        """Return all registered commodity types."""
        return list(_REGISTRY.keys())
