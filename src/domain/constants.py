"""Global constants and the PRINT_MODE toggle.

PRINT_MODE is re-exported here for convenient access across the
codebase.  Its actual value is managed by ``src.config.settings``.

Enums (CommodityType, InvoiceType) are defined in entities.py and
re-exported here for backward compatibility.
"""

from __future__ import annotations

from src.config.settings import PrintMode, get_settings

# Re-export enums from entities for backward compatibility
from src.domain.entities import CommodityType, InvoiceType

# ── Re-export PRINT_MODE as a module-level convenience ───────────
# Usage:
#   from src.domain.constants import PRINT_MODE
#   if PRINT_MODE() == PrintMode.GRAYSCALE: ...


def PRINT_MODE() -> PrintMode:  # noqa: N802 – intentional UPPER_CASE name
    """Return the current print-mode setting (COLOR | GRAYSCALE).

    This is a function so the value always reflects the latest
    configuration (useful in tests that patch settings).
    """
    return get_settings().print_mode


# ── Miscellaneous ────────────────────────────────────────────────

DATE_FORMAT = "%d.%m.%Y"
SUPPORTED_FILE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

__all__ = [
    "PRINT_MODE",
    "PrintMode",
    "CommodityType",
    "InvoiceType",
    "DATE_FORMAT",
    "SUPPORTED_FILE_EXTENSIONS",
]
