"""Analysis module – correction and transitional invoice logic.

Modul analýzy – logika opravných a přechodových faktur.
"""

from src.core.analysis.base import AnalysisReport, BaseAnalyser
from src.core.analysis.correction import (
    CorrectionAnalyser,
    CorrectionAnalysisResult,
    CorrectionDelta,
    create_correction_analyser,
)
from src.core.analysis.transitional import (
    TransitionalAnalyser,
    TransitionalSplitResult,
    YearSplit,
    create_transitional_analyser,
)

__all__ = [
    # Base
    "AnalysisReport",
    "BaseAnalyser",
    # Transitional
    "TransitionalAnalyser",
    "TransitionalSplitResult",
    "YearSplit",
    "create_transitional_analyser",
    # Correction
    "CorrectionAnalyser",
    "CorrectionAnalysisResult",
    "CorrectionDelta",
    "create_correction_analyser",
]
