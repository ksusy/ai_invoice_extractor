"""OCR engine module – strategy pattern for multiple OCR backends.

Modul OCR enginů – vzor Strategie pro různé OCR backendy.
"""

from src.core.ocr_engine.base import BaseOCREngine, OCRResult
from src.core.ocr_engine.ocr_processor import (
    UI_STYLE,
    AnalysisResult,
    DetectedBlock,
    DocumentAnalyzer,
    PreprocessConfig,
)
from src.core.ocr_engine.tesseract_engine import TesseractEngine, create_tesseract_engine

__all__ = [
    "BaseOCREngine",
    "OCRResult",
    "TesseractEngine",
    "create_tesseract_engine",
    "DocumentAnalyzer",
    "AnalysisResult",
    "DetectedBlock",
    "PreprocessConfig",
    "UI_STYLE",
]
