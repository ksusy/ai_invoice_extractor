"""Layout Information Extraction pipeline.

Implements the two-stage document understanding approach:

  Stage 1 — Layout Analysis:
      Image → Skew correction → Region detection (text / table / header)
              → Table structure recognition → OCR per region

  Stage 2 — Layout Recovery:
      Detected regions → Reading-order reconstruction → LLM-ready payload

The pipeline is intentionally self-contained (no PaddleOCR / PPStructure
dependency) and works with EasyOCR + OpenCV only, both of which are
available in the current project environment.

Public surface
──────────────
    LayoutPipeline       — main entry point
    LayoutResult         — structured output (mirrors AnalysisResult API)
    DetectedRegion       — single layout region (text block / table / header)
    SkewCorrector        — standalone skew-correction utility
    TableExtractor       — standalone table-structure utility
"""

from .pipeline import LayoutPipeline, LayoutResult, DetectedRegion
from .skew_corrector import SkewCorrector
from .table_extractor import TableExtractor

__all__ = [
    "LayoutPipeline",
    "LayoutResult",
    "DetectedRegion",
    "SkewCorrector",
    "TableExtractor",
]
