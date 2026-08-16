"""Per-page analysis and HYBRID document detection.

This module detects documents that contain both scanned and native pages.

Examples of HYBRID documents:
- Multi-supplier invoice with first page scanned, rest native
- Document that was partially OCR'd and re-saved
- Appendices scanned and appended to native invoice
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTChar, LTAnno, LTImage, LTFigure, LTPage

from src.core.classification.base import DocumentKind

logger = logging.getLogger(__name__)


@dataclass
class PageAnalysis:
    """Analysis result for a single PDF page."""

    page_number: int
    estimated_kind: DocumentKind
    confidence: float  # [0, 1]
    
    # Metrics
    char_count: int = 0
    font_count: int = 0
    image_area_ratio: float = 0.0
    invisible_font_ratio: float = 0.0
    
    reasoning: list[str] = field(default_factory=list)


@dataclass
class HybridAnalysisResult:
    """Result of HYBRID document analysis."""

    is_hybrid: bool  # True if document has mixed pages
    page_analyses: list[PageAnalysis]
    
    # Summary
    scanned_page_count: int = 0
    native_page_count: int = 0
    uncertain_page_count: int = 0
    scanned_page_ratio: float = 0.0  # [0, 1]
    
    recommendation: str = ""


class PerPageAnalyzer:
    """Analyze individual pages to detect HYBRID documents."""

    @staticmethod
    def analyze_page(page_num: int, layout: LTPage) -> PageAnalysis:
        """Analyze a single page for its type (scanned vs native).

        Returns: PageAnalysis with detailed metrics.
        """
        page_width = layout.width or 595
        page_height = layout.height or 842
        page_area = page_width * page_height

        char_count = 0
        font_dict: dict[str, dict] = {}
        image_area = 0.0

        for element in PerPageAnalyzer._iter_layout(layout):
            if isinstance(element, (LTChar, LTAnno)):
                char_count += 1
                if isinstance(element, LTChar):
                    fontname = getattr(element, "fontname", None) or "UNKNOWN"
                    if fontname not in font_dict:
                        font_dict[fontname] = {
                            "count": 0,
                            "is_invisible": PerPageAnalyzer._is_invisible_font(fontname),
                        }
                    font_dict[fontname]["count"] += 1

            elif isinstance(element, (LTImage, LTFigure)):
                w = abs(element.x1 - element.x0)
                h = abs(element.y1 - element.y0)
                image_area += w * h

        image_area_ratio = image_area / page_area if page_area > 0 else 0.0

        # Calculate invisible font ratio
        invisible_count = sum(
            f["count"] for f in font_dict.values() if f["is_invisible"]
        )
        invisible_font_ratio = invisible_count / max(char_count, 1)

        # Decide page type
        kind, confidence = PerPageAnalyzer._decide_page_type(
            char_count=char_count,
            font_count=len(font_dict),
            image_area_ratio=image_area_ratio,
            invisible_font_ratio=invisible_font_ratio,
        )

        reasoning = []
        if char_count < 50:
            reasoning.append(f"Low char count ({char_count})")
        if invisible_font_ratio > 0.5:
            reasoning.append(f"High invisible font ratio ({invisible_font_ratio:.1%})")
        if image_area_ratio > 0.6:
            reasoning.append(f"High image ratio ({image_area_ratio:.1%})")

        return PageAnalysis(
            page_number=page_num,
            estimated_kind=kind,
            confidence=confidence,
            char_count=char_count,
            font_count=len(font_dict),
            image_area_ratio=image_area_ratio,
            invisible_font_ratio=invisible_font_ratio,
            reasoning=reasoning,
        )

    @staticmethod
    def _decide_page_type(
        char_count: int,
        font_count: int,
        image_area_ratio: float,
        invisible_font_ratio: float,
    ) -> tuple[DocumentKind, float]:
        """Decide if page is scanned or native.

        Returns: (DocumentKind, confidence)
        """
        # Heuristics
        score_scanned = 0.0

        # Invisible fonts → likely scanned
        if invisible_font_ratio > 0.4:
            score_scanned += 0.5
        elif invisible_font_ratio > 0.2:
            score_scanned += 0.2

        # Image-heavy → likely scanned
        if image_area_ratio > 0.7:
            score_scanned += 0.4
        elif image_area_ratio > 0.5:
            score_scanned += 0.2

        # Low font diversity → likely scanned
        if font_count <= 1 and char_count > 50:
            score_scanned += 0.3

        # Low char count → ambiguous
        if char_count < 30:
            score_scanned += 0.1

        # Decide
        if score_scanned > 0.6:
            return DocumentKind.SCANNED, min(1.0, score_scanned)
        elif score_scanned < 0.2:
            return DocumentKind.NATIVE_PDF, 1.0 - score_scanned
        else:
            return DocumentKind.SCANNED, 0.5  # Uncertain → safe default

    @staticmethod
    def _is_invisible_font(fontname: str) -> bool:
        """Check if font is likely invisible (OCR marker)."""
        if not fontname:
            return True
        fontname_lower = fontname.lower()
        return any(
            marker in fontname_lower
            for marker in ["unknown", "gid", "tdsr", "t1_0"]
        )

    @staticmethod
    def _iter_layout(layout_obj):
        """Recursively iterate over layout elements."""
        if hasattr(layout_obj, "__iter__"):
            for child in layout_obj:
                yield child
                yield from PerPageAnalyzer._iter_layout(child)


class HybridDocumentDetector:
    """Detect and analyze HYBRID documents (mixed native/scanned pages)."""

    @staticmethod
    def analyze(file_bytes: bytes) -> HybridAnalysisResult:
        """Analyze PDF for hybrid nature.

        Args:
            file_bytes: PDF bytes

        Returns:
            HybridAnalysisResult with per-page breakdown.
        """
        stream = io.BytesIO(file_bytes)
        parser = PDFParser(stream)
        document = PDFDocument(parser)

        rsrcmgr = PDFResourceManager()
        laparams = LAParams(line_margin=0.5, word_margin=0.1, char_margin=2.0)
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        page_analyses = []
        page_num = 0

        stream.seek(0)
        for page in PDFPage.create_pages(document):
            page_num += 1

            try:
                interpreter.process_page(page)
                layout: LTPage = device.get_result()
                analysis = PerPageAnalyzer.analyze_page(page_num, layout)
                page_analyses.append(analysis)
            except Exception as e:
                logger.debug("Error analyzing page %d: %s", page_num, e)
                # Skip page
                continue

        # Summarize
        scanned_count = sum(
            1 for pa in page_analyses
            if pa.estimated_kind == DocumentKind.SCANNED
        )
        native_count = sum(
            1 for pa in page_analyses
            if pa.estimated_kind == DocumentKind.NATIVE_PDF
        )
        uncertain_count = len(page_analyses) - scanned_count - native_count

        total_pages = max(len(page_analyses), 1)
        scanned_ratio = scanned_count / total_pages

        # Detect hybrid
        is_hybrid = (scanned_count > 0 and native_count > 0)

        recommendation = ""
        if is_hybrid:
            recommendation = (
                f"HYBRID document: {scanned_count} scanned + {native_count} native pages. "
                f"Consider preprocessing to separate pages."
            )
        elif scanned_count == total_pages:
            recommendation = "Fully scanned document – recommend OCR."
        elif native_count == total_pages:
            recommendation = "Fully native document – extract text directly."

        return HybridAnalysisResult(
            is_hybrid=is_hybrid,
            page_analyses=page_analyses,
            scanned_page_count=scanned_count,
            native_page_count=native_count,
            uncertain_page_count=uncertain_count,
            scanned_page_ratio=scanned_ratio,
            recommendation=recommendation,
        )

    @staticmethod
    def overall_classification(result: HybridAnalysisResult) -> DocumentKind:
        """
        Determine overall document classification from per-page analysis.

        Strategy:
        - If any page is scanned with high confidence → Document is SCANNED
        - Only if all pages are clearly native → NATIVE_PDF
        """
        # Check for any strong scanned indicators
        for analysis in result.page_analyses:
            if (analysis.estimated_kind == DocumentKind.SCANNED and
                analysis.confidence > 0.7):
                return DocumentKind.SCANNED

        # Check if all are clearly native
        all_native = all(
            analysis.estimated_kind == DocumentKind.NATIVE_PDF
            for analysis in result.page_analyses
        )
        if all_native:
            return DocumentKind.NATIVE_PDF

        # Otherwise (mixed), default to SCANNED (safe)
        return DocumentKind.SCANNED
