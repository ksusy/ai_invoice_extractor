"""Tests for the PDF classifier decision logic."""

from __future__ import annotations

import pytest

from src.core.classification.pdf_classifier import (
    MAX_IMAGE_AREA_RATIO,
    MIN_CODEPOINT_DIVERSITY,
    MIN_FONT_COUNT,
    MIN_PAGE_TEXT_RATIO,
    MIN_TOTAL_CHARS,
    PDFClassifier,
)
from src.core.classification.base import DocumentKind


class TestPDFClassifierDecision:
    """Test the _decide() method with controlled analysis dicts."""

    @pytest.fixture()
    def classifier(self) -> PDFClassifier:
        return PDFClassifier(session=None)

    def _good_analysis(self) -> dict:
        """Return an analysis dict that passes all criteria."""
        return {
            "total_chars": 500,
            "codepoint_diversity": 40,
            "font_count": 5,
            "total_pages": 2,
            "pages_with_text": 2,
            "page_text_ratio": 1.0,
            "image_area_ratio": 0.1,
        }

    def test_native_pdf_all_criteria_pass(self, classifier: PDFClassifier):
        result = classifier._decide(self._good_analysis())
        assert result == DocumentKind.NATIVE_PDF

    def test_scanned_low_char_count(self, classifier: PDFClassifier):
        analysis = self._good_analysis()
        analysis["total_chars"] = MIN_TOTAL_CHARS - 1
        assert classifier._decide(analysis) == DocumentKind.SCANNED

    def test_scanned_low_codepoint_diversity(self, classifier: PDFClassifier):
        analysis = self._good_analysis()
        analysis["codepoint_diversity"] = MIN_CODEPOINT_DIVERSITY - 1
        assert classifier._decide(analysis) == DocumentKind.SCANNED

    def test_scanned_low_font_count(self, classifier: PDFClassifier):
        analysis = self._good_analysis()
        analysis["font_count"] = MIN_FONT_COUNT - 1
        assert classifier._decide(analysis) == DocumentKind.SCANNED

    def test_scanned_low_page_text_ratio(self, classifier: PDFClassifier):
        analysis = self._good_analysis()
        analysis["page_text_ratio"] = MIN_PAGE_TEXT_RATIO - 0.01
        assert classifier._decide(analysis) == DocumentKind.SCANNED

    def test_scanned_high_image_ratio(self, classifier: PDFClassifier):
        analysis = self._good_analysis()
        analysis["image_area_ratio"] = MAX_IMAGE_AREA_RATIO + 0.01
        assert classifier._decide(analysis) == DocumentKind.SCANNED

    @pytest.mark.asyncio
    async def test_non_pdf_bytes_classified_as_scanned(self, classifier: PDFClassifier):
        result = await classifier.classify(b"not a pdf at all")
        assert result == DocumentKind.SCANNED

    @pytest.mark.asyncio
    async def test_empty_bytes_classified_as_scanned(self, classifier: PDFClassifier):
        result = await classifier.classify(b"")
        assert result == DocumentKind.SCANNED
