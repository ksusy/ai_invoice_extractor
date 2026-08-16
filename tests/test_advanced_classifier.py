"""Tests for the advanced PDF classifier with asymmetric error cost.

This test suite focuses on:
1. Detecting false negatives (scans misclassified as native) – MOST CRITICAL
2. Acceptable false positives (natives misclassified as scans)
3. Component score correctness
"""

from __future__ import annotations

import pytest

from src.core.classification.advanced_classifier import (
    AdvancedPDFClassifier,
)
from src.core.classification.base import DocumentKind


class TestAdvancedClassifierComponentScores:
    """Test individual component scoring logic."""

    @pytest.fixture
    def classifier(self) -> AdvancedPDFClassifier:
        return AdvancedPDFClassifier(session=None)

    def test_invisible_font_scoring_empty(self, classifier: AdvancedPDFClassifier):
        """Invisible font score should be 0 when no fonts."""
        analysis = {
            "all_fonts": {},
            "total_chars": 100,
        }
        score = classifier._score_invisible_fonts(analysis)
        assert score == 0.0

    def test_invisible_font_scoring_high_invisible(
        self, classifier: AdvancedPDFClassifier
    ):
        """High score when invisible fonts dominate."""
        analysis = {
            "all_fonts": {
                "UNKNOWN": {"count": 80, "is_invisible": True},
                "Arial": {"count": 20, "is_invisible": False},
            },
            "total_chars": 100,
        }
        score = classifier._score_invisible_fonts(analysis)
        assert score > 0.6  # Should be high (0.80 invisible ratio)

    def test_invisible_font_scoring_low_invisible(
        self, classifier: AdvancedPDFClassifier
    ):
        """Low score when visible fonts dominate."""
        analysis = {
            "all_fonts": {
                "Times New Roman": {"count": 90, "is_invisible": False},
                "Helvetica": {"count": 10, "is_invisible": False},
            },
            "total_chars": 100,
        }
        score = classifier._score_invisible_fonts(analysis)
        assert score < 0.1  # Should be low

    def test_text_entropy_calculation(self, classifier: AdvancedPDFClassifier):
        """Entropy calculation should distinguish ordered vs random text."""
        # Very repetitive text (low entropy)
        entropy_low = classifier._calculate_text_entropy("aaaaaaa bbbbbbb ccccccc")
        # Diverse text (higher entropy)
        entropy_high = classifier._calculate_text_entropy(
            "The quick brown fox jumps over the lazy dog"
        )

        assert entropy_low < entropy_high

    def test_text_corruption_scoring_clean(self, classifier: AdvancedPDFClassifier):
        """Low corruption score for clean text."""
        analysis = {
            "full_text": "The quick brown fox jumps over the lazy dog. " * 10,
            "invalid_unicode_count": 0,
            "total_chars": 100,
        }
        score = classifier._score_text_corruption(analysis)
        assert score < 0.30  # Should be very low for clean text

    def test_text_corruption_scoring_garbled(
        self, classifier: AdvancedPDFClassifier
    ):
        """High corruption score for garbled text."""
        # Simulate OCR garbage: repetitive, lots of spaces
        garbled = "a a a a a  \n\n\nb b b b b  \n\n\n" * 5
        analysis = {
            "full_text": garbled,
            "invalid_unicode_count": 5,
            "total_chars": len(garbled),
        }
        score = classifier._score_text_corruption(analysis)
        assert score > 0.30  # Should be higher for garbled text

    def test_image_dominance_scoring(self, classifier: AdvancedPDFClassifier):
        """Image dominance score correlates with image area ratio."""
        # Low image area
        analysis_low = {"image_area_ratio": 0.1}
        score_low = classifier._score_image_dominance(analysis_low)

        # High image area
        analysis_high = {"image_area_ratio": 0.85}
        score_high = classifier._score_image_dominance(analysis_high)

        assert score_low < score_high
        assert score_high > 0.7  # Should be high

    def test_text_font_density_scoring(self, classifier: AdvancedPDFClassifier):
        """Low diversity (few fonts, many chars) should score high for scan."""
        # Few fonts, many chars (typical OCR)
        analysis_ocr = {"total_chars": 1000, "font_count": 1}
        score_ocr = classifier._score_text_font_density(analysis_ocr)

        # Many fonts, chars spread (typical native)
        analysis_native = {"total_chars": 500, "font_count": 8}
        score_native = classifier._score_text_font_density(analysis_native)

        assert score_ocr > score_native


class TestAdvancedClassifierDecisionLogic:
    """Test the final decision-making with asymmetric thresholds."""

    @pytest.fixture
    def classifier(self) -> AdvancedPDFClassifier:
        return AdvancedPDFClassifier(session=None)

    def _good_native_analysis(self) -> dict:
        """Analysis dict for a good native PDF."""
        return {
            "total_chars": 2000,
            "invalid_unicode_count": 0,
            "codepoint_diversity": 50,
            "all_fonts": {
                "TimesNewRoman": {"count": 1000, "is_invisible": False},
                "Helvetica": {"count": 500, "is_invisible": False},
                "Courier": {"count": 300, "is_invisible": False},
                "Symbol": {"count": 200, "is_invisible": False},
            },
            "font_count": 4,
            "total_pages": 3,
            "pages_with_text": 3,
            "page_text_ratio": 1.0,
            "image_area_ratio": 0.1,
            "full_text": "The quick brown fox jumps over the lazy dog. " * 50,
            "total_image_area": 10000,
            "total_page_area": 100000,
        }

    def _good_scan_analysis(self) -> dict:
        """Analysis dict for a good scanned PDF with OCR."""
        return {
            "total_chars": 500,
            "invalid_unicode_count": 8,
            "codepoint_diversity": 20,
            "all_fonts": {
                "UNKNOWN": {"count": 450, "is_invisible": True},
                "Arial": {"count": 50, "is_invisible": False},
            },
            "font_count": 2,
            "total_pages": 1,
            "pages_with_text": 1,
            "page_text_ratio": 1.0,
            "image_area_ratio": 0.70,
            "full_text": "a a a a  \n\n\nb b b b  \n\n\n" * 10,
            "total_image_area": 70000,
            "total_page_area": 100000,
        }

    def test_native_pdf_decision(self, classifier: AdvancedPDFClassifier):
        """High-quality native PDF should be classified as NATIVE_PDF."""
        analysis = self._good_native_analysis()
        result = classifier._decide_with_confidence(analysis)

        assert result.document_kind == DocumentKind.NATIVE_PDF
        assert result.scanned_likelihood < 0.25
        assert result.confidence > 0.7

    def test_scanned_pdf_decision(self, classifier: AdvancedPDFClassifier):
        """High-quality scanned PDF should be classified as SCANNED."""
        analysis = self._good_scan_analysis()
        result = classifier._decide_with_confidence(analysis)

        assert result.document_kind == DocumentKind.SCANNED
        assert result.scanned_likelihood > 0.75
        assert result.confidence > 0.7

    def test_uncertain_defaults_to_scanned(self, classifier: AdvancedPDFClassifier):
        """Uncertain classification should default to SCANNED (safe fallback)."""
        # Middling scores
        analysis = {
            "total_chars": 300,
            "invalid_unicode_count": 2,
            "codepoint_diversity": 30,
            "all_fonts": {
                "Arial": {"count": 200, "is_invisible": False},
                "UNKNOWN": {"count": 100, "is_invisible": True},
            },
            "font_count": 2,
            "total_pages": 1,
            "pages_with_text": 1,
            "page_text_ratio": 1.0,
            "image_area_ratio": 0.45,
            "full_text": "Some text with some artifacts    \n\n",
            "total_image_area": 45000,
            "total_page_area": 100000,
        }
        result = classifier._decide_with_confidence(analysis)

        # With scanned_likelihood in [0.25, 0.75], should default to SCANNED
        if 0.25 <= result.scanned_likelihood <= 0.75:
            assert result.document_kind == DocumentKind.SCANNED

    def test_asymmetric_error_cost_protects_against_fn(
        self, classifier: AdvancedPDFClassifier
    ):
        """
        Test the key requirement: no false negatives (scans classified as native).

        This test should FAIL if false negative happens.
        """
        # A scanned document with some native-like properties
        # (but still clearly a scan with hidden OCR layer)
        analysis = {
            "total_chars": 600,
            "invalid_unicode_count": 5,
            "codepoint_diversity": 25,
            "all_fonts": {
                "UNKNOWN": {"count": 500, "is_invisible": True},  # OCR keyword
                "TimesNewRoman": {"count": 100, "is_invisible": False},
            },
            "font_count": 2,
            "total_pages": 1,
            "pages_with_text": 1,
            "page_text_ratio": 1.0,
            "image_area_ratio": 0.65,  # High image component
            "full_text": "OCR text with artifacts    \n\n" + "a " * 50,
            "total_image_area": 65000,
            "total_page_area": 100000,
        }
        result = classifier._decide_with_confidence(analysis)

        # CRITICAL: Must never classify this as NATIVE_PDF
        assert result.document_kind == DocumentKind.SCANNED, (
            "CRITICAL: Scanned document with OCR artifacts misclassified as NATIVE_PDF! "
            f"Scores: {result.component_scores}, likelihood: {result.scanned_likelihood}"
        )


class TestAdvancedClassifierHeuristics:
    """Test individual heuristics like invisible font detection."""

    def test_invisible_font_detection_unknown(self):
        """Font named 'UNKNOWN' should be detected as invisible."""
        assert (
            AdvancedPDFClassifier._is_invisible_font("UNKNOWN")
        )

    def test_invisible_font_detection_gid(self):
        """Font with 'GID' prefix should be detected as invisible."""
        assert (
            AdvancedPDFClassifier._is_invisible_font("GID_STD_FONT")
        )

    def test_invisible_font_detection_tesseract(self):
        """Tesseract default font."""
        assert (
            AdvancedPDFClassifier._is_invisible_font("TDSR+Arial")
        )

    def test_invisible_font_detection_empty(self):
        """Empty font name should be invisible."""
        assert (
            AdvancedPDFClassifier._is_invisible_font("")
        )

    def test_invisible_font_detection_normal(self):
        """Normal font names should not be detected as invisible."""
        assert (
            not AdvancedPDFClassifier._is_invisible_font("TimesNewRoman")
        )
        assert (
            not AdvancedPDFClassifier._is_invisible_font("Helvetica")
        )

    def test_valid_unicode_check(self):
        """Unicode validity checker."""
        assert AdvancedPDFClassifier._is_valid_unicode(ord("A"))
        assert AdvancedPDFClassifier._is_valid_unicode(ord("ß"))
        assert not AdvancedPDFClassifier._is_valid_unicode(-1)
        assert not AdvancedPDFClassifier._is_valid_unicode(0x110000)
        assert not AdvancedPDFClassifier._is_valid_unicode(0xD800)  # Surrogate


class TestAdvancedClassifierIntegration:
    """Integration tests with actual PDF bytes (if available in test fixtures)."""

    @pytest.fixture
    def classifier(self) -> AdvancedPDFClassifier:
        return AdvancedPDFClassifier(session=None)

    @pytest.mark.asyncio
    async def test_non_pdf_classified_as_scanned(
        self, classifier: AdvancedPDFClassifier
    ):
        """Non-PDF bytes should be classified as SCANNED."""
        result = await classifier.classify(b"not a pdf")
        assert result == DocumentKind.SCANNED

    @pytest.mark.asyncio
    async def test_empty_bytes_classified_as_scanned(
        self, classifier: AdvancedPDFClassifier
    ):
        """Empty bytes should be classified as SCANNED."""
        result = await classifier.classify(b"")
        assert result == DocumentKind.SCANNED

    @pytest.mark.asyncio
    async def test_exception_handling_defaults_to_scanned(
        self, classifier: AdvancedPDFClassifier
    ):
        """Exceptions during analysis should default to SCANNED (safe)."""
        # Create a malformed PDF-like byte sequence
        malformed_pdf = b"%PDF-1.4\n" + b"\x00" * 100
        result = await classifier.classify(malformed_pdf)

        # Should not raise – should default to SCANNED
        assert result == DocumentKind.SCANNED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
