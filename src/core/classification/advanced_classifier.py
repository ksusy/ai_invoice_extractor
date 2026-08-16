"""Advanced PDF classifier with asymmetric error cost – never misclassify scans as native.

Uses multi-factor heuristics to detect OCR text layers, garbled unicode,
and invisible fonts that could indicate a scanned document incorrectly
classified as native PDF.

Asymmetric loss: False Negatives (scan->native) cost ~100x more than
False Positives (native->scan).

Classification policy:
    - This classifier scores a document on a [0, 1] scan-likelihood scale.
    - A score > 0.75 means "very likely a scanned PDF"
    - A score < 0.25 means "very likely a native PDF"
    - 0.25 to 0.75 range = uncertain (HYBRID / needs review)

    When in doubt, CLASSIFY AS SCANNED to avoid silent data loss.
"""

from __future__ import annotations

import io
import logging
import math
import re
import uuid
from pathlib import Path
from typing import NamedTuple

from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTFigure,
    LTImage,
    LTPage,
    LTTextBox,
    LTTextLine,
)
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.classification.base import BaseClassifier, DocumentKind
from src.infrastructure.db.database import get_session_context
from src.infrastructure.db.models import Transaction

logger = logging.getLogger(__name__)


class ComponentScores(NamedTuple):
    """Individual scores for each detection component."""

    invisible_font_score: float
    text_corruption_score: float
    image_dominance_score: float
    text_font_density_score: float


class AdvancedClassificationResult(NamedTuple):
    """Extended classification result with confidence and reasoning."""

    document_kind: DocumentKind
    confidence: float  # [0, 1] – how certain we are
    scanned_likelihood: float  # [0, 1] – probability it's a scan
    component_scores: ComponentScores
    reasoning: list[str]


class AdvancedPDFClassifier(BaseClassifier):
    """Advanced classifier with multi-factor OCR artifact detection.

    Classification strategy (asymmetric – prefer false-scan):
        1. Analyze PDF structure: fonts, text, images across up to 20 pages.
        2. Calculate component scores:
           a. Invisible font prevalence (indicator of OCR text layer)
           b. Text corruption / garbled unicode (OCR artifacts)
           c. Image dominance (scan = mostly image + text overlay)
           d. Text-font density (OCR = few fonts with many chars)
        3. Combine into weighted scanned_likelihood score [0, 1].
        4. If scanned_likelihood > 0.75 → SCANNED (safe)
           If scanned_likelihood < 0.25 → NATIVE_PDF (high confidence)
           Else → HYBRID (uncertain, log for review).
    """

    def __init__(self, session: AsyncSession | None = None) -> None:
        self._session = session

    async def classify(self, file_bytes: bytes) -> DocumentKind:
        """Classify a PDF (returns only DocumentKind for minimal API change).

        Returns DocumentKind.SCANNED by default (safe fallback).
        """
        if not file_bytes.startswith(b"%PDF"):
            return DocumentKind.SCANNED

        try:
            result = self.classify_with_confidence(file_bytes)
            return result.document_kind
        except Exception:
            logger.debug("Advanced pdfminer analysis failed – defaulting to SCANNED", exc_info=True)
            return DocumentKind.SCANNED

    def classify_with_confidence(self, file_bytes: bytes) -> AdvancedClassificationResult:
        """Classify with full reasoning and confidence scores.

        For diagnostics and parameter tuning. Called internally by classify().
        """
        if not file_bytes.startswith(b"%PDF"):
            return AdvancedClassificationResult(
                document_kind=DocumentKind.SCANNED,
                confidence=0.99,
                scanned_likelihood=0.99,
                component_scores=ComponentScores(0, 0, 0, 0),
                reasoning=["Input is not a PDF"],
            )

        try:
            analysis = self._deep_analyze(file_bytes)
            return self._decide_with_confidence(analysis)
        except Exception:
            logger.debug("Analysis failed – defaulting to SCANNED", exc_info=True)
            return AdvancedClassificationResult(
                document_kind=DocumentKind.SCANNED,
                confidence=0.95,
                scanned_likelihood=0.95,
                component_scores=ComponentScores(0, 0, 0, 0),
                reasoning=["PDF analysis failed – safe fallback to SCANNED"],
            )

    # ── Deep Structural Analysis ─────────────────────────────────

    def _deep_analyze(self, file_bytes: bytes) -> dict:
        """Perform comprehensive analysis of PDF structure and content.

        Returns: dict with metrics for all pages up to 20.
        """
        stream = io.BytesIO(file_bytes)
        parser = PDFParser(stream)
        document = PDFDocument(parser)

        rsrcmgr = PDFResourceManager()
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
        )
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        total_chars = 0
        all_codepoints: set[int] = set()
        all_fonts: dict[str, dict] = {}
        total_pages = 0
        pages_with_text = 0
        total_image_area = 0.0
        total_page_area = 0.0
        invalid_unicode_count = 0

        stream.seek(0)
        for page in PDFPage.create_pages(document):
            total_pages += 1
            if total_pages > 20:
                break

            try:
                interpreter.process_page(page)
                layout: LTPage = device.get_result()
            except Exception:
                continue

            page_width = layout.width if layout.width else 595
            page_height = layout.height if layout.height else 842
            page_area = page_width * page_height
            total_page_area += page_area

            page_chars = 0
            page_image_area = 0.0

            for element in self._iter_layout(layout):
                if isinstance(element, LTChar):
                    page_chars += 1
                    total_chars += 1
                    fontname = getattr(element, "fontname", None) or "UNKNOWN"

                    # Track font usage
                    if fontname not in all_fonts:
                        all_fonts[fontname] = {
                            "count": 0,
                            "is_invisible": self._is_invisible_font(fontname),
                        }
                    all_fonts[fontname]["count"] += 1

                    # Track codepoints
                    try:
                        char_code = ord(element.get_text())
                        if self._is_valid_unicode(char_code):
                            all_codepoints.add(char_code)
                        else:
                            invalid_unicode_count += 1
                    except (ValueError, TypeError):
                        invalid_unicode_count += 1

                elif isinstance(element, LTAnno):
                    page_chars += 1
                    total_chars += 1
                    txt = element.get_text()
                    if txt:
                        try:
                            ord_val = ord(txt[0])
                            if self._is_valid_unicode(ord_val):
                                all_codepoints.add(ord_val)
                            else:
                                invalid_unicode_count += 1
                        except (ValueError, TypeError):
                            invalid_unicode_count += 1

                elif isinstance(element, (LTImage, LTFigure)):
                    w = abs(element.x1 - element.x0)
                    h = abs(element.y1 - element.y0)
                    page_image_area += w * h

            total_image_area += page_image_area
            if page_chars >= 50:  # Threshold for "has text"
                pages_with_text += 1

        image_area_ratio = (
            total_image_area / total_page_area if total_page_area > 0 else 0.0
        )
        page_text_ratio = (
            pages_with_text / total_pages if total_pages > 0 else 0.0
        )

        # Extract full text for entropy/corruption analysis
        full_text = self._extract_full_text(file_bytes)

        return {
            "total_chars": total_chars,
            "invalid_unicode_count": invalid_unicode_count,
            "codepoint_diversity": len(all_codepoints),
            "all_fonts": all_fonts,
            "font_count": len(all_fonts),
            "total_pages": total_pages,
            "pages_with_text": pages_with_text,
            "page_text_ratio": page_text_ratio,
            "image_area_ratio": image_area_ratio,
            "full_text": full_text,
            "total_image_area": total_image_area,
            "total_page_area": total_page_area,
        }

    def _extract_full_text(self, file_bytes: bytes) -> str:
        """Extract full text from PDF for entropy/corruption analysis."""
        try:
            stream = io.BytesIO(file_bytes)
            text = extract_text(stream)
            return text[:10000] if text else ""  # Cap at 10K chars for performance
        except Exception:
            return ""

    @staticmethod
    def _is_invisible_font(fontname: str) -> bool:
        """Detect fonts commonly used by OCR engines as 'invisible' text layers."""
        if not fontname:
            return True

        fontname_lower = fontname.lower()

        # Markers of invisible fonts
        invisible_markers = [
            "unknown",
            "gid",  # Glyph ID fonts
            "arial",  # OCR often uses Arial as default
            "tdsr",  # Tesseract default
            "t1_0",  # Type 1 encoding
            "\x00",  # Null font
        ]

        for marker in invisible_markers:
            if marker in fontname_lower:
                return True

        # Single-char or very short names are suspicious
        if len(fontname) < 3:
            return True

        # Fonts with embedded numbers often indicate OCR engines
        if bool(re.search(r"[0-9]", fontname)) and len(fontname) < 10:
            return True

        return False

    @staticmethod
    def _is_valid_unicode(code_point: int) -> bool:
        """Check if a unicode codepoint is valid."""
        if code_point < 0 or code_point > 0x10FFFF:
            return False
        if 0xD800 <= code_point <= 0xDFFF:  # Surrogate pair (invalid)
            return False
        return True

    @staticmethod
    def _iter_layout(layout_obj):
        """Recursively iterate over all layout elements."""
        if hasattr(layout_obj, "__iter__"):
            for child in layout_obj:
                yield child
                yield from AdvancedPDFClassifier._iter_layout(child)

    # ── Component Scoring ────────────────────────────────────────

    def _score_invisible_fonts(self, analysis: dict) -> float:
        """Score likelihood of invisible OCR fonts.

        Return: [0, 1] where 1 = definitely has invisible fonts (scan indicator).
        """
        all_fonts = analysis["all_fonts"]
        if not all_fonts:
            return 0.0

        invisible_count = sum(
            font_info["count"]
            for font_info in all_fonts.values()
            if font_info["is_invisible"]
        )
        total_chars = max(analysis["total_chars"], 1)

        invisible_char_ratio = invisible_count / total_chars

        # Sigmoid-like curve: small ratio = low score, high ratio = high score
        if invisible_char_ratio < 0.1:
            return 0.05
        elif invisible_char_ratio < 0.3:
            return 0.30
        elif invisible_char_ratio < 0.6:
            return 0.60
        else:
            return 0.90

    def _score_text_corruption(self, analysis: dict) -> float:
        """Score likelihood of OCR artifacts / garbled text.

        Analyzes:
        - Invalid unicode sequences / surrogates
        - Text entropy (OCR garble has different entropy profile)
        - Abnormal spacing patterns

        Return: [0, 1] where 1 = definitely corrupted (scan indicator).
        """
        full_text = analysis["full_text"]
        if not full_text:
            return 0.0

        # Sub-score 1: Invalid unicode ratio
        invalid_count = analysis["invalid_unicode_count"]
        text_len = max(len(full_text), 1)
        invalid_ratio = invalid_count / text_len  # [0, 1]

        # Sub-score 2: Text entropy
        entropy = self._calculate_text_entropy(full_text)
        # Natural English/Czech: entropy ~4.0-4.8
        # OCR garbage: entropy ~3.0-3.5 (too repetitive)
        entropy_anomaly = max(0, (4.5 - entropy) / 2)  # [0, 1]

        # Sub-score 3: Spacing anomalies
        double_space_count = len(re.findall(r"  +", full_text))
        unusual_line_breaks = len(re.findall(r"\n\n\n+", full_text))
        spacing_anomaly_count = double_space_count + unusual_line_breaks
        spacing_score = min(1.0, spacing_anomaly_count / max(len(full_text) / 100, 1))

        # Weighted combination
        corruption_score = (
            invalid_ratio * 0.4 +  # Highest weight to invalid unicode
            entropy_anomaly * 0.35 +  # Entropy matters
            spacing_score * 0.25  # Spacing is supplementary
        )

        return min(1.0, corruption_score)

    @staticmethod
    def _calculate_text_entropy(text: str) -> float:
        """Calculate Shannon entropy of text (bits per character)."""
        if not text:
            return 0.0

        char_freqs = {}
        for char in text:
            char_freqs[char] = char_freqs.get(char, 0) + 1

        entropy = 0.0
        text_len = len(text)
        for freq in char_freqs.values():
            p = freq / text_len
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _score_image_dominance(self, analysis: dict) -> float:
        """Score likelihood that the document is image-heavy (scan indicator).

        Scanned PDFs render mostly as a large image with text overlay.
        Native PDFs have text directly.

        Return: [0, 1] where 1 = dominated by images (scan indicator).
        """
        image_ratio = analysis["image_area_ratio"]

        # Graduated scale
        if image_ratio < 0.20:
            return 0.0
        elif image_ratio < 0.40:
            return 0.1
        elif image_ratio < 0.60:
            return 0.3
        elif image_ratio < 0.80:
            return 0.7
        else:
            return 0.95

    def _score_text_font_density(self, analysis: dict) -> float:
        """Score text-to-font ratio.

        Native PDFs: multiple fonts per page (high diversity).
        Scanned PDFs: 1-2 fonts carry all text (low diversity).

        Return: [0, 1] where 1 = low diversity (scan indicator).
        """
        total_chars = analysis["total_chars"]
        font_count = max(analysis["font_count"], 1)

        chars_per_font = total_chars / font_count

        # Graduated scale
        if chars_per_font < 50:
            return 0.0  # Many fonts per char → native
        elif chars_per_font < 100:
            return 0.1
        elif chars_per_font < 200:
            return 0.3
        elif chars_per_font < 500:
            return 0.6
        else:
            return 0.9  # Very few fonts, many chars → scanned

    # ── Decision Logic ───────────────────────────────────────────

    def _decide_with_confidence(
        self, analysis: dict
    ) -> AdvancedClassificationResult:
        """Make final decision based on component scores.

        Combines component scores into weighted scanned_likelihood.
        Uses asymmetric thresholds to minimize false negatives (scan→native).
        """
        # Calculate component scores
        invisible_font_score = self._score_invisible_fonts(analysis)
        text_corruption_score = self._score_text_corruption(analysis)
        image_dominance_score = self._score_image_dominance(analysis)
        text_font_density_score = self._score_text_font_density(analysis)

        component_scores = ComponentScores(
            invisible_font_score=invisible_font_score,
            text_corruption_score=text_corruption_score,
            image_dominance_score=image_dominance_score,
            text_font_density_score=text_font_density_score,
        )

        # Weighted combination (weights sum to 1.0)
        scanned_likelihood = (
            invisible_font_score * 0.35 +  # Highest – invisible fonts are strong indicator
            text_corruption_score * 0.30 +  # Strong – garbled text is OCR artifact
            image_dominance_score * 0.20 +  # Medium – images could be legitimate content
            text_font_density_score * 0.15  # Medium – could vary in native PDFs
        )

        # Decision thresholds (asymmetric)
        reasoning = self._build_reasoning(component_scores, analysis, scanned_likelihood)

        # Jakmile skóre překročí rozhodovací práh, je klasifikace jednoznačná —
        # jistota proto začíná na 0,7 (hranici, pod kterou navazující kód považuje
        # výsledek za nejistý) a k 1,0 roste s tím, jak daleko za prahem skóre leží.
        # Původní lineární škála od prahu vracela jistotu blízkou nule právě
        # u dokumentů těsně za prahem, tedy přesně tam, kde bylo rozhodnutí učiněno.
        DECISIVE_CONFIDENCE = 0.7
        confidence_slope = (1.0 - DECISIVE_CONFIDENCE) / 0.25

        if scanned_likelihood > 0.75:
            # Jednoznačný sken
            confidence = min(
                1.0, DECISIVE_CONFIDENCE + (scanned_likelihood - 0.75) * confidence_slope
            )
            return AdvancedClassificationResult(
                document_kind=DocumentKind.SCANNED,
                confidence=confidence,
                scanned_likelihood=scanned_likelihood,
                component_scores=component_scores,
                reasoning=reasoning,
            )
        elif scanned_likelihood < 0.25:
            # Jednoznačné nativní PDF
            confidence = min(
                1.0, DECISIVE_CONFIDENCE + (0.25 - scanned_likelihood) * confidence_slope
            )
            return AdvancedClassificationResult(
                document_kind=DocumentKind.NATIVE_PDF,
                confidence=confidence,
                scanned_likelihood=scanned_likelihood,
                component_scores=component_scores,
                reasoning=reasoning,
            )
        else:
            # Uncertain – default to SCANNED to avoid data loss
            confidence = 0.4 + (abs(scanned_likelihood - 0.5) * 0.2)  # Lower confidence
            logger.warning(
                "Uncertain classification (%.2f) – defaulting to SCANNED. Reasoning: %s",
                scanned_likelihood,
                "; ".join(reasoning),
            )
            return AdvancedClassificationResult(
                document_kind=DocumentKind.SCANNED,  # SAFE FALLBACK
                confidence=confidence,
                scanned_likelihood=scanned_likelihood,
                component_scores=component_scores,
                reasoning=reasoning + ["Uncertain range – safeguard: classify as SCANNED"],
            )

    @staticmethod
    def _build_reasoning(
        scores: ComponentScores,
        analysis: dict,
        scanned_likelihood: float,
    ) -> list[str]:
        """Build human-readable explanation of decision."""
        reasons = []

        # Component explanations
        if scores.invisible_font_score > 0.6:
            reasons.append(
                f"Invisible fonts detected ({scores.invisible_font_score:.2f}): OCR marker"
            )

        if scores.text_corruption_score > 0.5:
            reasons.append(
                f"Text corruption detected ({scores.text_corruption_score:.2f}): garbled unicode or anomalous entropy"
            )

        if scores.image_dominance_score > 0.5:
            reasons.append(
                f"Image dominance ({scores.image_dominance_score:.2f}): {analysis['image_area_ratio']:.1%} of page area"
            )

        if scores.text_font_density_score > 0.5:
            chars_per_font = analysis["total_chars"] / max(analysis["font_count"], 1)
            reasons.append(
                f"Low font diversity ({scores.text_font_density_score:.2f}): {chars_per_font:.0f} chars/font"
            )

        # Final score
        if scanned_likelihood > 0.75:
            reasons.append(f"Overall: HIGH SCAN LIKELIHOOD ({scanned_likelihood:.2f})")
        elif scanned_likelihood < 0.25:
            reasons.append(f"Overall: HIGH NATIVE LIKELIHOOD ({scanned_likelihood:.2f})")
        else:
            reasons.append(f"Overall: UNCERTAIN ({scanned_likelihood:.2f}) – defaulting to SCANNED")

        return reasons

    # ── Transaction Helpers ──────────────────────────────────────

    async def classify_transaction(self, transaction: Transaction) -> DocumentKind:
        """Classify a transaction's file and update the database."""
        if not transaction.file_path:
            raise ValueError("Transaction has no file_path set")

        file_path = Path(transaction.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {transaction.file_path}")

        file_bytes = file_path.read_bytes()
        result = self.classify_with_confidence(file_bytes)
        doc_kind = result.document_kind

        is_scan = doc_kind != DocumentKind.NATIVE_PDF

        # Log the reasoning if uncertain
        if result.confidence < 0.7:
            logger.warning(
                "Low confidence classification for %s (%.2f): %s",
                transaction.filename,
                result.confidence,
                "; ".join(result.reasoning),
            )

        await self._update_transaction(transaction.id, is_scan, doc_kind)

        return doc_kind

    async def _update_transaction(
        self,
        transaction_id: uuid.UUID,
        is_scan: bool,
        doc_kind: DocumentKind,
    ) -> None:
        """Update the transaction with classification result."""
        stmt = (
            update(Transaction)
            .where(Transaction.id == transaction_id)
            .values(is_scan=is_scan, status="classified")
        )

        if self._session:
            await self._session.execute(stmt)
        else:
            async with get_session_context() as session:
                await session.execute(stmt)
