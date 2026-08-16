"""PDF classifier for detecting native PDF vs scanned documents.

Uses pdfminer.six to analyze PDF structure and determine if the document
has a selectable text layer (native) or is a scanned image.

**Classification policy (conservative):**
    A *native* PDF misclassified as *scanned* is tolerable
    (we simply OCR it – slightly slower, but no data loss).
    A *scanned* PDF misclassified as *native* is **catastrophic**
    (pdfminer returns garbage / empty text, extraction fails silently).

    → The classifier MUST default to SCANNED and only declare NATIVE_PDF
      when multiple strong indicators converge.

Klasifikátor PDF pro detekci nativního PDF vs naskenovaného dokumentu.
"""

from __future__ import annotations

import io
import logging
import uuid
from collections.abc import Sequence
from pathlib import Path

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTFigure,
    LTImage,
    LTPage,
)
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.classification.base import BaseClassifier, DocumentKind
from src.infrastructure.db.database import get_session_context
from src.infrastructure.db.models import Transaction

logger = logging.getLogger(__name__)


# ── Thresholds ───────────────────────────────────────────────────
# All thresholds are deliberately HIGH to avoid false-native verdicts.

# Minimum total characters across all pages to even consider "native".
MIN_TOTAL_CHARS = 200

# Minimum ratio of pages that must individually pass the "has real text"
# check before the document is considered native.
MIN_PAGE_TEXT_RATIO = 0.8

# Minimum distinct unicode codepoints in extracted text.
# Scanned PDFs with OCR layers often have garbled / low-diversity text.
MIN_CODEPOINT_DIVERSITY = 15

# Minimum number of font programs referenced by the text operators.
# Real native PDFs embed fonts; scanned OCR layers typically use a
# single invisible font.
MIN_FONT_COUNT = 2

# Per-page minimum character count to mark the page as "text-bearing".
MIN_PAGE_CHARS = 50

# Maximum ratio of LTImage / LTFigure area to total page area.
# If images dominate, it is likely a scan even if some text exists.
MAX_IMAGE_AREA_RATIO = 0.60


class PDFClassifier(BaseClassifier):
    """Concrete classifier using pdfminer.six for PDF analysis.

    Classification strategy (conservative – prefer false-scan):
        1. If the file is not a PDF at all → SCANNED.
        2. Extract page layouts via pdfminer and gather:
           a. total character count and codepoint diversity,
           b. number of distinct font names referenced,
           c. per-page text / image area ratio,
           d. ratio of pages that carry "enough" text.
        3. ALL of the following must hold to classify as NATIVE_PDF:
           - total characters  ≥  MIN_TOTAL_CHARS
           - codepoint diversity  ≥  MIN_CODEPOINT_DIVERSITY
           - distinct fonts  ≥  MIN_FONT_COUNT
           - pages-with-text ratio  ≥  MIN_PAGE_TEXT_RATIO
           - image area ratio  ≤  MAX_IMAGE_AREA_RATIO
        4. If any single criterion fails → SCANNED.
    """

    def __init__(self, session: AsyncSession | None = None) -> None:
        self._session = session

    # ── public API ───────────────────────────────────────────────

    async def classify(self, file_bytes: bytes) -> DocumentKind:
        """Classify a PDF document as native or scanned.

        Returns DocumentKind.SCANNED by default (safe fallback).
        """
        if not file_bytes.startswith(b"%PDF"):
            return DocumentKind.SCANNED

        try:
            analysis = self._deep_analyze(file_bytes)
            return self._decide(analysis)
        except Exception:
            logger.debug("pdfminer analysis failed – defaulting to SCANNED", exc_info=True)
            return DocumentKind.SCANNED

    # ── deep structural analysis ─────────────────────────────────

    def _deep_analyze(self, file_bytes: bytes) -> dict:
        """Perform deep layout analysis of every page.

        Returns a dict with aggregated metrics.
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
        all_fonts: set[str] = set()
        total_pages = 0
        pages_with_text = 0
        total_image_area = 0.0
        total_page_area = 0.0

        stream.seek(0)
        for page in PDFPage.create_pages(document):
            total_pages += 1
            if total_pages > 20:  # cap for performance
                break

            try:
                interpreter.process_page(page)
                layout: LTPage = device.get_result()
            except Exception:
                continue

            page_width = layout.width if layout.width else 595  # A4 default
            page_height = layout.height if layout.height else 842
            page_area = page_width * page_height
            total_page_area += page_area

            page_chars = 0
            page_image_area = 0.0

            for element in self._iter_layout(layout):
                if isinstance(element, LTChar):
                    page_chars += 1
                    total_chars += 1
                    all_codepoints.add(ord(element.get_text()))
                    fontname = getattr(element, "fontname", None)
                    if fontname:
                        all_fonts.add(fontname)

                elif isinstance(element, LTAnno):
                    page_chars += 1
                    total_chars += 1
                    txt = element.get_text()
                    if txt:
                        all_codepoints.add(ord(txt[0]))

                elif isinstance(element, (LTImage, LTFigure)):
                    w = abs(element.x1 - element.x0)
                    h = abs(element.y1 - element.y0)
                    page_image_area += w * h

            total_image_area += page_image_area

            if page_chars >= MIN_PAGE_CHARS:
                pages_with_text += 1

        image_area_ratio = (
            total_image_area / total_page_area if total_page_area > 0 else 1.0
        )
        page_text_ratio = (
            pages_with_text / total_pages if total_pages > 0 else 0.0
        )

        return {
            "total_chars": total_chars,
            "codepoint_diversity": len(all_codepoints),
            "font_count": len(all_fonts),
            "total_pages": total_pages,
            "pages_with_text": pages_with_text,
            "page_text_ratio": page_text_ratio,
            "image_area_ratio": image_area_ratio,
        }

    @staticmethod
    def _iter_layout(layout_obj):
        """Recursively iterate over all layout elements."""
        if hasattr(layout_obj, "__iter__"):
            for child in layout_obj:
                yield child
                yield from PDFClassifier._iter_layout(child)

    # ── decision logic ───────────────────────────────────────────

    def _decide(self, analysis: dict) -> DocumentKind:
        """Apply conservative decision rules.

        Every condition must pass; a single failure → SCANNED.
        """
        reasons: list[str] = []

        if analysis["total_chars"] < MIN_TOTAL_CHARS:
            reasons.append(
                f"total_chars={analysis['total_chars']} < {MIN_TOTAL_CHARS}"
            )

        if analysis["codepoint_diversity"] < MIN_CODEPOINT_DIVERSITY:
            reasons.append(
                f"codepoint_diversity={analysis['codepoint_diversity']} < {MIN_CODEPOINT_DIVERSITY}"
            )

        if analysis["font_count"] < MIN_FONT_COUNT:
            reasons.append(
                f"font_count={analysis['font_count']} < {MIN_FONT_COUNT}"
            )

        if analysis["page_text_ratio"] < MIN_PAGE_TEXT_RATIO:
            reasons.append(
                f"page_text_ratio={analysis['page_text_ratio']:.2f} < {MIN_PAGE_TEXT_RATIO}"
            )

        if analysis["image_area_ratio"] > MAX_IMAGE_AREA_RATIO:
            reasons.append(
                f"image_area_ratio={analysis['image_area_ratio']:.2f} > {MAX_IMAGE_AREA_RATIO}"
            )

        if reasons:
            logger.debug("Classified as SCANNED: %s", "; ".join(reasons))
            return DocumentKind.SCANNED

        logger.debug(
            "Classified as NATIVE_PDF: chars=%d diversity=%d fonts=%d page_ratio=%.2f img_ratio=%.2f",
            analysis["total_chars"],
            analysis["codepoint_diversity"],
            analysis["font_count"],
            analysis["page_text_ratio"],
            analysis["image_area_ratio"],
        )
        return DocumentKind.NATIVE_PDF

    # ── transaction helpers ──────────────────────────────────────

    async def classify_transaction(self, transaction: Transaction) -> DocumentKind:
        """Classify a transaction's file and update the database."""
        if not transaction.file_path:
            raise ValueError("Transaction has no file_path set")

        file_path = Path(transaction.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {transaction.file_path}")

        file_bytes = file_path.read_bytes()
        doc_kind = await self.classify(file_bytes)

        is_scan = doc_kind != DocumentKind.NATIVE_PDF
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

    async def classify_batch(
        self,
        transactions: Sequence[Transaction],
    ) -> list[tuple[Transaction, DocumentKind]]:
        """Classify multiple transactions."""
        results: list[tuple[Transaction, DocumentKind]] = []

        for tx in transactions:
            try:
                doc_kind = await self.classify_transaction(tx)
                results.append((tx, doc_kind))
            except Exception as e:
                logger.warning("Error classifying transaction %s: %s", tx.id, e)
                results.append((tx, DocumentKind.SCANNED))

        return results


def create_pdf_classifier(session: AsyncSession | None = None) -> PDFClassifier:
    """Factory function to create a PDFClassifier instance."""
    return PDFClassifier(session=session)
