"""Layout Information Extraction pipeline.

Implements the full pipeline from the academic diagram:

    Image
      │
      ├─► Image Direction Correction  (SkewCorrector)
      │
      └─► Layout Analysis
            ├─► Table Recognition     (TableExtractor)
            │     └─► HTML table string
            └─► OCR (EasyOCR)
                  └─► Spatially-ordered text blocks
      │
      └─► Layout Recovery             (_build_llm_payload)
            └─► LLM-ready text string

Public API
──────────
    LayoutPipeline.analyze(image_path_or_array) -> LayoutResult
    LayoutPipeline.analyze_pdf(pdf_path)        -> list[LayoutResult]
    LayoutResult.llm_payload                    -> str (send to LLM)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .skew_corrector import SkewCorrector, SkewResult
from .table_extractor import TableExtractor, TableRegion

logger = logging.getLogger(__name__)

# ── Lazy EasyOCR singleton ───────────────────────────────────────────────────

_easyocr_reader: Any | None = None


def _get_reader(langs: list[str] = None):
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        langs = langs or ["cs", "en"]
        logger.info("Initialising EasyOCR (langs=%s)…", langs)
        _easyocr_reader = easyocr.Reader(langs, gpu=False, verbose=False)
        logger.info("EasyOCR ready.")
    return _easyocr_reader


# ── Data containers ──────────────────────────────────────────────────────────

@dataclass
class DetectedRegion:
    """A single layout region detected on a page.

    Attributes:
        region_type:  ``"text"`` | ``"table"`` | ``"header"`` | ``"footer"``.
        content:      Plain text for text/header/footer; HTML for tables.
        bbox:         ``(x1, y1, x2, y2)`` in image pixels.
        confidence:   Average OCR confidence for this region (0–1).
        is_table:     Convenience flag.
    """
    region_type: str
    content: str
    bbox: tuple[int, int, int, int]
    confidence: float = 0.0

    @property
    def is_table(self) -> bool:
        return self.region_type == "table"


@dataclass
class LayoutResult:
    """Complete layout-extraction result for one document page.

    This class mirrors ``AnalysisResult`` from ``ocr_processor.py`` so
    both backends are interchangeable in experiments.

    Attributes:
        regions:          Detected regions in reading order.
        llm_payload:      LLM-ready string (tables as HTML, text as plain).
        page_index:       0-based page number.
        skew_angle_deg:   Detected skew angle (0 if not corrected).
        latency_ms:       Total wall-clock time for the full pipeline (ms).
        ocr_latency_ms:   Time spent in EasyOCR only (ms).
        table_count:      Number of table regions.
        block_count:      Total region count.
        raw_ocr_boxes:    Raw EasyOCR output (detail=1) for further analysis.
    """
    regions:          list[DetectedRegion] = field(default_factory=list)
    llm_payload:      str = ""
    page_index:       int = 0
    skew_angle_deg:   float = 0.0
    latency_ms:       float = 0.0
    ocr_latency_ms:   float = 0.0
    table_count:      int = 0
    block_count:      int = 0
    raw_ocr_boxes:    list = field(default_factory=list)

    # Bridge to project OCRResult
    def to_ocr_result(self, engine_name: str = "layout_easyocr"):
        from src.core.ocr_engine.base import OCRResult
        return OCRResult(
            full_text=self.llm_payload,
            pages=[self.llm_payload],
            confidence=float(np.mean([r.confidence for r in self.regions]) if self.regions else 0.0),
            engine_name=engine_name,
            latency_ms=self.latency_ms,
        )


# ── Main pipeline ─────────────────────────────────────────────────────────────

class LayoutPipeline:
    """Full Layout Information Extraction pipeline.

    Steps
    ─────
    1. Load image (bytes / path / ndarray).
    2. Skew correction (optional, default on).
    3. EasyOCR with bounding-box coordinates.
    4. Table detection (line-based → bbox-grid fallback).
    5. Non-table text grouping into spatial blocks.
    6. Reading-order reconstruction.
    7. LLM payload assembly (tables → HTML, text → plain).

    Args:
        langs:            EasyOCR language list.  Default ``["cs", "en"]``.
        correct_skew:     Run skew correction before OCR.
        dpi:              Used only for PDF rendering annotation.
        table_prefer_lines: Use line-based table detection first.
        min_confidence:   Drop OCR tokens below this confidence.
        row_gap_px:       Vertical pixel gap for grouping text into lines.
        col_gap_ratio:    Column-split threshold as fraction of page width.
        header_ratio:     Top fraction of page treated as header zone.
        footer_ratio:     Bottom fraction of page treated as footer zone.
    """

    def __init__(
        self,
        langs: list[str] | None = None,
        *,
        correct_skew: bool = True,
        dpi: int = 150,
        table_prefer_lines: bool = True,
        min_confidence: float = 0.3,
        row_gap_px: int = 12,
        col_gap_ratio: float = 0.45,
        header_ratio: float = 0.08,
        footer_ratio: float = 0.05,
    ) -> None:
        self.langs             = langs or ["cs", "en"]
        self.correct_skew      = correct_skew
        self.dpi               = dpi
        self.min_confidence    = min_confidence
        self.row_gap_px        = row_gap_px
        self.col_gap_ratio     = col_gap_ratio
        self.header_ratio      = header_ratio
        self.footer_ratio      = footer_ratio
        self._skew_corrector   = SkewCorrector()
        self._table_extractor  = TableExtractor(prefer_lines=table_prefer_lines)

    # ── Public API ───────────────────────────────────────────────────────────

    def analyze(
        self,
        image_input: bytes | str | Path | np.ndarray,
        *,
        page_index: int = 0,
    ) -> LayoutResult:
        """Run the full pipeline on a single page image.

        Args:
            image_input: BGR ndarray, file path, or raw image bytes.
            page_index:  Page number hint (for multi-page PDFs).

        Returns:
            :class:`LayoutResult` with all detected regions + LLM payload.
        """
        t_start = time.perf_counter()
        image   = self._load_image(image_input)

        # Step 1 — Skew correction
        skew_angle = 0.0
        if self.correct_skew:
            skew_result: SkewResult = self._skew_corrector.correct(image)
            image       = skew_result.image
            skew_angle  = skew_result.angle_deg

        # Step 2 — OCR with bounding boxes
        t_ocr   = time.perf_counter()
        reader  = _get_reader(self.langs)
        raw_boxes: list[tuple[list, str, float]] = reader.readtext(
            image, detail=1
        )
        ocr_ms  = (time.perf_counter() - t_ocr) * 1000

        # Filter low-confidence tokens
        if self.min_confidence > 0:
            raw_boxes = [b for b in raw_boxes if b[2] >= self.min_confidence]

        # Step 3 — Table detection
        table_regions: list[TableRegion] = self._table_extractor.extract(
            image, raw_boxes
        )
        table_bboxes = [t.bbox for t in table_regions]

        # Step 4 — Separate table tokens from text tokens
        non_table_boxes = [
            b for b in raw_boxes
            if not _box_inside_any(b[0], table_bboxes)
        ]

        # Step 5 — Group non-table text into spatial blocks
        h, w = image.shape[:2]
        text_regions = self._group_text_regions(non_table_boxes, image_h=h, image_w=w)

        # Step 6 — Merge all regions, sort reading order
        regions: list[DetectedRegion] = []
        for tr in table_regions:
            regions.append(DetectedRegion(
                region_type="table",
                content=tr.html,
                bbox=tr.bbox,
                confidence=1.0,
            ))
        regions.extend(text_regions)
        regions = self._sort_reading_order(regions, image_w=w)

        # Step 7 — Build LLM payload
        llm_payload = self._build_llm_payload(regions)

        latency_ms = (time.perf_counter() - t_start) * 1000
        table_count = sum(1 for r in regions if r.is_table)

        return LayoutResult(
            regions=regions,
            llm_payload=llm_payload,
            page_index=page_index,
            skew_angle_deg=skew_angle,
            latency_ms=latency_ms,
            ocr_latency_ms=ocr_ms,
            table_count=table_count,
            block_count=len(regions),
            raw_ocr_boxes=raw_boxes,
        )

    def analyze_pdf(
        self,
        pdf_path: str | Path,
        *,
        max_pages: int = 1,
    ) -> list[LayoutResult]:
        """Render PDF pages and analyze each one.

        Args:
            pdf_path:   Path to the PDF file.
            max_pages:  Maximum number of pages to process (default: 1,
                        because energy invoices are typically single-page).

        Returns:
            List of :class:`LayoutResult`, one per processed page.
        """
        import pypdfium2 as pdfium

        doc     = pdfium.PdfDocument(str(pdf_path))
        results = []
        for idx in range(min(len(doc), max_pages)):
            page = doc[idx]
            bm   = page.render(scale=self.dpi / 72)
            img  = bm.to_pil()
            bgr  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            results.append(self.analyze(bgr, page_index=idx))
        return results

    def get_llm_payload(self, pdf_path: str | Path) -> tuple[str, LayoutResult]:
        """One-shot: analyze first page of a PDF, return (payload, result)."""
        result = self.analyze_pdf(pdf_path, max_pages=1)[0]
        return result.llm_payload, result

    # ── Text grouping ────────────────────────────────────────────────────────

    def _group_text_regions(
        self,
        ocr_boxes: list[tuple[list, str, float]],
        image_h: int,
        image_w: int,
    ) -> list[DetectedRegion]:
        """Group individual OCR word boxes into logical text regions.

        Algorithm:
        1. Convert quads to axis-aligned rects with Y-centre.
        2. Cluster rects by Y-centre into lines (gap = row_gap_px).
        3. Within each line, sort left-to-right and concatenate text.
        4. Cluster lines by vertical proximity into paragraphs.
        5. Label very-top / very-bottom paragraphs as header / footer.
        """
        if not ocr_boxes:
            return []

        # Convert to axis-aligned rects
        rects: list[tuple[int, int, int, int, str, float]] = []
        for quad, text, conf in ocr_boxes:
            pts = np.array(quad, dtype=np.float32)
            x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
            rects.append((x1, y1, x2, y2, text, conf))

        # Cluster into lines by Y-centre
        y_centres = [(r[1] + r[3]) // 2 for r in rects]
        from .table_extractor import _cluster_by_axis
        line_clusters = _cluster_by_axis(y_centres, gap=self.row_gap_px)

        lines: list[tuple[int, int, int, int, str, float]] = []
        for cluster in line_clusters:
            cluster_rects = [rects[i] for i in cluster]
            cluster_rects.sort(key=lambda r: r[0])  # sort left→right
            line_text = " ".join(r[4] for r in cluster_rects)
            x1 = min(r[0] for r in cluster_rects)
            y1 = min(r[1] for r in cluster_rects)
            x2 = max(r[2] for r in cluster_rects)
            y2 = max(r[3] for r in cluster_rects)
            avg_conf = float(np.mean([r[5] for r in cluster_rects]))
            lines.append((x1, y1, x2, y2, line_text, avg_conf))

        # Cluster lines into paragraph blocks (larger gap threshold)
        para_gap = self.row_gap_px * 3
        line_y_centres = [(l[1] + l[3]) // 2 for l in lines]
        para_clusters  = _cluster_by_axis(line_y_centres, gap=para_gap)

        regions: list[DetectedRegion] = []
        header_threshold = image_h * self.header_ratio
        footer_threshold = image_h * (1.0 - self.footer_ratio)

        for cluster in para_clusters:
            para_lines = [lines[i] for i in cluster]
            text  = "\n".join(l[4] for l in para_lines)
            x1    = min(l[0] for l in para_lines)
            y1    = min(l[1] for l in para_lines)
            x2    = max(l[2] for l in para_lines)
            y2    = max(l[3] for l in para_lines)
            conf  = float(np.mean([l[5] for l in para_lines]))
            y_mid = (y1 + y2) / 2

            if y_mid < header_threshold:
                rtype = "header"
            elif y_mid > footer_threshold:
                rtype = "footer"
            else:
                rtype = "text"

            regions.append(DetectedRegion(
                region_type=rtype,
                content=text.strip(),
                bbox=(x1, y1, x2, y2),
                confidence=conf,
            ))
        return regions

    # ── Reading-order sort ───────────────────────────────────────────────────

    def _sort_reading_order(
        self,
        regions: list[DetectedRegion],
        image_w: int,
    ) -> list[DetectedRegion]:
        """Sort regions top-to-bottom, with column detection.

        Regions whose X-centre is in the left half of the page are placed
        before right-column regions at the same vertical position.
        """
        def _key(r: DetectedRegion) -> tuple[int, int, int]:
            x_centre = (r.bbox[0] + r.bbox[2]) // 2
            col = 0 if x_centre < image_w * self.col_gap_ratio else 1
            return (col, r.bbox[1], r.bbox[0])

        return sorted(regions, key=_key)

    # ── LLM payload builder ──────────────────────────────────────────────────

    @staticmethod
    def _build_llm_payload(regions: list[DetectedRegion]) -> str:
        """Assemble LLM-ready text: tables → HTML, text blocks → plain."""
        parts: list[str] = []
        for r in regions:
            if r.region_type == "header":
                parts.append(f"[HEADER]\n{r.content}")
            elif r.region_type == "footer":
                parts.append(f"[FOOTER]\n{r.content}")
            elif r.is_table:
                parts.append(r.content)  # already HTML
            else:
                parts.append(r.content)
        return "\n\n".join(p for p in parts if p.strip())

    # ── Image loader ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_image(source: bytes | str | Path | np.ndarray) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            return img
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image from bytes")
        return img


# ── Stand-alone helper: box-in-region check ──────────────────────────────────

def _box_inside_any(
    quad: list,
    regions: list[tuple[int, int, int, int]],
    overlap_threshold: float = 0.6,
) -> bool:
    """Return True if the OCR quad's centre overlaps with any region bbox."""
    pts = np.array(quad, dtype=np.float32)
    cx  = float(pts[:, 0].mean())
    cy  = float(pts[:, 1].mean())
    for x1, y1, x2, y2 in regions:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False
