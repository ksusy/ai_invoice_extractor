"""High-quality table and text extraction from scanned documents.

Uses PaddleOCR PPStructure for layout analysis: tables are converted to
HTML strings ready for LLM consumption, while non-table regions are kept
as plain text.  Everything is returned in correct reading order so that
Claude / GPT receives a faithful representation of the original document.

Modul pro extrakci tabulek a textu ze skenů pomocí PaddleOCR.

Design decisions
────────────────
- Heavy deps (cv2, numpy, paddleocr) are imported *lazily* so the module
  can be safely imported even when only a subset of ``pip install .[ocr]``
  is available.
- All preprocessing parameters are exposed via :class:`PreprocessConfig`
  so that Jupyter experiments can sweep them without touching source code.
- :meth:`DocumentAnalyzer.analyze` returns an :class:`AnalysisResult` that
  carries both structured blocks *and* timing metrics, bridging the gap
  with :class:`~src.core.ocr_engine.base.OCRResult`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

if TYPE_CHECKING:
    import numpy as np

    from src.core.ocr_engine.base import OCRResult

logger = logging.getLogger(__name__)


# ── Lazy dependency helpers ──────────────────────────────────────────────

_cv2: Any | None = None
_np: Any | None = None
_PPStructure: Any | None = None


def _ensure_numpy():
    """Import numpy lazily and cache the module reference."""
    global _np
    if _np is not None:
        return _np
    try:
        import numpy as np  # noqa: F811

        _np = np
        return np
    except ImportError as exc:
        raise ImportError(
            "NumPy is required for image processing. "
            "Install with:  pip install numpy"
        ) from exc


def _ensure_cv2():
    """Import OpenCV lazily and cache the module reference."""
    global _cv2
    if _cv2 is not None:
        return _cv2
    try:
        import cv2

        _cv2 = cv2
        return cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for image preprocessing. "
            "Install with:  pip install opencv-python-headless"
        ) from exc


def _ensure_ppstructure():
    """Import PaddleOCR PPStructure lazily and cache the class reference."""
    global _PPStructure
    if _PPStructure is not None:
        return _PPStructure
    try:
        from paddleocr import PPStructure  # type: ignore[import-untyped]

        _PPStructure = PPStructure
        return PPStructure
    except ImportError as exc:
        raise ImportError(
            "PaddleOCR is required for table extraction. "
            "Install with:  pip install paddlepaddle paddleocr"
        ) from exc


def _ensure_pdf2image():
    """Import pdf2image lazily."""
    try:
        from pdf2image import convert_from_bytes

        return convert_from_bytes
    except ImportError as exc:
        raise ImportError(
            "pdf2image is required for PDF-to-image conversion. "
            "Install with:  pip install pdf2image"
        ) from exc


# ── Configuration dataclasses ────────────────────────────────────────────


@dataclass
class PreprocessConfig:
    """Tunable image preprocessing parameters.

    Expose this in notebooks to sweep values and compare OCR quality::

        for h in [5, 10, 15, 20]:
            cfg = PreprocessConfig(denoise_strength=h)
            result = analyzer.analyze(img, preprocess_config=cfg)

    Attributes:
        denoise_strength: ``h`` parameter of ``fastNlMeansDenoising``.
            Higher → more smoothing (risk losing thin text strokes).
        denoise_template_window: Size of the template patch (must be odd).
        denoise_search_window: Size of the search area (must be odd).
        adaptive_block_size: Block size for ``adaptiveThreshold`` (must be odd, ≥3).
        adaptive_c: Constant subtracted from the mean in adaptive threshold.
        adaptive_method: ``"GAUSSIAN"`` or ``"MEAN"`` — gaussian gives smoother edges.
        skip_denoise: Disable denoising step entirely.
        skip_threshold: Disable binarisation step entirely.
    """

    denoise_strength: int = 10
    denoise_template_window: int = 7
    denoise_search_window: int = 21
    adaptive_block_size: int = 15
    adaptive_c: int = 8
    adaptive_method: Literal["GAUSSIAN", "MEAN"] = "GAUSSIAN"
    skip_denoise: bool = False
    skip_threshold: bool = False

    def __post_init__(self) -> None:
        if self.adaptive_block_size < 3 or self.adaptive_block_size % 2 == 0:
            raise ValueError(
                f"adaptive_block_size must be odd and ≥3, got {self.adaptive_block_size}"
            )
        if self.denoise_template_window % 2 == 0:
            raise ValueError(
                f"denoise_template_window must be odd, got {self.denoise_template_window}"
            )


# Default preprocessing — used when no override is supplied.
DEFAULT_PREPROCESS = PreprocessConfig()


# ── Visualisation style ──────────────────────────────────────────────────
# Kept as a type alias; the actual value is read from Settings at runtime
# with a fallback constant so the module works without a database / .env.

_VIS_STYLE = Literal["COLOR", "BW"]
_FALLBACK_UI_STYLE: _VIS_STYLE = "COLOR"


def _resolve_ui_style(override: _VIS_STYLE | None = None) -> _VIS_STYLE:
    """Determine visualisation style: explicit override → Settings → fallback."""
    if override is not None:
        return override
    try:
        from src.config.settings import get_settings

        mode = get_settings().print_mode.value  # "color" | "grayscale"
        return "COLOR" if mode == "color" else "BW"
    except Exception:
        return _FALLBACK_UI_STYLE


# Re-export for backward compatibility with code that imported UI_STYLE.
UI_STYLE: _VIS_STYLE = _FALLBACK_UI_STYLE


# ── Data containers ──────────────────────────────────────────────────────


@dataclass
class DetectedBlock:
    """A single region detected by layout analysis.

    Attributes:
        block_type: ``"table"`` | ``"text"`` | ``"title"`` | ``"figure"``.
        content: HTML string for tables, plain text for other blocks.
        bbox: Bounding box ``(x_min, y_min, x_max, y_max)``.
        confidence: Model confidence for this region (0.0–1.0).
        raw: Original dict from PPStructure (for experiment inspection).
    """

    block_type: str
    content: str
    bbox: tuple[int, int, int, int]
    confidence: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def area(self) -> int:
        """Bounding-box area in pixels."""
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])

    @property
    def is_table(self) -> bool:
        return self.block_type == "table"


@dataclass
class AnalysisResult:
    """Complete extraction result for a single page / image.

    Attributes:
        blocks: Detected regions sorted in reading order (top→bottom, left→right).
        llm_payload: Pre-merged string ready for LLM prompt injection.
        page_index: Page number (0-based) within a multi-page document.
        latency_ms: Total wall-clock time for analysis (ms).
        preprocess_ms: Time spent in preprocessing only (ms).
        block_count: Number of blocks after confidence filtering.
        table_count: How many of those blocks are tables.
        min_confidence: Lowest block confidence that passed the filter.
        annotated_image: Optional visualisation with bounding boxes drawn.
        preprocessing_debug: Intermediate images (gray, denoised, binary) when
            ``debug_preprocessing=True``.
    """

    blocks: list[DetectedBlock] = field(default_factory=list)
    llm_payload: str = ""
    page_index: int = 0
    latency_ms: float = 0.0
    preprocess_ms: float = 0.0
    block_count: int = 0
    table_count: int = 0
    min_confidence: float = 0.0
    annotated_image: Any = None  # np.ndarray | None — lazy to avoid top-level numpy
    preprocessing_debug: dict[str, Any] = field(default_factory=dict)

    # ── Bridge to the rest of the pipeline ───────────────────────

    def to_ocr_result(self, engine_name: str = "ppstructure") -> "OCRResult":
        """Convert to project-standard :class:`OCRResult`.

        This allows ``DocumentAnalyzer`` output to be stored in the
        ``extraction_results`` DB table via the normal pipeline path.
        """
        from src.core.ocr_engine.base import OCRResult

        return OCRResult(
            full_text=self.llm_payload,
            pages=[self.llm_payload],
            confidence=self.min_confidence,
            engine_name=engine_name,
            latency_ms=self.latency_ms,
        )


# ── Core analyser ────────────────────────────────────────────────────────


class DocumentAnalyzer:
    """Extract tables (→ HTML) and text from scanned document images.

    Typical usage::

        analyzer = DocumentAnalyzer()
        result = analyzer.analyze(image_bytes)
        prompt = result.llm_payload          # send this to Claude / GPT

    Experiment usage (notebooks)::

        from src.core.ocr_engine.ocr_processor import DocumentAnalyzer, PreprocessConfig

        cfg = PreprocessConfig(denoise_strength=15, adaptive_block_size=11)
        analyzer = DocumentAnalyzer(lang="en", min_confidence=0.5)
        result = analyzer.analyze(img, preprocess_config=cfg, debug_preprocessing=True)

        # Inspect intermediate images
        show(result.preprocessing_debug["gray"])
        show(result.preprocessing_debug["denoised"])
        show(result.preprocessing_debug["binary"])

    Args:
        lang: OCR language code understood by PaddleOCR (default ``"en"``).
        use_gpu: Enable GPU acceleration in PaddleOCR.
        show_log: Show PaddleOCR internal logs.
        ui_style: Override visualisation style (``"COLOR"`` / ``"BW"``).
        min_confidence: Drop blocks below this confidence.  *Experiment knob!*
        pdf_dpi: DPI for PDF→image conversion.  *Experiment knob!*
    """

    def __init__(
        self,
        lang: str = "en",
        *,
        use_gpu: bool = False,
        show_log: bool = False,
        ui_style: _VIS_STYLE | None = None,
        min_confidence: float = 0.0,
        pdf_dpi: int = 300,
    ) -> None:
        self._lang = lang
        self._use_gpu = use_gpu
        self._show_log = show_log
        self._ui_style = _resolve_ui_style(ui_style)
        self._min_confidence = min_confidence
        self._pdf_dpi = pdf_dpi
        self._engine: Any | None = None  # lazy-initialised PPStructure

    # ── Public API ───────────────────────────────────────────────

    def analyze(
        self,
        image_input: bytes | str | Path | "np.ndarray",
        *,
        preprocess: bool = True,
        preprocess_config: PreprocessConfig | None = None,
        draw_boxes: bool = False,
        debug_preprocessing: bool = False,
        page_index: int = 0,
    ) -> AnalysisResult:
        """Run layout analysis + OCR on a single image.

        Args:
            image_input: Raw bytes, file path, or a NumPy BGR array.
            preprocess: Apply denoising / binarisation before analysis.
            preprocess_config: Override default preprocessing parameters.
            draw_boxes: Produce an annotated copy of the image.
            debug_preprocessing: Store intermediate images in result.
            page_index: Page number hint (for multi-page workflows).

        Returns:
            An :class:`AnalysisResult` with blocks sorted in reading order.
        """
        t_start = time.perf_counter()
        cv2 = _ensure_cv2()
        np = _ensure_numpy()
        img = self._load_image(image_input, cv2, np)

        # ── Preprocessing ────────────────────────────────────────
        t_prep = time.perf_counter()
        debug_imgs: dict[str, Any] = {}
        cfg = preprocess_config or DEFAULT_PREPROCESS

        if preprocess:
            processed, debug_imgs = self._prepare_image(img, cfg, debug=debug_preprocessing)
        else:
            processed = img

        preprocess_ms = (time.perf_counter() - t_prep) * 1000

        logger.info(
            "Running PPStructure on %dx%d image (page %d, preprocess=%.1fms)",
            processed.shape[1],
            processed.shape[0],
            page_index,
            preprocess_ms,
        )

        # ── Structure recognition ────────────────────────────────
        raw_results = self._run_structure(processed)
        blocks = self._parse_blocks(raw_results)

        # ── Filter low-confidence blocks ─────────────────────────
        if self._min_confidence > 0:
            before = len(blocks)
            blocks = [b for b in blocks if b.confidence >= self._min_confidence]
            dropped = before - len(blocks)
            if dropped:
                logger.info("Dropped %d blocks below confidence %.2f", dropped, self._min_confidence)

        blocks = self._sort_reading_order(blocks)

        # ── Post-processing ──────────────────────────────────────
        annotated = self._draw_boxes(img, blocks) if draw_boxes else None
        llm_payload = self._build_llm_payload(blocks)
        latency_ms = (time.perf_counter() - t_start) * 1000
        table_count = sum(1 for b in blocks if b.is_table)
        confidences = [b.confidence for b in blocks]

        return AnalysisResult(
            blocks=blocks,
            llm_payload=llm_payload,
            page_index=page_index,
            latency_ms=latency_ms,
            preprocess_ms=preprocess_ms,
            block_count=len(blocks),
            table_count=table_count,
            min_confidence=min(confidences) if confidences else 0.0,
            annotated_image=annotated,
            preprocessing_debug=debug_imgs,
        )

    def analyze_pdf(
        self,
        pdf_input: bytes | str | Path,
        **kwargs,
    ) -> list[AnalysisResult]:
        """Convert a PDF to images and analyse every page.

        Args:
            pdf_input: Raw PDF bytes or path to a ``.pdf`` file.
            **kwargs: Forwarded to :meth:`analyze`.

        Returns:
            One :class:`AnalysisResult` per page.
        """
        convert_from_bytes = _ensure_pdf2image()
        np = _ensure_numpy()

        if isinstance(pdf_input, (str, Path)):
            pdf_bytes = Path(pdf_input).read_bytes()
        else:
            pdf_bytes = pdf_input

        logger.info("Converting PDF to images at %d DPI", self._pdf_dpi)
        pil_images = convert_from_bytes(pdf_bytes, dpi=self._pdf_dpi)

        results: list[AnalysisResult] = []
        for idx, pil_img in enumerate(pil_images):
            arr = np.array(pil_img)
            # PIL gives RGB, OpenCV needs BGR
            cv2 = _ensure_cv2()
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            results.append(self.analyze(bgr, page_index=idx, **kwargs))
        return results

    def analyze_pages(
        self,
        images: Sequence[bytes | str | Path | "np.ndarray"],
        **kwargs,
    ) -> list[AnalysisResult]:
        """Analyse a sequence of pre-split page images."""
        return [self.analyze(img, page_index=idx, **kwargs) for idx, img in enumerate(images)]

    def get_llm_payload(
        self,
        image_input: bytes | str | Path | "np.ndarray",
        *,
        preprocess: bool = True,
        preprocess_config: PreprocessConfig | None = None,
    ) -> str:
        """One-shot helper: analyse an image and return LLM-ready text.

        Equivalent to ``analyzer.analyze(…).llm_payload``.
        """
        return self.analyze(
            image_input,
            preprocess=preprocess,
            preprocess_config=preprocess_config,
        ).llm_payload

    def get_llm_payload_pdf(
        self,
        pdf_input: bytes | str | Path,
        *,
        preprocess: bool = True,
        preprocess_config: PreprocessConfig | None = None,
    ) -> str:
        """One-shot: analyse all pages of a PDF, return merged LLM-ready text."""
        pages = self.analyze_pdf(
            pdf_input,
            preprocess=preprocess,
            preprocess_config=preprocess_config,
        )
        return "\n\n--- PAGE BREAK ---\n\n".join(r.llm_payload for r in pages)

    # ── Image loading ────────────────────────────────────────────

    @staticmethod
    def _load_image(source: bytes | str | Path | "np.ndarray", cv2, np) -> "np.ndarray":
        """Convert various input types to a BGR NumPy array."""
        if isinstance(source, np.ndarray):
            return source
        if isinstance(source, (str, Path)):
            path = str(source)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image file: {path}")
            return img
        # bytes
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image from bytes — corrupt or unsupported format")
        return img

    # ── Preprocessing ────────────────────────────────────────────

    @staticmethod
    def _prepare_image(
        img: "np.ndarray",
        cfg: PreprocessConfig,
        *,
        debug: bool = False,
    ) -> tuple["np.ndarray", dict[str, Any]]:
        """Denoise and binarise a BGR image for better OCR accuracy.

        Pipeline:
            1. Convert to grayscale.
            2. ``cv2.fastNlMeansDenoising`` (unless ``cfg.skip_denoise``).
            3. ``cv2.adaptiveThreshold`` (unless ``cfg.skip_threshold``).
            4. Convert back to BGR (PPStructure expects 3-channel input).

        Returns:
            ``(processed_bgr, debug_dict)`` — debug_dict is empty when
            ``debug=False``.
        """
        cv2 = _ensure_cv2()
        debug_imgs: dict[str, Any] = {}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if debug:
            debug_imgs["gray"] = gray.copy()

        # Step 2 — denoising
        if cfg.skip_denoise:
            denoised = gray
        else:
            denoised = cv2.fastNlMeansDenoising(
                gray,
                h=cfg.denoise_strength,
                templateWindowSize=cfg.denoise_template_window,
                searchWindowSize=cfg.denoise_search_window,
            )
        if debug:
            debug_imgs["denoised"] = denoised.copy()

        # Step 3 — binarisation
        if cfg.skip_threshold:
            out = denoised
        else:
            method = (
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                if cfg.adaptive_method == "GAUSSIAN"
                else cv2.ADAPTIVE_THRESH_MEAN_C
            )
            out = cv2.adaptiveThreshold(
                denoised,
                maxValue=255,
                adaptiveMethod=method,
                thresholdType=cv2.THRESH_BINARY,
                blockSize=cfg.adaptive_block_size,
                C=cfg.adaptive_c,
            )
        if debug:
            debug_imgs["binary"] = out.copy()

        # PPStructure requires a 3-channel image
        bgr = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        return bgr, debug_imgs

    # ── PPStructure execution ────────────────────────────────────

    def _get_engine(self):
        """Create the PPStructure engine on first use (heavy init)."""
        if self._engine is None:
            PPStructure = _ensure_ppstructure()
            self._engine = PPStructure(
                table=True,
                ocr=True,
                lang=self._lang,
                use_gpu=self._use_gpu,
                show_log=self._show_log,
            )
            logger.info(
                "PPStructure engine initialised (lang=%s, gpu=%s)",
                self._lang,
                self._use_gpu,
            )
        return self._engine

    def _run_structure(self, img: "np.ndarray") -> list[dict]:
        """Execute PPStructure and return the raw result list."""
        engine = self._get_engine()
        try:
            result: list[dict] = engine(img)
        except Exception:
            logger.exception("PPStructure inference failed")
            return []
        logger.debug("PPStructure returned %d raw blocks", len(result))
        return result

    # ── Result parsing ───────────────────────────────────────────

    @staticmethod
    def _parse_blocks(raw_results: list[dict]) -> list[DetectedBlock]:
        """Convert PPStructure output dicts into :class:`DetectedBlock` objects."""
        blocks: list[DetectedBlock] = []

        for item in raw_results:
            region_type: str = item.get("type", "text").lower()
            bbox_raw = item.get("bbox", [0, 0, 0, 0])
            try:
                bbox = (int(bbox_raw[0]), int(bbox_raw[1]), int(bbox_raw[2]), int(bbox_raw[3]))
            except (IndexError, TypeError, ValueError):
                logger.warning("Malformed bbox in block: %r", bbox_raw)
                continue

            score = float(item.get("score", 0.0))

            if region_type == "table":
                res = item.get("res", {})
                html = res.get("html", "") if isinstance(res, dict) else ""
                if not html:
                    logger.warning("Table block at %s produced no HTML — skipping", bbox)
                    continue
                blocks.append(
                    DetectedBlock(
                        block_type="table",
                        content=html,
                        bbox=bbox,
                        confidence=score,
                        raw=item,
                    )
                )
            else:
                # Text / title / figure-caption → extract OCR text lines
                text_lines: list[str] = []
                res = item.get("res", [])
                if isinstance(res, list):
                    for line_item in res:
                        if isinstance(line_item, dict):
                            text_lines.append(line_item.get("text", ""))
                        elif isinstance(line_item, (list, tuple)) and len(line_item) >= 2:
                            text_part = line_item[1]
                            if isinstance(text_part, (list, tuple)):
                                text_lines.append(str(text_part[0]))
                            else:
                                text_lines.append(str(text_part))
                content = "\n".join(t for t in text_lines if t.strip())
                if not content.strip():
                    logger.debug("Empty text block at %s (type=%s) — skipping", bbox, region_type)
                    continue
                blocks.append(
                    DetectedBlock(
                        block_type=region_type,  # preserve title/figure/etc.
                        content=content,
                        bbox=bbox,
                        confidence=score,
                        raw=item,
                    )
                )
        return blocks

    @staticmethod
    def _sort_reading_order(
        blocks: list[DetectedBlock],
        column_gap_ratio: float = 0.4,
    ) -> list[DetectedBlock]:
        """Sort blocks in reading order with basic column detection.

        Blocks whose x-centre differs by more than ``column_gap_ratio``
        of the image width are treated as separate columns (left column
        first, then right column).  Within a column, blocks are sorted
        top-to-bottom.

        ``column_gap_ratio`` is an *experiment knob* — try values 0.2–0.6
        in notebooks.
        """
        if not blocks:
            return blocks

        # Find effective image width from bboxes
        max_x = max(b.bbox[2] for b in blocks)
        gap_threshold = max_x * column_gap_ratio

        # Assign each block to a column bucket based on x-centre
        def _col_key(b: DetectedBlock) -> tuple[int, int]:
            x_centre = (b.bbox[0] + b.bbox[2]) // 2
            col_bucket = int(x_centre // gap_threshold) if gap_threshold > 0 else 0
            return (col_bucket, b.bbox[1])

        return sorted(blocks, key=_col_key)

    # ── LLM payload builder ──────────────────────────────────────

    @staticmethod
    def _build_llm_payload(blocks: list[DetectedBlock]) -> str:
        """Merge text and HTML blocks into a single LLM-ready string.

        Tables are wrapped in ``<table>…</table>`` if not already.
        Text blocks are separated by blank lines.
        """
        parts: list[str] = []

        for block in blocks:
            if block.is_table:
                html = block.content.strip()
                if not html.lower().startswith("<table"):
                    html = f"<table>{html}</table>"
                parts.append(html)
            else:
                parts.append(block.content.strip())

        return "\n\n".join(parts)

    # ── Visualisation ────────────────────────────────────────────

    def _draw_boxes(
        self,
        img: "np.ndarray",
        blocks: list[DetectedBlock],
    ) -> "np.ndarray":
        """Draw bounding boxes onto a copy of the source image.

        Respects :attr:`_ui_style`:
            - ``"COLOR"`` → 2 px green (tables), blue (text), orange (title/figure).
            - ``"BW"``    → 1 px black for everything.
        """
        cv2 = _ensure_cv2()
        canvas = img.copy()

        _COLOR_MAP: dict[str, tuple[int, int, int]] = {
            "table": (0, 200, 0),    # green
            "text": (200, 120, 0),   # blue
            "title": (0, 140, 255),  # orange
            "figure": (200, 0, 200), # magenta
        }

        for block in blocks:
            x1, y1, x2, y2 = block.bbox

            if self._ui_style == "COLOR":
                color = _COLOR_MAP.get(block.block_type, (180, 180, 180))
                thickness = 2
            else:
                color, thickness = (0, 0, 0), 1

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

            label = f"{block.block_type.upper()} {block.confidence:.0%}"
            font_scale = 0.5 if self._ui_style == "COLOR" else 0.4
            cv2.putText(
                canvas,
                label,
                (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas
