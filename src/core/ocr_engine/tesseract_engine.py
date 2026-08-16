"""Tesseract OCR engine implementation.

Uses pytesseract for optical character recognition with Czech language support.

Implementace Tesseract OCR enginu pro rozpoznávání českého textu.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from pdfminer.high_level import extract_text

from src.core.ocr_engine.base import BaseOCREngine, OCRResult

if TYPE_CHECKING:
    from PIL import Image

# Thread pool for CPU-bound OCR operations
_executor = ThreadPoolExecutor(max_workers=4)


def _import_dependencies():
    """Lazy import of heavy dependencies."""
    try:
        import pytesseract
        from PIL import Image
        return pytesseract, Image
    except ImportError as e:
        raise ImportError(
            "Tesseract dependencies not installed. "
            "Install with: pip install pytesseract pillow"
        ) from e


def _import_pdf2image():
    """Lazy import of pdf2image for PDF conversion."""
    try:
        from pdf2image import convert_from_bytes
        return convert_from_bytes
    except ImportError as e:
        raise ImportError(
            "pdf2image not installed. "
            "Install with: pip install pdf2image"
        ) from e


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR engine with Czech language support.

    Configuration:
        - Language: 'ces' (Czech utility invoices)
        - PSM mode: 4 (single column of text — best for utility invoices)
        - OEM mode: 3 (default, LSTM neural net)
        - Preprocessing: Gaussian unsharp mask + NLM denoising (OCR sweep winner)

    Requires:
        - Tesseract binary installed on system
        - Czech language pack (tesseract-ocr-ces)
        - pytesseract Python wrapper
    """

    def __init__(
        self,
        lang: str = "ces",
        config: str = "--psm 4 --oem 3",
        tesseract_cmd: str | None = None,
        layout_format: str = "json_light",
    ) -> None:
        """Initialize Tesseract engine with configuration.

        Args:
            lang: Tesseract language codes ('ces' for Czech utility invoices).
            config: Additional Tesseract configuration flags.
            tesseract_cmd: Path to Tesseract executable (optional).
            layout_format: Spatial-layout encoding for the recognised text —
                ``"json_light"`` (default, LIE-benchmark winner: token-efficient,
                stable across quality classes) or ``"markdown"`` (kept for A/B
                comparison). This is the primary (non-vision) extraction path's
                text serialisation.
        """
        self._lang = lang
        self._config = config
        self._tesseract_cmd = tesseract_cmd
        self._layout_format = layout_format
        # Configure once at init (not per-call)
        if tesseract_cmd:
            pytesseract, _ = _import_dependencies()
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    @property
    def name(self) -> str:
        """Return engine name identifier."""
        return "tesseract"

    @staticmethod
    def _preprocess_for_ocr(image: "Image.Image") -> "Image.Image":
        """Apply grayscale normalize + NLM denoising before OCR.

        OCR sweep tested 18 configs; Gaussian unsharp mask gave +10% raw OCR
        hit-rate but hurt end-to-end F1 on some invoice types (extra chars
        from sharpening added noise that confused the LLM). Baseline
        (normalize + NLM) gives the best overall end-to-end result.
        Falls back to the original image if cv2 is unavailable.
        """
        try:
            import cv2
            import numpy as np
            from PIL import Image as PILImage

            arr = np.array(image)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if len(arr.shape) == 3 else arr
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            denoised = cv2.fastNlMeansDenoising(
                normalized, None, h=10, templateWindowSize=7, searchWindowSize=21
            )
            return PILImage.fromarray(denoised)
        except ImportError:
            return image

    @staticmethod
    def _group_words_into_lines(
        data: dict,
        conf_min: int = 30,
        y_tol: int = 4,
    ) -> list[list[dict]]:
        """Group image_to_data words into spatially-ordered lines.

        Shared block/line grouping logic used by both the Markdown and the
        JSON-light serialisers so the two encodings see an identical layout.

        Returns a list of lines (top→bottom); each line is a list of word
        dicts (``text``/``left``/``top``/``width``/``height``) sorted left→right.

        y_tol=4 keeps adjacent table rows separate (vs Tesseract's block/line
        grouping which tends to merge rows in compact tables).
        """
        # Collect words with spatial info, skipping low-confidence / empty
        word_list: list[dict] = []
        for i in range(len(data["text"])):
            txt = str(data["text"][i]).strip()
            conf = data["conf"][i]
            if not txt or conf < conf_min:
                continue
            try:
                left = int(data["left"][i])
                top = int(data["top"][i])
                width = int(data["width"][i])
                height = int(data["height"][i])
            except (TypeError, ValueError):
                continue
            word_list.append({"text": txt, "left": left, "top": top,
                              "width": width, "height": max(height, 1)})

        if not word_list:
            return []

        word_list.sort(key=lambda w: (w["top"], w["left"]))

        # Group words into lines by Y-overlap
        lines: list[list[dict]] = []
        cur: list[dict] = []
        for w in word_list:
            if not cur:
                cur.append(w)
                continue
            lt = min(x["top"] for x in cur)
            lb = max(x["top"] + x["height"] for x in cur)
            if w["top"] + w["height"] < lt - y_tol or w["top"] > lb + y_tol:
                lines.append(sorted(cur, key=lambda x: x["left"]))
                cur = [w]
            else:
                cur.append(w)
        if cur:
            lines.append(sorted(cur, key=lambda x: x["left"]))

        return lines

    @staticmethod
    def _split_line_into_cells(line: list[dict], col_gap_px: int = 60) -> list[str]:
        """Split one line into column cells by horizontal gap.

        col_gap_px=60 is the minimum pixel gap that marks a column boundary.
        """
        cells: list[list[str]] = [[line[0]["text"]]]
        for i in range(1, len(line)):
            gap = line[i]["left"] - (line[i - 1]["left"] + line[i - 1]["width"])
            if gap >= col_gap_px:
                cells.append([line[i]["text"]])
            else:
                cells[-1].append(line[i]["text"])
        return [" ".join(c) for c in cells]

    @classmethod
    def _words_to_markdown(
        cls,
        data: dict,
        conf_min: int = 30,
        y_tol: int = 4,
        col_gap_px: int = 60,
    ) -> str:
        """Convert image_to_data output to Markdown text with table detection.

        Multi-column lines (gap ≥ col_gap_px between word groups) are emitted as
        Markdown pipe-table rows: | cell1 | cell2 | ...
        Single-column lines are emitted as plain text.

        Kept in place alongside :meth:`_words_to_json_light` so the two spatial
        encodings can be A/B compared later.
        """
        lines = cls._group_words_into_lines(data, conf_min=conf_min, y_tol=y_tol)
        if not lines:
            return ""

        # Emit Markdown: consecutive multi-column lines → pipe-table block
        parts: list[str] = []
        pending: list[str] = []

        def _flush():
            if pending:
                parts.append("\n".join(pending))
                pending.clear()

        for line in lines:
            cell_texts = cls._split_line_into_cells(line, col_gap_px)
            if len(cell_texts) >= 2:
                pending.append("| " + " | ".join(cell_texts) + " |")
            else:
                _flush()
                parts.append(cell_texts[0])

        _flush()
        return "\n".join(parts)

    @classmethod
    def _words_to_json_light(
        cls,
        data: dict,
        conf_min: int = 30,
        y_tol: int = 4,
        col_gap_px: int = 60,
    ) -> str:
        """Convert image_to_data output to the token-efficient JSON-light layout.

        Mirrors :meth:`_words_to_markdown`'s block/column grouping logic but
        serialises to the compact JSON structure validated as the LIE benchmark
        winner (token-efficient, stable accuracy across quality classes):

        - multi-column line  → ``"rN": ["col1", "col2", ...]``   (table row)
        - single "key: value" line → ``"sanitised_key": "value"``
        - other single-column line → ``"rN": "full line text"``

        N is a running index over non-empty lines. Emitted with no whitespace
        and ``ensure_ascii=False`` so Czech diacritics survive.
        """
        lines = cls._group_words_into_lines(data, conf_min=conf_min, y_tol=y_tol)
        if not lines:
            return ""

        out: dict[str, object] = {}
        row_idx = 0
        for line in lines:
            cells = cls._split_line_into_cells(line, col_gap_px)
            text = " ".join(w["text"] for w in line).strip()
            if not text:
                continue

            if len(cells) >= 2:
                # Multi-column / table-like line → list of column segments
                out[f"r{row_idx}"] = cells
            elif ":" in text:
                # Key-value pair → sanitised textual key (ASCII-safe, ≤30 chars)
                idx = text.index(":")
                k_raw = text[:idx].strip()
                v_raw = text[idx + 1:].strip()
                k_clean = re.sub(r"[^\w\s]", "", k_raw, flags=re.UNICODE)
                k_clean = re.sub(r"\s+", "_", k_clean.strip())[:30].lower()
                if k_clean and v_raw:
                    out[k_clean] = v_raw
                else:
                    out[f"r{row_idx}"] = text
            else:
                out[f"r{row_idx}"] = text

            row_idx += 1

        return json.dumps(out, ensure_ascii=False, separators=(",", ":"))

    def _run_ocr_sync(self, image: "Image.Image") -> tuple[str, float]:
        """Run OCR synchronously on a single image.

        Uses a single tesseract call (image_to_data) to get both text and
        confidence scores. The recognised words are serialised with the
        configured spatial-layout encoding (``self._layout_format``:
        JSON-light by default, Markdown optional) so the downstream LLM can
        parse tabular invoice data correctly.

        Returns:
            Tuple of (encoded_text, confidence_score).
        """
        pytesseract, _ = _import_dependencies()

        image = self._preprocess_for_ocr(image)

        data = pytesseract.image_to_data(
            image,
            lang=self._lang,
            config=self._config,
            output_type=pytesseract.Output.DICT,
        )

        confidences = [c for c in data["conf"] if c > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        if self._layout_format == "markdown":
            text = self._words_to_markdown(data, conf_min=30)
        else:
            text = self._words_to_json_light(data, conf_min=30)

        return text, avg_confidence / 100.0  # Normalize to 0-1

    async def recognize(self, image_bytes: bytes) -> OCRResult:
        """Run OCR on a single image.

        Args:
            image_bytes: Raw bytes of an image (PNG, JPEG, TIFF, etc.).

        Returns:
            OCRResult with extracted text and metrics.
        """
        start_time = time.perf_counter()

        try:
            _, Image = _import_dependencies()
            image = Image.open(io.BytesIO(image_bytes))

            # Run OCR in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            text, confidence = await loop.run_in_executor(
                _executor,
                self._run_ocr_sync,
                image,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            return OCRResult(
                full_text=text,
                pages=[text],
                confidence=confidence,
                engine_name=self.name,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return OCRResult(
                full_text="",
                pages=[],
                confidence=0.0,
                engine_name=self.name,
                latency_ms=latency_ms,
                error_message=str(e),
            )

    async def recognize_pdf(self, pdf_bytes: bytes, dpi: int = 200) -> OCRResult:
        """Run OCR on a multi-page PDF.

        Converts each page to image via fitz (PyMuPDF) at 200 DPI, applies
        Gaussian-unsharp preprocessing, then runs Tesseract.

        Args:
            pdf_bytes: Raw bytes of a PDF document.
            dpi: Rendering resolution (200 DPI is the OCR sweep optimum).

        Returns:
            OCRResult with per-page text and aggregate metrics.
        """
        start_time = time.perf_counter()

        try:
            loop = asyncio.get_running_loop()

            def _render_pages() -> list["Image.Image"]:
                """Render all pages via fitz, fall back to pdf2image if unavailable."""
                try:
                    import fitz
                    import numpy as np
                    from PIL import Image as PILImage

                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    imgs: list[PILImage.Image] = []
                    scale = dpi / 72.0
                    for i in range(doc.page_count):
                        pix = doc[i].get_pixmap(
                            matrix=fitz.Matrix(scale, scale),
                            colorspace=fitz.csRGB,
                        )
                        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3)
                        imgs.append(PILImage.fromarray(arr, "RGB"))
                    doc.close()
                    return imgs
                except ImportError:
                    convert = _import_pdf2image()
                    return convert(pdf_bytes, dpi=dpi)

            images = await loop.run_in_executor(_executor, _render_pages)

            # Process each page
            pages: list[str] = []
            confidences: list[float] = []

            for image in images:
                text, conf = await loop.run_in_executor(
                    _executor,
                    self._run_ocr_sync,
                    image,
                )
                pages.append(text)
                if conf > 0:
                    confidences.append(conf)

            full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            latency_ms = (time.perf_counter() - start_time) * 1000

            return OCRResult(
                full_text=full_text,
                pages=pages,
                confidence=avg_confidence,
                engine_name=self.name,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return OCRResult(
                full_text="",
                pages=[],
                confidence=0.0,
                engine_name=self.name,
                latency_ms=latency_ms,
                error_message=str(e),
            )

    async def extract_native_text(self, pdf_bytes: bytes) -> OCRResult:
        """Extract text from a native PDF (text layer extraction).

        Uses pdfminer instead of OCR for native PDFs.

        Args:
            pdf_bytes: Raw bytes of a native PDF.

        Returns:
            OCRResult with extracted text, engine_name='native'.
        """
        start_time = time.perf_counter()

        try:
            # Run pdfminer in thread pool
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                _executor,
                lambda: extract_text(io.BytesIO(pdf_bytes)),
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            return OCRResult(
                full_text=text.strip(),
                pages=[text.strip()],  # Single "page" for simplicity
                confidence=1.0,  # Native text is 100% accurate
                engine_name="native",
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return OCRResult(
                full_text="",
                pages=[],
                confidence=0.0,
                engine_name="native",
                latency_ms=latency_ms,
                error_message=str(e),
            )


def create_tesseract_engine(
    lang: str = "ces",
    config: str = "--psm 4 --oem 3",
    tesseract_cmd: str | None = None,
    layout_format: str = "json_light",
) -> TesseractEngine:
    """Factory function to create a TesseractEngine instance."""
    return TesseractEngine(
        lang=lang,
        config=config,
        tesseract_cmd=tesseract_cmd,
        layout_format=layout_format,
    )
