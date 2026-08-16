"""Table detection and structure extraction from document images.

Detects table regions by finding intersecting horizontal/vertical lines
(or dense grid-like patterns of bounding boxes) and reconstructs the
cell grid as an HTML table string for LLM consumption.

Two detection modes
───────────────────
1. **Line-based** (``LineTableDetector``) — uses morphological operations
   to find horizontal + vertical ruling lines, then intersects them to
   locate cell boundaries.  Works well for ruled-table invoices.

2. **Bbox-grid** (``BboxGridDetector``) — groups EasyOCR bounding boxes
   into a grid by clustering Y-centres (rows) and X-centres (columns).
   Works for borderless tables where columns are aligned.

Both detectors return a list of :class:`TableRegion` objects that carry
the detected bounding box and the reconstructed HTML.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Data containers ──────────────────────────────────────────────────────────

@dataclass
class TableCell:
    """One cell inside a detected table.

    Attributes:
        row:    0-based row index.
        col:    0-based column index.
        text:   OCR text content (may be empty).
        bbox:   Pixel bbox ``(x1, y1, x2, y2)`` within the full image.
    """
    row: int
    col: int
    text: str
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class TableRegion:
    """A detected table within the document image.

    Attributes:
        bbox:   Pixel bbox of the table ``(x1, y1, x2, y2)``.
        cells:  List of :class:`TableCell` objects.
        html:   Reconstructed HTML ``<table>`` string.
        n_rows: Number of rows detected.
        n_cols: Number of columns detected.
        source: Detection mode: ``"lines"`` or ``"bbox_grid"``.
    """
    bbox: tuple[int, int, int, int]
    cells: list[TableCell] = field(default_factory=list)
    html: str = ""
    n_rows: int = 0
    n_cols: int = 0
    source: str = "lines"

    @classmethod
    def from_cells(
        cls,
        cells: list[TableCell],
        bbox: tuple[int, int, int, int],
        source: str = "lines",
    ) -> TableRegion:
        """Build a TableRegion from a cell list and generate HTML."""
        if not cells:
            return cls(bbox=bbox, source=source)

        n_rows = max(c.row for c in cells) + 1
        n_cols = max(c.col for c in cells) + 1

        # Build grid (row × col) → text
        grid: list[list[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        for cell in cells:
            grid[cell.row][cell.col] = cell.text.strip()

        # Render HTML
        rows_html: list[str] = []
        for row_idx, row in enumerate(grid):
            tag = "th" if row_idx == 0 else "td"
            cells_html = "".join(f"<{tag}>{v}</{tag}>" for v in row)
            rows_html.append(f"<tr>{cells_html}</tr>")

        html = "<table>" + "".join(rows_html) + "</table>"
        return cls(bbox=bbox, cells=cells, html=html,
                   n_rows=n_rows, n_cols=n_cols, source=source)


# ── Line-based table detector ────────────────────────────────────────────────

class LineTableDetector:
    """Detect tables by finding intersecting ruling lines via morphology.

    This is the standard approach for ruled-border tables common in
    Czech energy invoices (ČEZ, E.ON, etc.).

    Args:
        min_line_len_ratio: Minimum line length as a fraction of image
                            width/height.  Shorter segments are noise.
        line_thickness:     Dilation size for line-kernel morphology.
        min_cell_area:      Minimum cell area in pixels (filters small noise cells).
    """

    def __init__(
        self,
        min_line_len_ratio: float = 0.25,
        line_thickness: int = 3,
        min_cell_area: int = 400,
    ) -> None:
        self.min_line_len_ratio = min_line_len_ratio
        self.line_thickness     = line_thickness
        self.min_cell_area      = min_cell_area

    def detect(
        self,
        image: np.ndarray,
        ocr_boxes: list[tuple[list, str, float]] | None = None,
    ) -> list[TableRegion]:
        """Detect table regions in a BGR image.

        Args:
            image:     Full-page BGR image.
            ocr_boxes: EasyOCR output (detail=1) for text assignment to cells.
                       Each entry: (bbox_quad, text, confidence).

        Returns:
            List of :class:`TableRegion` objects (may be empty).
        """
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, w = bw.shape
        h_len = max(1, int(w * self.min_line_len_ratio))
        v_len = max(1, int(h * self.min_line_len_ratio))

        # Extract horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, self.line_thickness))
        h_lines  = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)

        # Extract vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.line_thickness, v_len))
        v_lines  = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

        # Intersect to find grid structure
        grid_mask = cv2.add(h_lines, v_lines)
        grid_mask = cv2.dilate(
            grid_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=2,
        )

        # Find table bounding rectangles
        contours, _ = cv2.findContours(
            grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions: list[TableRegion] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw * ch < self.min_cell_area * 4:
                continue  # Too small to be a meaningful table

            table_bbox = (x, y, x + cw, y + ch)
            cells = self._extract_cells(
                bw[y:y+ch, x:x+cw],
                offset=(x, y),
                ocr_boxes=ocr_boxes or [],
            )
            if cells:
                region = TableRegion.from_cells(cells, table_bbox, source="lines")
                regions.append(region)
                logger.debug(
                    "Table detected (lines): %dx%d at %s",
                    region.n_rows, region.n_cols, table_bbox,
                )

        return regions

    def _extract_cells(
        self,
        table_bw: np.ndarray,
        offset: tuple[int, int],
        ocr_boxes: list[tuple[list, str, float]],
    ) -> list[TableCell]:
        """Find individual cells within a table region via contour detection."""
        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(1, table_bw.shape[1] // 20), self.line_thickness)
        )
        v_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.line_thickness, max(1, table_bw.shape[0] // 20))
        )
        h_lines = cv2.morphologyEx(table_bw, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(table_bw, cv2.MORPH_OPEN, v_kernel)
        grid    = cv2.add(h_lines, v_lines)

        # Invert and find cell contours (cells are white space between lines)
        cell_mask = cv2.bitwise_not(grid)
        contours, _ = cv2.findContours(
            cell_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        cell_bboxes: list[tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw * ch >= self.min_cell_area:
                # Convert back to full-image coordinates
                cell_bboxes.append((
                    x + offset[0], y + offset[1],
                    x + offset[0] + cw, y + offset[1] + ch,
                ))

        if not cell_bboxes:
            return []

        return _assign_ocr_to_grid(cell_bboxes, ocr_boxes)


# ── Bbox-grid detector ───────────────────────────────────────────────────────

class BboxGridDetector:
    """Detect borderless tables by clustering OCR bounding boxes into a grid.

    When invoices have no ruling lines, column-aligned text can still form
    implicit tables.  This detector groups boxes by Y-centre (rows) and
    X-centre (columns) and treats dense, regular clusters as tables.

    Args:
        row_gap_px:  Maximum vertical gap between boxes in the same row.
        col_gap_px:  Maximum horizontal gap between boxes in the same column.
        min_cols:    Minimum number of columns to qualify as a table.
        min_rows:    Minimum number of rows to qualify as a table.
    """

    def __init__(
        self,
        row_gap_px: int = 12,
        col_gap_px: int = 40,
        min_cols: int = 2,
        min_rows: int = 2,
    ) -> None:
        self.row_gap_px = row_gap_px
        self.col_gap_px = col_gap_px
        self.min_cols   = min_cols
        self.min_rows   = min_rows

    def detect(
        self,
        ocr_boxes: list[tuple[list, str, float]],
    ) -> list[TableRegion]:
        """Detect implicit table regions from OCR bounding boxes.

        Args:
            ocr_boxes: EasyOCR output (detail=1).

        Returns:
            List of :class:`TableRegion` objects.
        """
        if not ocr_boxes:
            return []

        # Convert quad bboxes to axis-aligned rects with centre points
        rects: list[tuple[int, int, int, int, str]] = []
        for quad, text, _ in ocr_boxes:
            pts = np.array(quad, dtype=np.float32)
            x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
            rects.append((x1, y1, x2, y2, text))

        # Group into rows by Y-centre
        rows = _cluster_by_axis(
            [(r[1] + r[3]) // 2 for r in rects],
            gap=self.row_gap_px,
        )

        # For each row-cluster, check column regularity
        row_groups: list[list[tuple[int, int, int, int, str]]] = []
        for cluster_indices in rows:
            row_groups.append([rects[i] for i in cluster_indices])

        if len(row_groups) < self.min_rows:
            return []

        # Group columns by X-centre across all rows
        all_x_centres = []
        for rg in row_groups:
            for r in rg:
                all_x_centres.append((r[0] + r[2]) // 2)

        col_clusters = _cluster_by_axis(all_x_centres, gap=self.col_gap_px)
        if len(col_clusters) < self.min_cols:
            return []

        col_centres = sorted(
            np.mean([all_x_centres[i] for i in cl])
            for cl in col_clusters
        )

        # Build cell list
        all_cells: list[TableCell] = []
        for row_idx, rg in enumerate(row_groups):
            for x1, y1, x2, y2, text in rg:
                xc = (x1 + x2) // 2
                col_idx = int(np.argmin([abs(xc - cc) for cc in col_centres]))
                all_cells.append(TableCell(
                    row=row_idx, col=col_idx, text=text, bbox=(x1, y1, x2, y2)
                ))

        if not all_cells:
            return []

        # Compute overall bounding box
        xs = [c.bbox[0] for c in all_cells] + [c.bbox[2] for c in all_cells]
        ys = [c.bbox[1] for c in all_cells] + [c.bbox[3] for c in all_cells]
        table_bbox = (min(xs), min(ys), max(xs), max(ys))

        region = TableRegion.from_cells(all_cells, table_bbox, source="bbox_grid")
        logger.debug(
            "Table detected (bbox_grid): %dx%d",
            region.n_rows, region.n_cols,
        )
        return [region]


# ── Shared utility: TableExtractor facade ────────────────────────────────────

class TableExtractor:
    """Combined table detector: tries line-based first, falls back to bbox-grid.

    Args:
        prefer_lines:   If True, run :class:`LineTableDetector` first and
                        only run :class:`BboxGridDetector` when no line-based
                        tables are found.
        line_kwargs:    Forwarded to :class:`LineTableDetector`.
        grid_kwargs:    Forwarded to :class:`BboxGridDetector`.
    """

    def __init__(
        self,
        prefer_lines: bool = True,
        line_kwargs: dict | None = None,
        grid_kwargs: dict | None = None,
    ) -> None:
        self._line = LineTableDetector(**(line_kwargs or {}))
        self._grid = BboxGridDetector(**(grid_kwargs or {}))
        self._prefer_lines = prefer_lines

    def extract(
        self,
        image: np.ndarray,
        ocr_boxes: list[tuple[list, str, float]],
    ) -> list[TableRegion]:
        """Run both detectors and return merged results."""
        regions: list[TableRegion] = []

        if self._prefer_lines:
            line_regions = self._line.detect(image, ocr_boxes)
            if line_regions:
                regions.extend(line_regions)
            else:
                regions.extend(self._grid.detect(ocr_boxes))
        else:
            regions.extend(self._line.detect(image, ocr_boxes))
            regions.extend(self._grid.detect(ocr_boxes))

        return regions


# ── Shared utilities ─────────────────────────────────────────────────────────

def _cluster_by_axis(values: list[int | float], gap: int) -> list[list[int]]:
    """Cluster 1-D values by maximum gap distance.

    Returns a list of index-lists, each representing one cluster.
    """
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    clusters: list[list[int]] = [[indexed[0][0]]]

    for idx, val in indexed[1:]:
        if val - values[clusters[-1][-1]] <= gap:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return clusters


def _assign_ocr_to_grid(
    cell_bboxes: list[tuple[int, int, int, int]],
    ocr_boxes: list[tuple[list, str, float]],
) -> list[TableCell]:
    """Assign OCR text snippets to the nearest cell bbox.

    Returns :class:`TableCell` objects with ``(row, col)`` indices based
    on the sorted bounding-box positions.
    """
    if not cell_bboxes:
        return []

    # Sort cells: top-to-bottom, left-to-right
    sorted_cells = sorted(cell_bboxes, key=lambda b: (b[1], b[0]))

    # Assign row/col indices by Y-centre clustering
    y_centres = [(b[1] + b[3]) // 2 for b in sorted_cells]
    row_clusters = _cluster_by_axis(y_centres, gap=10)
    row_map: dict[int, int] = {}
    for row_idx, cluster in enumerate(row_clusters):
        for i in cluster:
            row_map[i] = row_idx

    x_centres = [(b[0] + b[2]) // 2 for b in sorted_cells]
    col_clusters = _cluster_by_axis(x_centres, gap=20)
    col_map: dict[int, int] = {}
    for col_idx, cluster in enumerate(col_clusters):
        for i in cluster:
            col_map[i] = col_idx

    # For each cell bbox, find overlapping OCR boxes
    def _overlap(cb, qb):
        """Check if OCR quad centre falls inside cell bbox."""
        pts = np.array(qb, dtype=np.float32)
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        return cb[0] <= cx <= cb[2] and cb[1] <= cy <= cb[3]

    cells: list[TableCell] = []
    for cell_idx, cb in enumerate(sorted_cells):
        texts = [
            text for quad, text, _ in ocr_boxes
            if _overlap(cb, quad)
        ]
        cells.append(TableCell(
            row=row_map.get(cell_idx, 0),
            col=col_map.get(cell_idx, 0),
            text=" ".join(texts),
            bbox=cb,
        ))
    return cells
