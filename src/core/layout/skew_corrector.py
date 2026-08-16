"""Skew detection and correction for scanned document images.

Uses the Hough line transform to detect the dominant text angle and
rotates the image to horizontal alignment.

Typical usage::

    from src.core.layout.skew_corrector import SkewCorrector

    corrector = SkewCorrector(max_skew_deg=10.0)
    corrected, angle = corrector.correct(bgr_image)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SkewResult:
    """Output of :meth:`SkewCorrector.correct`.

    Attributes:
        image:      Corrected BGR image (or original if skew below threshold).
        angle_deg:  Detected skew angle in degrees (positive = clockwise tilt).
        corrected:  True if the image was actually rotated.
    """
    image: np.ndarray
    angle_deg: float
    corrected: bool


class SkewCorrector:
    """Detect and correct document skew using Hough line transform.

    Args:
        max_skew_deg:    Maximum skew to correct (degrees). Images with larger
                         estimated angles are left unchanged (likely portrait
                         layout or double-page scan).
        min_skew_deg:    Minimum angle to trigger correction. Tiny drifts
                         (< 0.3°) are ignored to avoid unnecessary resampling.
        canny_low:       Lower threshold for Canny edge detection.
        canny_high:      Upper threshold for Canny edge detection.
        hough_threshold: Minimum number of votes for a Hough line.
        dilation_iter:   Dilation iterations applied before Hough to thicken
                         text strokes (improves line detection on thin fonts).
    """

    def __init__(
        self,
        max_skew_deg: float = 10.0,
        min_skew_deg: float = 0.3,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 100,
        dilation_iter: int = 1,
    ) -> None:
        self.max_skew_deg   = max_skew_deg
        self.min_skew_deg   = min_skew_deg
        self.canny_low      = canny_low
        self.canny_high     = canny_high
        self.hough_threshold = hough_threshold
        self.dilation_iter  = dilation_iter

    # ── Public API ───────────────────────────────────────────────────────────

    def correct(self, image: np.ndarray) -> SkewResult:
        """Detect skew and return a (possibly rotated) image.

        Args:
            image: BGR NumPy array.

        Returns:
            :class:`SkewResult` with corrected image and angle metadata.
        """
        angle = self._detect_angle(image)

        if angle is None or abs(angle) < self.min_skew_deg:
            return SkewResult(image=image, angle_deg=angle or 0.0, corrected=False)

        if abs(angle) > self.max_skew_deg:
            logger.info(
                "Skew angle %.1f° exceeds max %.1f° — skipping correction",
                angle, self.max_skew_deg,
            )
            return SkewResult(image=image, angle_deg=angle, corrected=False)

        rotated = self._rotate(image, -angle)
        logger.debug("Skew corrected by %.2f°", angle)
        return SkewResult(image=rotated, angle_deg=angle, corrected=True)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _detect_angle(self, image: np.ndarray) -> float | None:
        """Estimate skew angle via probabilistic Hough on Canny edges."""
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # Dilate slightly to connect broken stroke segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges  = cv2.dilate(edges, kernel, iterations=self.dilation_iter)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=image.shape[1] // 4,
            maxLineGap=20,
        )

        if lines is None or len(lines) == 0:
            return None

        angles: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)
                # Keep only near-horizontal lines (−45° … +45°)
                if -45 < angle_deg < 45:
                    angles.append(angle_deg)

        if not angles:
            return None

        # Use median to suppress outliers
        return float(np.median(angles))

    @staticmethod
    def _rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate image around its centre, filling borders with white."""
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return rotated
