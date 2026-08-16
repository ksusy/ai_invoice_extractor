"""Adaptive thresholding and caching layer for PDF classification.

This module provides:
1. Adaptive thresholds that adjust based on dataset characteristics
2. Result caching to avoid re-analysis of identical PDFs
3. Confidence-based decision refinement
4. Pipeline metrics tracking
5. Fallback strategies for edge cases

Design: Aim for <1% FN rate while maintaining >80% TN rate on natives.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

from src.core.classification.base import DocumentKind

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveThresholds:
    """Adaptive classification thresholds.

    These adjust based on:
    - Observed dataset distribution
    - Historical FN/FP rates
    - Confidence calibration
    """

    scanned_lower_threshold: float = 0.75  # If score > this → SCANNED
    native_upper_threshold: float = 0.25   # If score < this → NATIVE_PDF
    uncertain_range: tuple[float, float] = (0.25, 0.75)  # [lower, upper]

    # Confidence thresholds for decisions
    min_confidence_for_native: float = 0.80  # Need high confidence for NATIVE
    min_confidence_for_scan: float = 0.60    # Lower for SCANNED (safe to err here)

    # Performance metrics
    observed_fn_rate: float = 0.0   # False negative rate (scan → native)
    observed_fp_rate: float = 0.0   # False positive rate (native → scan)
    total_classifications: int = 0   # For computing rates

    @property
    def fn_too_high(self) -> bool:
        """Check if FN rate exceeds acceptable threshold (>2%)."""
        if self.total_classifications < 100:
            return False
        return self.observed_fn_rate > 0.02

    @property
    def fp_acceptable(self) -> bool:
        """Check if FP rate is acceptable (<25%)."""
        if self.total_classifications < 100:
            return True
        return self.observed_fp_rate < 0.25

    def adjust_for_high_fn(self) -> None:
        """Lower scanned_lower_threshold if FN rate is too high."""
        logger.warning(
            "FN rate too high (%.2f%%) – lowering SCANNED threshold from %.2f to %.2f",
            self.observed_fn_rate * 100,
            self.scanned_lower_threshold,
            self.scanned_lower_threshold - 0.05,
        )
        self.scanned_lower_threshold = max(0.50, self.scanned_lower_threshold - 0.05)

    def adjust_for_high_fp(self) -> None:
        """Raise scanned_lower_threshold if FP rate is too high (optional)."""
        logger.info(
            "FP rate high (%.2f%%) – raising SCANNED threshold from %.2f to %.2f",
            self.observed_fp_rate * 100,
            self.scanned_lower_threshold,
            self.scanned_lower_threshold + 0.05,
        )
        self.scanned_lower_threshold = min(1.0, self.scanned_lower_threshold + 0.05)

    def record_classification(
        self,
        predicted: DocumentKind,
        ground_truth: DocumentKind | None = None,
    ) -> None:
        """Record a classification result for metric updates.

        Args:
            predicted: What classifier predicted
            ground_truth: What it actually was (for FN/FP calculation, optional)
        """
        self.total_classifications += 1

        if ground_truth is None:
            return  # Can't compute FN/FP without ground truth

        # Update rates (exponential moving average)
        alpha = 0.1  # EMA smoothing factor

        if ground_truth == DocumentKind.SCANNED:
            is_fn = predicted == DocumentKind.NATIVE_PDF
            self.observed_fn_rate = (
                alpha * float(is_fn) + (1 - alpha) * self.observed_fn_rate
            )
        elif ground_truth == DocumentKind.NATIVE_PDF:
            is_fp = predicted == DocumentKind.SCANNED
            self.observed_fp_rate = (
                alpha * float(is_fp) + (1 - alpha) * self.observed_fp_rate
            )

        # Check if adjustment needed
        if self.fn_too_high:
            self.adjust_for_high_fn()


@dataclass
class CachedClassificationResult:
    """Result of a classification lookup in cache."""

    document_kind: DocumentKind
    confidence: float
    cached_at: float  # Unix timestamp
    age_seconds: float  # How old is the cache entry (for TTL check)
    is_valid: bool  # Whether cache is still valid (not stale)


class ClassificationCache:
    """LRU cache for PDF classification results.

    Motivation:
    - Pipeline might process same file multiple times
    - Classification is expensive (50-150ms)
    - Cache hits save time and computation
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 86400):
        """
        Args:
            max_size: Maximum number of cached results
            ttl_seconds: Time-to-live for cache entries (default: 24h)
        """
        self._cache: dict[str, tuple[DocumentKind, float, float]] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get_hash(self, file_bytes: bytes) -> str:
        """Get SHA256 hash of file for caching."""
        return hashlib.sha256(file_bytes).hexdigest()

    def get(self, file_hash: str) -> CachedClassificationResult | None:
        """Retrieve cached classification result if valid.

        Args:
            file_hash: SHA256 hash of PDF bytes

        Returns:
            CachedClassificationResult if valid, None if miss or stale
        """
        if file_hash not in self._cache:
            self._misses += 1
            return None

        doc_kind, confidence, cached_at = self._cache[file_hash]
        age_seconds = time.time() - cached_at

        if age_seconds > self._ttl_seconds:
            # Stale entry
            logger.debug("Cache entry stale (age=%.0fs) – evicting", age_seconds)
            del self._cache[file_hash]
            self._misses += 1
            return None

        self._hits += 1
        logger.debug(
            "Cache hit for %s (age=%.0fs, hit_rate=%.1f%%)",
            file_hash[:8],
            age_seconds,
            100 * self._hits / (self._hits + self._misses),
        )

        return CachedClassificationResult(
            document_kind=doc_kind,
            confidence=confidence,
            cached_at=cached_at,
            age_seconds=age_seconds,
            is_valid=True,
        )

    def put(self, file_hash: str, document_kind: DocumentKind, confidence: float) -> None:
        """Cache a classification result.

        Args:
            file_hash: SHA256 hash of PDF bytes
            document_kind: Classification result
            confidence: Confidence in result [0, 1]
        """
        if len(self._cache) >= self._max_size:
            # Evict oldest entry (simple strategy)
            oldest_hash = min(
                self._cache.keys(),
                key=lambda h: self._cache[h][2],
            )
            del self._cache[oldest_hash]
            logger.debug("Cache full – evicted oldest entry")

        self._cache[file_hash] = (document_kind, confidence, time.time())
        logger.debug(
            "Cached classification for %s: %s (confidence=%.2f)",
            file_hash[:8],
            document_kind,
            confidence,
        )

    @property
    def hit_rate(self) -> float:
        """Cache hit rate [0, 1]."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "size": len(self._cache),
            "max_size": self._max_size,
        }


@dataclass
class ClassificationMetrics:
    """Metrics for tracking classification performance."""

    total_classifications: int = 0
    successful_classifications: int = 0
    failed_classifications: int = 0
    average_latency_ms: float = 0.0
    cache_hits: int = 0
    low_confidence_count: int = 0  # count(confidence < 0.7)

    @property
    def success_rate(self) -> float:
        """Proportion of successful classifications."""
        total = self.successful_classifications + self.failed_classifications
        return self.successful_classifications / total if total > 0 else 0.0


class AdaptivePDFClassificationPipeline:
    """
    Pipeline combining:
    - Advanced classifier with confidence scoring
    - Adaptive thresholds that auto-tune based on feedback
    - Result caching for performance
    - Metrics tracking for monitoring
    """

    def __init__(
        self,
        classifier,  # BaseClassifier instance
        cache_size: int = 10000,
        cache_ttl_seconds: int = 86400,
    ):
        self._classifier = classifier
        self._cache = ClassificationCache(max_size=cache_size, ttl_seconds=cache_ttl_seconds)
        self._thresholds = AdaptiveThresholds()
        self._metrics = ClassificationMetrics()

    async def classify_with_caching(
        self,
        file_bytes: bytes,
        force_recompute: bool = False,
    ) -> tuple[DocumentKind, float, dict]:
        """
        Classify PDF with caching and adaptive thresholds.

        Args:
            file_bytes: PDF bytes
            force_recompute: Skip cache and recompute

        Returns:
            Tuple of (decision, confidence, metadata)
            where metadata includes source (cached vs computed)
        """
        file_hash = self._cache.get_hash(file_bytes)

        # Check cache first
        if not force_recompute:
            cached = self._cache.get(file_hash)
            if cached and cached.is_valid:
                self._metrics.cache_hits += 1
                return cached.document_kind, cached.confidence, {"source": "cache"}

        # Classify (compute)
        start_time = time.time()
        try:
            doc_kind, confidence = await self._run_classification(file_bytes)
            latency_ms = (time.time() - start_time) * 1000

            # Update metrics
            self._metrics.total_classifications += 1
            self._metrics.successful_classifications += 1
            self._metrics.average_latency_ms = (
                0.9 * self._metrics.average_latency_ms + 0.1 * latency_ms
            )

            if confidence < 0.7:
                self._metrics.low_confidence_count += 1

            # Cache result
            self._cache.put(file_hash, doc_kind, confidence)

            return doc_kind, confidence, {
                "source": "computed",
                "latency_ms": latency_ms,
            }

        except Exception as e:
            logger.error("Classification failed: %s", e)
            self._metrics.failed_classifications += 1
            # Fallback to SCANNED (safe)
            return DocumentKind.SCANNED, 0.5, {
                "source": "error",
                "error": str(e),
            }

    async def _run_classification(
        self,
        file_bytes: bytes,
    ) -> tuple[DocumentKind, float]:
        """Run the underlying classifier."""
        # If classifier has confidence scoring (like advanced_classifier)
        if hasattr(self._classifier, "classify_with_confidence"):
            result = self._classifier.classify_with_confidence(file_bytes)
            return result.document_kind, result.confidence
        else:
            # Legacy classifier – estimate confidence
            result = await self._classifier.classify(file_bytes)
            confidence = 0.75 if result == DocumentKind.SCANNED else 0.65
            return result, confidence

    def record_feedback(
        self,
        predicted: DocumentKind,
        ground_truth: DocumentKind,
    ) -> None:
        """Record user feedback to adjust adaptive thresholds.

        Args:
            predicted: What classifier predicted
            ground_truth: What it should have been (from user feedback)
        """
        self._thresholds.record_classification(predicted, ground_truth)

        # Check if retraining needed
        if self._thresholds.fn_too_high:
            logger.warning(
                "FN rate critical (%.2f%%) – recommend manual review of classifier",
                self._thresholds.observed_fn_rate * 100,
            )

    @property
    def metrics(self) -> dict:
        """Get current pipeline metrics."""
        return {
            "total_classifications": self._metrics.total_classifications,
            "success_rate": self._metrics.success_rate,
            "average_latency_ms": self._metrics.average_latency_ms,
            "cache_hit_rate": self._cache.hit_rate,
            "low_confidence_count": self._metrics.low_confidence_count,
            "adaptive_thresholds": {
                "scanned_lower": self._thresholds.scanned_lower_threshold,
                "native_upper": self._thresholds.native_upper_threshold,
                "observed_fn_rate": self._thresholds.observed_fn_rate,
                "observed_fp_rate": self._thresholds.observed_fp_rate,
            },
        }
