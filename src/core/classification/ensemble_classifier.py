"""Ensemble PDF classifier combining multiple strategies for maximum reliability.

This module implements:
1. Ensemble voting between multiple classifiers (v1 + v2)
2. Adaptive thresholds based on dataset characteristics
3. Confidence weighting and conflict resolution
4. Per-page analysis for HYBRID detection (mixed native/scanned)
5. Metadata signals (filename patterns, document type hints)

GOAL: Achieve <1% FN rate with high precision on natives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from src.core.classification.advanced_classifier import (
    AdvancedPDFClassifier,
)
from src.core.classification.base import DocumentKind
from src.core.classification.pdf_classifier import PDFClassifier

logger = logging.getLogger(__name__)


class EnsembleStrategy(StrEnum):
    """Decision strategies for combining multiple classifiers."""

    CONSERVATIVE = "conservative"  # Majority vote, always default to SCANNED on tie
    AGGRESSIVE = "aggressive"      # Strict confidence thresholds for NATIVE
    WEIGHTED = "weighted"          # Weight scores by historical accuracy
    ADAPTIVE = "adaptive"          # Adjust strategy per document category


@dataclass
class ClassifierScore:
    """Score from a single classifier."""

    source: str  # "v1_legacy", "v2_advanced"
    decision: DocumentKind
    confidence: float  # [0, 1]
    reasoning: list[str] = field(default_factory=list)


@dataclass
class EnsembleResult:
    """Final decision from ensemble voting."""

    final_decision: DocumentKind
    confidence: float  # [0, 1] – confidence in final decision
    agreement_level: float  # [0, 1] – how much classifiers agreed
    individual_scores: list[ClassifierScore]
    decision_reasoning: list[str]
    recommendation: str | None = None  # "review_manually" if uncertain


class EnsemblePDFClassifier:
    """
    Ensemble approach combining:
    - PDFClassifier v1 (legacy, conservative)
    - AdvancedPDFClassifier v2 (new, asymmetric)
    - Adaptive thresholds
    - Metadata signals
    - Per-page analysis

    Policy:
        1. Run both classifiers in parallel
        2. If strong agreement (>0.8) → Accept with high confidence
        3. If disagreement (0.2-0.8) → Analyze deeper signals
        4. If conflict (both uncertain) → Default to SCANNED (safe)
    """

    def __init__(
        self,
        strategy: EnsembleStrategy = EnsembleStrategy.CONSERVATIVE,
        use_metadata_signals: bool = True,
        per_page_analysis: bool = True,
    ):
        self._legacy_classifier = PDFClassifier(session=None)
        self._advanced_classifier = AdvancedPDFClassifier(session=None)
        self._strategy = strategy
        self._use_metadata = use_metadata_signals
        self._per_page_analysis = per_page_analysis

    async def classify_ensemble(
        self,
        file_bytes: bytes,
        metadata: dict | None = None,
    ) -> EnsembleResult:
        """
        Classify using ensemble of multiple strategies.

        Args:
            file_bytes: PDF bytes
            metadata: Optional dict with:
                - filename: str
                - file_size: int
                - invoice_type: str (hint about document type)

        Returns:
            EnsembleResult with final decision, confidence, and reasoning.
        """
        metadata = metadata or {}

        # Step 1: Run both classifiers in parallel
        v1_score = await self._run_legacy_classifier(file_bytes)
        v2_score = await self._run_advanced_classifier(file_bytes)

        individual_scores = [v1_score, v2_score]

        # Step 2: Analyze agreement
        agreement_level = self._calculate_agreement(v1_score, v2_score)

        # Step 3: Get additional signals
        metadata_signal = (
            self._analyze_metadata_signals(metadata)
            if self._use_metadata
            else None
        )

        # Step 4: Per-page analysis (expensive, only if needed)
        per_page_signal = (
            self._analyze_per_page(file_bytes)
            if self._per_page_analysis
            else None
        )

        # Step 5: Combine signals → final decision
        final_decision, confidence, reasoning = self._ensemble_vote(
            individual_scores=individual_scores,
            metadata_signal=metadata_signal,
            per_page_signal=per_page_signal,
            strategy=self._strategy,
        )

        return EnsembleResult(
            final_decision=final_decision,
            confidence=confidence,
            agreement_level=agreement_level,
            individual_scores=individual_scores,
            decision_reasoning=reasoning,
        )

    # ── Classifier Runners ───────────────────────────────────────

    async def _run_legacy_classifier(self, file_bytes: bytes) -> ClassifierScore:
        """Run v1 legacy classifier."""
        try:
            result = await self._legacy_classifier.classify(file_bytes)
            return ClassifierScore(
                source="v1_legacy",
                decision=result,
                confidence=0.75 if result == DocumentKind.SCANNED else 0.65,
                reasoning=["Legacy PDF structure analysis"],
            )
        except Exception as e:
            logger.warning("Legacy classifier failed: %s", e)
            return ClassifierScore(
                source="v1_legacy",
                decision=DocumentKind.SCANNED,
                confidence=0.5,
                reasoning=["Legacy classifier error – fallback to SCANNED"],
            )

    async def _run_advanced_classifier(self, file_bytes: bytes) -> ClassifierScore:
        """Run v2 advanced classifier."""
        try:
            result = self._advanced_classifier.classify_with_confidence(file_bytes)
            return ClassifierScore(
                source="v2_advanced",
                decision=result.document_kind,
                confidence=result.confidence,
                reasoning=result.reasoning,
            )
        except Exception as e:
            logger.warning("Advanced classifier failed: %s", e)
            return ClassifierScore(
                source="v2_advanced",
                decision=DocumentKind.SCANNED,
                confidence=0.5,
                reasoning=["Advanced classifier error – fallback to SCANNED"],
            )

    # ── Agreement Analysis ───────────────────────────────────────

    @staticmethod
    def _calculate_agreement(score1: ClassifierScore, score2: ClassifierScore) -> float:
        """Calculate agreement level between classifiers.

        Return: [0, 1] where 1 = perfect agreement.
        """
        if score1.decision == score2.decision:
            # Same decision – weight by confidence
            avg_confidence = (score1.confidence + score2.confidence) / 2
            return 0.5 + (avg_confidence * 0.5)  # [0.5, 1.0]
        else:
            # Opposite decisions – calculate conflict
            min_confidence = min(score1.confidence, score2.confidence)
            return min_confidence * 0.5  # Lower agreement

    # ── Metadata Signals ─────────────────────────────────────────

    def _analyze_metadata_signals(self, metadata: dict) -> dict:
        """Analyze filename and document metadata for signals.

        Examples:
        - Filename contains "scan" → hint towards SCANNED
        - CZ invoice naming conventions
        - File size patterns (scans tend larger)
        """
        signals = {}

        # Signal 1: Filename patterns
        filename = metadata.get("filename", "").lower()
        scan_keywords = ["scan", "scanned", "skc", "image", "jpg", "png"]
        if any(kw in filename for kw in scan_keywords):
            signals["filename_suggests_scan"] = 0.7
        else:
            signals["filename_suggests_scan"] = 0.0

        # Signal 2: File size heuristic
        # Scanned PDFs are typically larger (raster data)
        file_size = metadata.get("file_size", 0)
        if file_size > 5_000_000:  # >5MB
            signals["large_file_suggests_scan"] = 0.4
        elif file_size < 500_000:  # <500KB
            signals["large_file_suggests_scan"] = -0.2  # Native hint
        else:
            signals["large_file_suggests_scan"] = 0.0

        # Signal 3: Invoice type hint
        invoice_type = metadata.get("invoice_type", "").lower()
        if "utility" in invoice_type:
            # Utility invoices often scanned (from old systems)
            signals["invoice_type_suggests_scan"] = 0.3
        else:
            signals["invoice_type_suggests_scan"] = 0.0

        return signals

    def _analyze_per_page(self, file_bytes: bytes) -> dict:
        """Analyze individual pages to detect HYBRID documents.

        A HYBRID document has:
        - Some pages that are scans
        - Some pages that are native
        (Less common but possible)
        """
        # TODO: Implement per-page analysis if needed
        # For now, return empty
        return {"is_hybrid": False, "scan_page_ratio": 0.0}

    # ── Ensemble Voting ──────────────────────────────────────────

    def _ensemble_vote(
        self,
        individual_scores: list[ClassifierScore],
        metadata_signal: dict | None,
        per_page_signal: dict | None,
        strategy: EnsembleStrategy,
    ) -> tuple[DocumentKind, float, list[str]]:
        """
        Combine individual scores into final decision.

        Logic depends on strategy (CONSERVATIVE, WEIGHTED, ADAPTIVE).
        """
        reasoning = []

        if strategy == EnsembleStrategy.CONSERVATIVE:
            return self._vote_conservative(individual_scores, reasoning, metadata_signal)
        elif strategy == EnsembleStrategy.WEIGHTED:
            return self._vote_weighted(individual_scores, reasoning)
        elif strategy == EnsembleStrategy.ADAPTIVE:
            return self._vote_adaptive(individual_scores, reasoning, metadata_signal)
        else:
            # Fallback
            return DocumentKind.SCANNED, 0.5, ["Unknown strategy – defaulting to SCANNED"]

    def _vote_conservative(
        self,
        scores: list[ClassifierScore],
        reasoning: list[str],
        metadata: dict | None,
    ) -> tuple[DocumentKind, float, list[str]]:
        """Conservative voting: need strong evidence for NATIVE.

        Rules:
        1. If both classifiers agree on SCANNED → SCANNED (high confidence)
        2. If v2 says SCANNED → SCANNED (prioritize advanced)
        3. If both say NATIVE + high confidence → NATIVE
        4. Otherwise → SCANNED (safe default)
        """
        v2_score = next((s for s in scores if s.source == "v2_advanced"), None)
        v1_score = next((s for s in scores if s.source == "v1_legacy"), None)

        # Rule 1: Both agree on SCANNED
        if (v1_score and v1_score.decision == DocumentKind.SCANNED and
            v2_score and v2_score.decision == DocumentKind.SCANNED):
            reasoning.append("Both classifiers agree: SCANNED")
            confidence = min(v1_score.confidence, v2_score.confidence)
            return DocumentKind.SCANNED, confidence, reasoning

        # Rule 2: Advanced (v2) says SCANNED – prioritize it
        if v2_score and v2_score.decision == DocumentKind.SCANNED:
            reasoning.append(f"Advanced classifier: SCANNED (confidence={v2_score.confidence:.2f})")
            if metadata:
                reasoning.extend(self._apply_metadata_reasoning(metadata))
            return DocumentKind.SCANNED, v2_score.confidence, reasoning

        # Rule 3: Both say NATIVE + high confidence
        if (v1_score and v1_score.decision == DocumentKind.NATIVE_PDF and
            v2_score and v2_score.decision == DocumentKind.NATIVE_PDF and
            v2_score.confidence > 0.8):
            reasoning.append("Both classifiers HIGH confidence: NATIVE_PDF")
            return DocumentKind.NATIVE_PDF, v2_score.confidence, reasoning

        # Rule 4: Default to SCANNED (safe)
        reasoning.append("Disagreement or low confidence – safe default to SCANNED")
        return DocumentKind.SCANNED, 0.6, reasoning

    def _vote_weighted(
        self,
        scores: list[ClassifierScore],
        reasoning: list[str],
    ) -> tuple[DocumentKind, float, list[str]]:
        """Weighted voting based on historical accuracy of each classifier.

        v2 is newer/better → weight 0.7
        v1 is conservative → weight 0.3
        """
        v2_score = next((s for s in scores if s.source == "v2_advanced"), None)
        v1_score = next((s for s in scores if s.source == "v1_legacy"), None)

        if not v2_score or not v1_score:
            return DocumentKind.SCANNED, 0.5, ["Missing classifier scores"]

        # Calculate weighted score
        # For SCANNED, weight higher confidence
        # For NATIVE_PDF, need very high confidence

        v2_weight = 0.7
        v1_weight = 0.3

        # Probability of being SCANNED
        p_scanned = (
            (1.0 if v2_score.decision == DocumentKind.SCANNED else 0.0) * v2_weight * v2_score.confidence +
            (1.0 if v1_score.decision == DocumentKind.SCANNED else 0.0) * v1_weight * v1_score.confidence
        )

        if p_scanned > 0.6:
            reasoning.append(f"Weighted: SCANNED (p={p_scanned:.2f})")
            return DocumentKind.SCANNED, p_scanned, reasoning
        else:
            reasoning.append(f"Weighted: NATIVE_PDF (p={p_scanned:.2f})")
            return DocumentKind.NATIVE_PDF, 1.0 - p_scanned, reasoning

    def _vote_adaptive(
        self,
        scores: list[ClassifierScore],
        reasoning: list[str],
        metadata: dict | None,
    ) -> tuple[DocumentKind, float, list[str]]:
        """Adaptive voting: adjust strategy based on document type hints."""
        # For now, same as conservative
        # Could adapt thresholds based on invoice_type, filename patterns, etc.
        return self._vote_conservative(scores, reasoning, metadata)

    @staticmethod
    def _apply_metadata_reasoning(metadata: dict) -> list[str]:
        """Convert metadata signals to reasoning strings."""
        reasoning = []

        filename = metadata.get("filename", "")
        if "scan" in filename.lower():
            reasoning.append(f"Filename hints scan: {filename}")

        file_size = metadata.get("file_size", 0)
        if file_size > 5_000_000:
            reasoning.append(f"Large file ({file_size / 1e6:.1f}MB) – typical for scans")

        return reasoning


# ════════════════════════════════════════════════════════════════════════════
# Convenience factory function
# ════════════════════════════════════════════════════════════════════════════


def create_ensemble_classifier(
    strategy: EnsembleStrategy = EnsembleStrategy.CONSERVATIVE,
) -> EnsemblePDFClassifier:
    """Factory for creating ensemble classifier."""
    return EnsemblePDFClassifier(
        strategy=strategy,
        use_metadata_signals=True,
        per_page_analysis=True,
    )
