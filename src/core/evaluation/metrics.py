"""Evaluation metrics for LLM-based invoice extraction.

Provides rigorous, thesis-grade evaluation of extraction quality across
multiple strategies (Regex, LangChain/LLM, Vision, Hybrid) on Czech
utility invoices.

Metrics implemented
-------------------
- Exact Match (EM) with Unicode-aware normalisation
- Numeric Tolerance Match  (default ±0.01 CZK)
- Date Match with format normalisation
- Character Error Rate (CER) and Levenshtein Similarity
- Word Error Rate (WER)
- Per-field Precision, Recall, F1 across a corpus
- Extraction Rate (fields extracted vs. expected)
- Expected Calibration Error (ECE)
- Cost Efficiency (accuracy per USD)
- Throughput (invoices per second)
- Bootstrap 95 % confidence intervals for all scalar metrics
- Cohen's d effect size for strategy-vs-strategy comparisons

Usage (notebook)
----------------
>>> from src.core.evaluation.metrics import (
...     FieldResult, InvoiceEvaluation, MetricsCalculator, EvaluationReport,
... )
>>> calc = MetricsCalculator()
>>> report = EvaluationReport(evaluations)
>>> report.to_csv("artifacts/results.csv")

Field coverage
--------------
EVAL_FIELDS lists the 14 canonical fields compared against ground truth.
Numeric and date fields receive specialised comparators in addition to EM.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from random import Random
from typing import Any, Sequence

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

#: Canonical field names evaluated against ground truth.
EVAL_FIELDS: list[str] = [
    "invoice_number",
    "variable_symbol",
    "customer_tax_id",
    "supplier_tax_id",
    "period_from",
    "period_to",
    "issue_date",
    "due_date",
    "total_amount_inc_vat",
    "total_amount_ex_vat",
    "commodity",
    "ean_code",
    "eic_code",
    "supply_point_code",
]

#: Fields compared with numeric tolerance instead of / in addition to EM.
NUMERIC_FIELDS: frozenset[str] = frozenset(
    ["total_amount_inc_vat", "total_amount_ex_vat"]
)

#: Fields compared with date normalisation in addition to EM.
DATE_FIELDS: frozenset[str] = frozenset(
    ["period_from", "period_to", "issue_date", "due_date"]
)

#: Default absolute tolerance for monetary amounts (CZK).
DEFAULT_NUMERIC_TOLERANCE: float = 0.01

#: Number of bootstrap resamples for confidence intervals.
BOOTSTRAP_N: int = 2_000

#: Confidence level for bootstrap intervals.
BOOTSTRAP_CI: float = 0.95

# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class FieldResult:
    """Comparison result for a single field on a single invoice.

    Parameters
    ----------
    field_name:
        One of :data:`EVAL_FIELDS`.
    expected:
        Ground-truth value as a string (``None`` if the field is absent
        from the ground-truth annotation, meaning *not evaluated*).
    extracted:
        Value returned by the extraction strategy (``None`` = not found).
    is_exact_match:
        True when normalised strings are identical.
    is_fuzzy_match:
        True when Levenshtein similarity >= 0.9 (set externally if needed).
    is_numeric_match:
        True for numeric fields when ``|expected - extracted| <= tolerance``.
    is_date_match:
        True for date fields when both parse to the same ``date`` object.
    is_present:
        True when *extracted* is a non-empty value (extraction attempted).
    cer:
        Character Error Rate  (0.0 = identical, 1.0 = completely different).
        ``None`` when either value is missing.
    confidence:
        Model confidence propagated from the parent :class:`InvoiceEvaluation`.
    """

    field_name: str
    expected: str | None
    extracted: str | None
    is_exact_match: bool = False
    is_fuzzy_match: bool = False
    is_numeric_match: bool = False
    is_date_match: bool = False
    is_present: bool = False
    cer: float | None = None
    confidence: float | None = None

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def is_evaluable(self) -> bool:
        """True when ground truth is available for this field."""
        return self.expected is not None

    @property
    def levenshtein_similarity(self) -> float | None:
        """1 - CER; ``None`` when CER is unavailable."""
        if self.cer is None:
            return None
        return 1.0 - self.cer

    @property
    def is_correct(self) -> bool:
        """Composite correctness: EM **or** numeric/date match (for typed fields)."""
        if self.is_exact_match:
            return True
        if self.field_name in NUMERIC_FIELDS and self.is_numeric_match:
            return True
        if self.field_name in DATE_FIELDS and self.is_date_match:
            return True
        return False


@dataclass
class InvoiceEvaluation:
    """Evaluation record for a single invoice extraction run.

    Parameters
    ----------
    filename:
        PDF / source file name (used as a corpus key).
    strategy:
        Strategy identifier, e.g. ``"regex"``, ``"langchain_gpt4o"``.
    model_name:
        LLM model name or ``""`` for rule-based strategies.
    confidence:
        Overall extraction confidence reported by the strategy (0–1).
    latency_ms:
        Wall-clock extraction time in milliseconds.
    cost_usd:
        Estimated API cost in USD (0.0 for local/free strategies).
    token_count:
        Total tokens consumed (prompt + completion), 0 if not applicable.
    field_results:
        One :class:`FieldResult` per entry in :data:`EVAL_FIELDS`.
    error:
        Exception message if extraction failed entirely; ``None`` on success.
    """

    filename: str
    strategy: str
    model_name: str = ""
    confidence: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    token_count: int = 0
    field_results: list[FieldResult] = field(default_factory=list)
    error: str | None = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_failed(self) -> bool:
        """True when the extraction pipeline raised an unhandled error."""
        return self.error is not None

    @property
    def evaluable_fields(self) -> list[FieldResult]:
        """Field results where ground truth is available."""
        return [fr for fr in self.field_results if fr.is_evaluable]

    @property
    def exact_accuracy(self) -> float:
        """Fraction of evaluable fields that are exact matches."""
        ev = self.evaluable_fields
        if not ev:
            return 0.0
        return sum(1 for fr in ev if fr.is_exact_match) / len(ev)

    @property
    def composite_accuracy(self) -> float:
        """Fraction of evaluable fields that are *correct* (EM or typed match)."""
        ev = self.evaluable_fields
        if not ev:
            return 0.0
        return sum(1 for fr in ev if fr.is_correct) / len(ev)

    @property
    def extraction_rate(self) -> float:
        """Fraction of evaluable fields where a non-empty value was extracted."""
        ev = self.evaluable_fields
        if not ev:
            return 0.0
        return sum(1 for fr in ev if fr.is_present) / len(ev)

    @property
    def mean_cer(self) -> float | None:
        """Mean CER across evaluable fields that have a CER value."""
        cers = [fr.cer for fr in self.evaluable_fields if fr.cer is not None]
        return statistics.mean(cers) if cers else None


@dataclass
class FieldMetrics:
    """Aggregated metrics for one field across the entire corpus.

    Parameters
    ----------
    field_name:
        Name of the field.
    n_evaluable:
        Number of invoices where ground truth exists for this field.
    n_correct:
        Exact / typed matches.
    n_present:
        Extractions that returned a non-empty value.
    n_true_positive:
        Correct *and* present.
    mean_cer:
        Mean Character Error Rate across all evaluable fields.
    mean_wer:
        Mean Word Error Rate across all evaluable fields.
    """

    field_name: str
    n_evaluable: int = 0
    n_correct: int = 0
    n_present: int = 0
    n_true_positive: int = 0
    mean_cer: float | None = None
    mean_wer: float | None = None

    @property
    def precision(self) -> float:
        """TP / (TP + FP)  =  TP / n_present."""
        if self.n_present == 0:
            return 0.0
        return self.n_true_positive / self.n_present

    @property
    def recall(self) -> float:
        """TP / (TP + FN)  =  TP / n_evaluable."""
        if self.n_evaluable == 0:
            return 0.0
        return self.n_true_positive / self.n_evaluable

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        if p + r == 0.0:
            return 0.0
        return 2.0 * p * r / (p + r)

    @property
    def accuracy(self) -> float:
        """n_correct / n_evaluable."""
        if self.n_evaluable == 0:
            return 0.0
        return self.n_correct / self.n_evaluable

    @property
    def extraction_rate(self) -> float:
        """n_present / n_evaluable."""
        if self.n_evaluable == 0:
            return 0.0
        return self.n_present / self.n_evaluable


@dataclass
class CalibrationMetrics:
    """Expected Calibration Error and reliability-diagram data.

    Parameters
    ----------
    ece:
        Expected Calibration Error (lower is better, 0 = perfectly calibrated).
    n_bins:
        Number of equal-width confidence bins used.
    bin_confidences:
        Mean confidence within each non-empty bin.
    bin_accuracies:
        Mean accuracy within each non-empty bin.
    bin_counts:
        Number of samples in each non-empty bin.
    """

    ece: float
    n_bins: int
    bin_confidences: list[float]
    bin_accuracies: list[float]
    bin_counts: list[int]

    @property
    def max_calibration_error(self) -> float:
        """Maximum per-bin |confidence - accuracy| (MCE)."""
        if not self.bin_confidences:
            return 0.0
        return max(
            abs(c - a)
            for c, a in zip(self.bin_confidences, self.bin_accuracies)
        )


@dataclass
class ConfidenceInterval:
    """Bootstrap confidence interval for a scalar metric.

    Parameters
    ----------
    point_estimate:
        Value computed on the full dataset.
    lower:
        Lower bound of the CI.
    upper:
        Upper bound of the CI.
    level:
        Nominal coverage (e.g. 0.95 for 95 %).
    n_resamples:
        Number of bootstrap resamples used.
    """

    point_estimate: float
    lower: float
    upper: float
    level: float = BOOTSTRAP_CI
    n_resamples: int = BOOTSTRAP_N

    def __str__(self) -> str:
        pct = int(self.level * 100)
        return (
            f"{self.point_estimate:.4f} "
            f"[{pct}% CI: {self.lower:.4f}–{self.upper:.4f}]"
        )


@dataclass
class CorpusMetrics:
    """All aggregated metrics for one (strategy, model) combination.

    Parameters
    ----------
    strategy:
        Extraction strategy identifier.
    model:
        LLM model name (empty string for rule-based strategies).
    n_invoices:
        Total invoices in the evaluation corpus.
    n_failed:
        Invoices where extraction raised an error.
    per_field:
        Mapping from field name to :class:`FieldMetrics`.
    overall_exact_accuracy:
        Corpus-level mean exact accuracy.
    overall_composite_accuracy:
        Corpus-level mean composite (EM + typed) accuracy.
    overall_extraction_rate:
        Corpus-level mean extraction rate.
    mean_cer:
        Corpus-level mean CER (all fields, all invoices).
    mean_wer:
        Corpus-level mean WER.
    mean_levenshtein_similarity:
        Corpus-level mean Levenshtein similarity.
    mean_latency_ms:
        Mean extraction latency per invoice.
    total_cost_usd:
        Sum of API costs across all invoices.
    throughput_invoices_per_sec:
        Corpus throughput (n_invoices / total_wall_time_s).
        ``None`` if latency data unavailable.
    cost_efficiency:
        overall_composite_accuracy / total_cost_usd.
        ``None`` if cost is zero.
    calibration:
        :class:`CalibrationMetrics` for confidence calibration.
        ``None`` if confidence scores unavailable.
    ci:
        Bootstrap CIs for the main scalar metrics.
    """

    strategy: str
    model: str
    n_invoices: int
    n_failed: int

    per_field: dict[str, FieldMetrics]

    overall_exact_accuracy: float
    overall_composite_accuracy: float
    overall_extraction_rate: float
    mean_cer: float | None
    mean_wer: float | None
    mean_levenshtein_similarity: float | None

    mean_latency_ms: float
    total_cost_usd: float
    throughput_invoices_per_sec: float | None
    cost_efficiency: float | None

    calibration: CalibrationMetrics | None

    ci: dict[str, ConfidenceInterval] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# NORMALISATION HELPERS
# ════════════════════════════════════════════════════════════════════════════


def _normalise_string(s: str) -> str:
    """Normalise a string for exact-match comparison.

    Steps applied (order matters):
    1. Unicode NFC normalisation.
    2. Strip leading/trailing whitespace.
    3. Collapse internal whitespace runs to a single space.
    4. Lower-case.
    5. Remove punctuation that is semantically transparent (hyphens used as
       thousands separators, trailing dots, etc.) — conservative: only
       leading/trailing hyphens and dots are stripped so invoice numbers like
       ``"FAK-2024-001"`` stay intact internally.
    """
    s = unicodedata.normalize("NFC", s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    return s


def _parse_date_flexible(raw: str | None) -> date | None:
    """Parse a date string in multiple formats common in Czech invoices.

    Supported formats (in priority order):
    - ``DD.MM.YYYY``  (Czech standard)
    - ``DD.MM.YY``    (short year, OCR artefact)
    - ``YYYY-MM-DD``  (ISO 8601)
    - ``YYYY.MM.DD``  (ISO with dots)
    - ``DD/MM/YYYY``  (slash-separated)
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("none", "null", ""):
        return None

    # Try ISO first (unambiguous)
    for fmt in ("%Y-%m-%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    # Czech / slash formats
    s2 = re.sub(r"\.\s+", ".", s)   # "24. 12. 2024" -> "24.12.2024"
    s2 = s2.replace("/", ".")

    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", s2)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass

    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{2})", s2)
    if m:
        yy = int(m.group(3))
        yyyy = 2000 + yy if yy < 80 else 1900 + yy
        try:
            return date(yyyy, int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass

    return None


def _parse_number_flexible(raw: str | None) -> float | None:
    """Parse a Czech-formatted numeric string to float.

    Handles:
    - Space / non-breaking-space as thousands separator.
    - Comma as decimal separator.
    - Dot as thousands separator (when followed by exactly 3 digits).
    - Standard English float strings.
    - Already-numeric types (passthrough).
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s or s.lower() in ("none", "null", ""):
        return None

    # Remove whitespace (incl. non-breaking)
    s = re.sub(r"[\s\u00a0]+", "", s)

    if "," in s:
        s = s.replace(".", "")  # dots are thousands separators
        s = s.replace(",", ".")
    elif "." in s:
        # dot as thousands separator pattern: digits.3digits(.3digits)*
        if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
            s = s.replace(".", "")

    if not re.fullmatch(r"-?\d+\.?\d*", s):
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ════════════════════════════════════════════════════════════════════════════
# EDIT-DISTANCE PRIMITIVES
# ════════════════════════════════════════════════════════════════════════════


def _levenshtein_distance(a: str, b: str) -> int:
    """Standard Levenshtein (edit) distance between two strings.

    Uses the classic DP O(m·n) algorithm with O(n) space.
    Returns 0 for identical strings and max(len(a), len(b)) for
    completely different strings of those lengths.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    m, n = len(a), len(b)
    # Keep two rows only
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    return prev[n]


def _word_levenshtein_distance(a: str, b: str) -> int:
    """Levenshtein distance at the word level.

    Tokenises by whitespace; each word counts as one unit.
    """
    wa = a.split()
    wb = b.split()
    if wa == wb:
        return 0

    m, n = len(wa), len(wb)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if wa[i - 1] == wb[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev

    return prev[n]


# ════════════════════════════════════════════════════════════════════════════
# METRICS CALCULATOR
# ════════════════════════════════════════════════════════════════════════════


class MetricsCalculator:
    """Stateless calculator for all individual and corpus-level metrics.

    Instantiate once and reuse across evaluations.

    Parameters
    ----------
    numeric_tolerance:
        Absolute tolerance for monetary amount comparisons (CZK).
        Defaults to :data:`DEFAULT_NUMERIC_TOLERANCE` (0.01).
    fuzzy_threshold:
        Levenshtein similarity threshold for ``is_fuzzy_match``.
        Defaults to 0.9.
    bootstrap_n:
        Number of bootstrap resamples for confidence intervals.
    bootstrap_ci:
        Nominal confidence interval coverage (0–1).
    rng_seed:
        Seed for the bootstrap random number generator.  Set to a fixed
        integer for reproducible results.
    """

    def __init__(
        self,
        numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
        fuzzy_threshold: float = 0.9,
        bootstrap_n: int = BOOTSTRAP_N,
        bootstrap_ci: float = BOOTSTRAP_CI,
        rng_seed: int | None = 42,
    ) -> None:
        self.numeric_tolerance = numeric_tolerance
        self.fuzzy_threshold = fuzzy_threshold
        self.bootstrap_n = bootstrap_n
        self.ci_level = bootstrap_ci  # coverage probability, e.g. 0.95
        self._rng = Random(rng_seed)

    # ------------------------------------------------------------------
    # String-level metrics
    # ------------------------------------------------------------------

    def exact_match(self, expected: str | None, extracted: str | None) -> bool:
        """Return True when both values normalise to the same string.

        Both arguments are run through :func:`_normalise_string` before
        comparison, so minor whitespace and case differences are ignored.
        ``None`` values never match.
        """
        if expected is None or extracted is None:
            return False
        return _normalise_string(expected) == _normalise_string(extracted)

    def cer(self, expected: str | None, extracted: str | None) -> float | None:
        """Character Error Rate: edit_distance / max(len(expected), len(extracted)).

        Returns ``None`` when either argument is ``None``.
        Returns 0.0 when both are empty strings.
        The denominator is the *maximum* length (not the reference length)
        to cap the value at 1.0 even when extracted is longer than expected.
        """
        if expected is None or extracted is None:
            return None
        a = _normalise_string(expected)
        b = _normalise_string(extracted)
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 0.0
        dist = _levenshtein_distance(a, b)
        return min(dist / max_len, 1.0)

    def levenshtein_similarity(
        self, expected: str | None, extracted: str | None
    ) -> float | None:
        """Return 1 - CER; ``None`` when CER is unavailable."""
        c = self.cer(expected, extracted)
        return None if c is None else 1.0 - c

    def wer(self, expected: str | None, extracted: str | None) -> float | None:
        """Word Error Rate: word_edit_distance / max(n_words_expected, n_words_extracted).

        Returns ``None`` when either argument is ``None``.
        Tokenises on whitespace (``str.split()``).
        """
        if expected is None or extracted is None:
            return None
        a = _normalise_string(expected)
        b = _normalise_string(extracted)
        wa, wb = a.split(), b.split()
        max_words = max(len(wa), len(wb))
        if max_words == 0:
            return 0.0
        dist = _word_levenshtein_distance(a, b)
        return min(dist / max_words, 1.0)

    def fuzzy_match(self, expected: str | None, extracted: str | None) -> bool:
        """True when Levenshtein similarity >= ``self.fuzzy_threshold``."""
        sim = self.levenshtein_similarity(expected, extracted)
        if sim is None:
            return False
        return sim >= self.fuzzy_threshold

    # ------------------------------------------------------------------
    # Typed-field metrics
    # ------------------------------------------------------------------

    def numeric_match(
        self,
        expected: str | None,
        extracted: str | None,
        tolerance: float | None = None,
    ) -> bool:
        """True when both parse to floats within ``tolerance`` of each other.

        Parameters
        ----------
        expected / extracted:
            Raw string values from ground truth / extraction result.
        tolerance:
            Override tolerance; defaults to ``self.numeric_tolerance``.
        """
        tol = tolerance if tolerance is not None else self.numeric_tolerance
        exp_f = _parse_number_flexible(expected)
        ext_f = _parse_number_flexible(extracted)
        if exp_f is None or ext_f is None:
            return False
        return abs(exp_f - ext_f) <= tol

    def date_match(self, expected: str | None, extracted: str | None) -> bool:
        """True when both parse to the same ``datetime.date`` object.

        Format-agnostic: ``"01.03.2024"`` and ``"2024-03-01"`` are equal.
        """
        exp_d = _parse_date_flexible(expected)
        ext_d = _parse_date_flexible(extracted)
        if exp_d is None or ext_d is None:
            return False
        return exp_d == ext_d

    # ------------------------------------------------------------------
    # FieldResult factory
    # ------------------------------------------------------------------

    def evaluate_field(
        self,
        field_name: str,
        expected: str | None,
        extracted: str | None,
        confidence: float | None = None,
    ) -> FieldResult:
        """Build a fully populated :class:`FieldResult` for one field.

        Parameters
        ----------
        field_name:
            Must be one of :data:`EVAL_FIELDS`.
        expected:
            Ground-truth value.  ``None`` means the field is not annotated
            in the ground truth and will not count towards corpus metrics.
        extracted:
            Value returned by the extraction strategy.  ``None`` / empty
            string means the field was not found.
        confidence:
            Model confidence inherited from the parent invoice evaluation.
        """
        is_present = extracted is not None and str(extracted).strip() not in (
            "",
            "None",
            "null",
        )

        fr = FieldResult(
            field_name=field_name,
            expected=expected,
            extracted=extracted if is_present else None,
            is_present=is_present,
            confidence=confidence,
        )

        if expected is None:
            # Field not in ground truth — skip all comparisons
            return fr

        exp_str = str(expected).strip()
        ext_str = str(extracted).strip() if extracted is not None else None

        # Exact match
        fr.is_exact_match = self.exact_match(exp_str, ext_str)

        # Fuzzy match
        fr.is_fuzzy_match = self.fuzzy_match(exp_str, ext_str)

        # CER
        fr.cer = self.cer(exp_str, ext_str)

        # Typed comparators
        if field_name in NUMERIC_FIELDS:
            fr.is_numeric_match = self.numeric_match(exp_str, ext_str)
        if field_name in DATE_FIELDS:
            fr.is_date_match = self.date_match(exp_str, ext_str)

        return fr

    def evaluate_invoice(
        self,
        filename: str,
        strategy: str,
        ground_truth: dict[str, Any],
        extracted: dict[str, Any],
        confidence: float = 0.0,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        token_count: int = 0,
        model_name: str = "",
        error: str | None = None,
    ) -> InvoiceEvaluation:
        """Build a complete :class:`InvoiceEvaluation` from raw dicts.

        Parameters
        ----------
        filename:
            Source PDF filename.
        strategy:
            Extraction strategy name.
        ground_truth:
            Flat dict mapping field names to expected string values.
        extracted:
            Flat dict mapping field names to extracted string values.
        confidence, latency_ms, cost_usd, token_count, model_name:
            Strategy-level metadata.
        error:
            Exception string if extraction failed.
        """
        field_results: list[FieldResult] = []
        for fn in EVAL_FIELDS:
            expected_raw = ground_truth.get(fn)
            expected = str(expected_raw).strip() if expected_raw is not None else None

            extracted_raw = extracted.get(fn)
            extracted_str = (
                str(extracted_raw).strip() if extracted_raw is not None else None
            )

            fr = self.evaluate_field(
                field_name=fn,
                expected=expected,
                extracted=extracted_str,
                confidence=confidence,
            )
            field_results.append(fr)

        return InvoiceEvaluation(
            filename=filename,
            strategy=strategy,
            model_name=model_name,
            confidence=confidence,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            token_count=token_count,
            field_results=field_results,
            error=error,
        )

    # ------------------------------------------------------------------
    # Per-field corpus aggregation
    # ------------------------------------------------------------------

    def compute_field_metrics(
        self, evaluations: Sequence[InvoiceEvaluation]
    ) -> dict[str, FieldMetrics]:
        """Compute :class:`FieldMetrics` for every field across the corpus.

        Only non-failed invoices with evaluable ground truth contribute
        to numerator counts; all non-failed invoices contribute to
        denominators (n_evaluable).

        Definition of TP for Precision/Recall
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        - **True Positive (TP):** field is *present* in extraction **and** *correct*.
        - **False Positive (FP):** field is *present* but *wrong*.
        - **False Negative (FN):** field is *correct in GT* but *not extracted*.
        - Precision = TP / (TP + FP) = TP / n_present
        - Recall    = TP / (TP + FN) = TP / n_evaluable
        """
        accum: dict[str, dict[str, list[float | int]]] = {
            fn: {
                "evaluable": [],
                "correct": [],
                "present": [],
                "tp": [],
                "cer": [],
                "wer": [],
            }
            for fn in EVAL_FIELDS
        }

        for inv in evaluations:
            if inv.is_failed:
                continue
            for fr in inv.field_results:
                if not fr.is_evaluable:
                    continue
                a = accum[fr.field_name]
                a["evaluable"].append(1)
                a["correct"].append(int(fr.is_correct))
                a["present"].append(int(fr.is_present))
                a["tp"].append(int(fr.is_correct and fr.is_present))
                if fr.cer is not None:
                    a["cer"].append(fr.cer)
                wer_val = self.wer(fr.expected, fr.extracted)
                if wer_val is not None:
                    a["wer"].append(wer_val)

        result: dict[str, FieldMetrics] = {}
        for fn in EVAL_FIELDS:
            a = accum[fn]
            n_ev = sum(a["evaluable"])
            n_correct = sum(a["correct"])
            n_present = sum(a["present"])
            n_tp = sum(a["tp"])
            mean_cer = statistics.mean(a["cer"]) if a["cer"] else None
            mean_wer = statistics.mean(a["wer"]) if a["wer"] else None
            result[fn] = FieldMetrics(
                field_name=fn,
                n_evaluable=n_ev,
                n_correct=n_correct,
                n_present=n_present,
                n_true_positive=n_tp,
                mean_cer=mean_cer,
                mean_wer=mean_wer,
            )

        return result

    # ------------------------------------------------------------------
    # Calibration (ECE)
    # ------------------------------------------------------------------

    def compute_ece(
        self,
        evaluations: Sequence[InvoiceEvaluation],
        n_bins: int = 10,
    ) -> CalibrationMetrics | None:
        """Compute Expected Calibration Error.

        Each invoice is treated as one sample with confidence = the
        strategy-reported confidence and accuracy = composite_accuracy.

        ECE = sum_b (|B_b| / N) * |conf_b - acc_b|

        Parameters
        ----------
        evaluations:
            Non-empty sequence of invoice evaluations.
        n_bins:
            Number of equal-width bins in [0, 1].

        Returns
        -------
        CalibrationMetrics or None
            ``None`` if all confidence values are 0 (strategy does not
            report confidence).
        """
        samples = [
            (inv.confidence, inv.composite_accuracy)
            for inv in evaluations
            if not inv.is_failed
        ]
        if not samples:
            return None
        confidences, accuracies = zip(*samples)
        if all(c == 0.0 for c in confidences):
            return None

        n_total = len(samples)
        bin_width = 1.0 / n_bins
        bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]

        for conf, acc in samples:
            idx = min(int(conf / bin_width), n_bins - 1)
            bins[idx].append((conf, acc))

        bin_confidences: list[float] = []
        bin_accuracies: list[float] = []
        bin_counts: list[int] = []
        ece = 0.0

        for b in bins:
            if not b:
                continue
            mean_conf = statistics.mean(c for c, _ in b)
            mean_acc = statistics.mean(a for _, a in b)
            count = len(b)
            ece += (count / n_total) * abs(mean_conf - mean_acc)
            bin_confidences.append(mean_conf)
            bin_accuracies.append(mean_acc)
            bin_counts.append(count)

        return CalibrationMetrics(
            ece=ece,
            n_bins=n_bins,
            bin_confidences=bin_confidences,
            bin_accuracies=bin_accuracies,
            bin_counts=bin_counts,
        )

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------

    def bootstrap_ci(
        self,
        values: Sequence[float],
        statistic: str = "mean",
    ) -> ConfidenceInterval:
        """Compute a bootstrap percentile confidence interval.

        Parameters
        ----------
        values:
            Sample of scalar measurements (e.g. per-invoice accuracies).
        statistic:
            ``"mean"`` or ``"median"``.

        Returns
        -------
        ConfidenceInterval
            Point estimate on the original data plus lower/upper bounds.

        Raises
        ------
        ValueError
            If *values* is empty or *statistic* is unrecognised.
        """
        if not values:
            raise ValueError("bootstrap_ci requires a non-empty sequence")
        if statistic not in ("mean", "median"):
            raise ValueError(f"Unknown statistic: {statistic!r}")

        vals = list(values)
        stat_fn = statistics.mean if statistic == "mean" else statistics.median
        point = stat_fn(vals)

        n = len(vals)
        bootstrap_stats: list[float] = []
        for _ in range(self.bootstrap_n):
            resample = [vals[self._rng.randrange(n)] for _ in range(n)]
            bootstrap_stats.append(stat_fn(resample))

        bootstrap_stats.sort()
        alpha = 1.0 - self.ci_level
        lower_idx = max(0, int(math.floor(alpha / 2 * self.bootstrap_n)))
        upper_idx = min(
            self.bootstrap_n - 1,
            int(math.ceil((1 - alpha / 2) * self.bootstrap_n)) - 1,
        )

        return ConfidenceInterval(
            point_estimate=point,
            lower=bootstrap_stats[lower_idx],
            upper=bootstrap_stats[upper_idx],
            level=self.ci_level,
            n_resamples=self.bootstrap_n,
        )

    def bootstrap_all_metrics(
        self, evaluations: Sequence[InvoiceEvaluation]
    ) -> dict[str, ConfidenceInterval]:
        """Bootstrap CIs for the main corpus-level scalar metrics.

        Returns a dict with keys:
        - ``"exact_accuracy"``
        - ``"composite_accuracy"``
        - ``"extraction_rate"``
        - ``"mean_cer"``
        - ``"mean_wer"``

        Each value is a :class:`ConfidenceInterval`.
        Only non-failed invoices contribute.
        """
        ok = [inv for inv in evaluations if not inv.is_failed]
        if not ok:
            return {}

        exact_accs = [inv.exact_accuracy for inv in ok]
        composite_accs = [inv.composite_accuracy for inv in ok]
        extraction_rates = [inv.extraction_rate for inv in ok]
        cers = [inv.mean_cer for inv in ok if inv.mean_cer is not None]

        wer_vals: list[float] = []
        for inv in ok:
            field_wers = [
                self.wer(fr.expected, fr.extracted)
                for fr in inv.evaluable_fields
                if self.wer(fr.expected, fr.extracted) is not None
            ]
            if field_wers:
                wer_vals.append(statistics.mean(field_wers))  # type: ignore[arg-type]

        result: dict[str, ConfidenceInterval] = {}
        result["exact_accuracy"] = self.bootstrap_ci(exact_accs)
        result["composite_accuracy"] = self.bootstrap_ci(composite_accs)
        result["extraction_rate"] = self.bootstrap_ci(extraction_rates)
        if cers:
            result["mean_cer"] = self.bootstrap_ci(cers)
        if wer_vals:
            result["mean_wer"] = self.bootstrap_ci(wer_vals)

        return result

    # ------------------------------------------------------------------
    # Effect size
    # ------------------------------------------------------------------

    def cohens_d(
        self,
        group_a: Sequence[float],
        group_b: Sequence[float],
    ) -> float | None:
        """Compute Cohen's d effect size between two groups.

        d = (mean_a - mean_b) / pooled_std

        The pooled standard deviation uses the *population* formula
        (divides by n, not n-1) to be consistent with the original
        Cohen (1988) definition.  Returns ``None`` when either group
        has fewer than 2 elements or the pooled SD is zero.

        Interpretation (conventional thresholds):
        - |d| < 0.2  : negligible
        - 0.2 ≤ |d| < 0.5 : small
        - 0.5 ≤ |d| < 0.8 : medium
        - |d| ≥ 0.8  : large
        """
        if len(group_a) < 2 or len(group_b) < 2:
            return None

        mean_a = statistics.mean(group_a)
        mean_b = statistics.mean(group_b)
        var_a = statistics.pvariance(group_a)
        var_b = statistics.pvariance(group_b)
        pooled_std = math.sqrt((var_a + var_b) / 2.0)

        if pooled_std == 0.0:
            return None

        return (mean_a - mean_b) / pooled_std

    @staticmethod
    def cohens_d_magnitude(d: float | None) -> str:
        """Verbal label for a Cohen's d value."""
        if d is None:
            return "undefined"
        magnitude = abs(d)
        if magnitude < 0.2:
            return "negligible"
        if magnitude < 0.5:
            return "small"
        if magnitude < 0.8:
            return "medium"
        return "large"

    # ------------------------------------------------------------------
    # Throughput & cost efficiency
    # ------------------------------------------------------------------

    def throughput(
        self,
        evaluations: Sequence[InvoiceEvaluation],
    ) -> float | None:
        """Compute throughput in invoices per second.

        Uses *total* wall-clock time = sum of per-invoice latencies.
        Returns ``None`` when total latency is zero.
        """
        total_ms = sum(inv.latency_ms for inv in evaluations)
        if total_ms == 0.0:
            return None
        n = len([inv for inv in evaluations if not inv.is_failed])
        return n / (total_ms / 1000.0)

    def cost_efficiency(
        self,
        evaluations: Sequence[InvoiceEvaluation],
    ) -> float | None:
        """Accuracy per USD spent across the corpus.

        Returns composite_accuracy / total_cost_usd.
        Returns ``None`` when total cost is zero.
        """
        total_cost = sum(inv.cost_usd for inv in evaluations)
        if total_cost == 0.0:
            return None
        ok = [inv for inv in evaluations if not inv.is_failed]
        if not ok:
            return None
        mean_acc = statistics.mean(inv.composite_accuracy for inv in ok)
        return mean_acc / total_cost

    # ------------------------------------------------------------------
    # Full corpus computation
    # ------------------------------------------------------------------

    def compute_corpus_metrics(
        self,
        evaluations: Sequence[InvoiceEvaluation],
        strategy: str = "",
        model: str = "",
    ) -> CorpusMetrics:
        """Compute all corpus-level metrics for a group of evaluations.

        Parameters
        ----------
        evaluations:
            All :class:`InvoiceEvaluation` objects for one (strategy, model)
            combination.
        strategy:
            Strategy identifier (for labelling the result).
        model:
            Model name (for labelling the result).

        Returns
        -------
        CorpusMetrics
            Fully populated metrics object including bootstrap CIs.
        """
        ok = [inv for inv in evaluations if not inv.is_failed]
        n_invoices = len(evaluations)
        n_failed = n_invoices - len(ok)

        # Per-field
        per_field = self.compute_field_metrics(evaluations)

        # Overall scalars
        overall_exact = statistics.mean(inv.exact_accuracy for inv in ok) if ok else 0.0
        overall_composite = (
            statistics.mean(inv.composite_accuracy for inv in ok) if ok else 0.0
        )
        overall_extraction_rate = (
            statistics.mean(inv.extraction_rate for inv in ok) if ok else 0.0
        )

        # CER / WER / similarity across all fields and all invoices
        all_cers: list[float] = []
        all_wers: list[float] = []
        for inv in ok:
            for fr in inv.evaluable_fields:
                if fr.cer is not None:
                    all_cers.append(fr.cer)
                wv = self.wer(fr.expected, fr.extracted)
                if wv is not None:
                    all_wers.append(wv)

        mean_cer = statistics.mean(all_cers) if all_cers else None
        mean_wer = statistics.mean(all_wers) if all_wers else None
        mean_lev_sim = (1.0 - mean_cer) if mean_cer is not None else None

        # Latency / cost
        mean_latency = statistics.mean(inv.latency_ms for inv in ok) if ok else 0.0
        total_cost = sum(inv.cost_usd for inv in evaluations)

        # Throughput
        tput = self.throughput(evaluations)

        # Cost efficiency
        cost_eff = self.cost_efficiency(evaluations)

        # Calibration (ECE)
        calibration = self.compute_ece(evaluations)

        # Bootstrap CIs
        ci = self.bootstrap_all_metrics(evaluations)

        return CorpusMetrics(
            strategy=strategy,
            model=model,
            n_invoices=n_invoices,
            n_failed=n_failed,
            per_field=per_field,
            overall_exact_accuracy=overall_exact,
            overall_composite_accuracy=overall_composite,
            overall_extraction_rate=overall_extraction_rate,
            mean_cer=mean_cer,
            mean_wer=mean_wer,
            mean_levenshtein_similarity=mean_lev_sim,
            mean_latency_ms=mean_latency,
            total_cost_usd=total_cost,
            throughput_invoices_per_sec=tput,
            cost_efficiency=cost_eff,
            calibration=calibration,
            ci=ci,
        )


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION REPORT
# ════════════════════════════════════════════════════════════════════════════


class EvaluationReport:
    """Aggregates evaluation results and generates human-readable reports.

    Parameters
    ----------
    evaluations:
        All :class:`InvoiceEvaluation` objects across all strategies.
    calculator:
        :class:`MetricsCalculator` instance to use.  A default instance
        is created when ``None`` is passed.
    group_by:
        How to group evaluations when computing :class:`CorpusMetrics`.
        ``"strategy"`` groups by ``inv.strategy``.
        ``"strategy_model"`` groups by ``(inv.strategy, inv.model_name)``.
    """

    def __init__(
        self,
        evaluations: Sequence[InvoiceEvaluation],
        calculator: MetricsCalculator | None = None,
        group_by: str = "strategy",
    ) -> None:
        self.evaluations: list[InvoiceEvaluation] = list(evaluations)
        self.calc = calculator or MetricsCalculator()
        self.group_by = group_by
        self._corpus_metrics: dict[str, CorpusMetrics] | None = None

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def _group_key(self, inv: InvoiceEvaluation) -> str:
        if self.group_by == "strategy_model":
            return f"{inv.strategy}/{inv.model_name}" if inv.model_name else inv.strategy
        return inv.strategy

    def _groups(self) -> dict[str, list[InvoiceEvaluation]]:
        groups: dict[str, list[InvoiceEvaluation]] = {}
        for inv in self.evaluations:
            key = self._group_key(inv)
            groups.setdefault(key, []).append(inv)
        return groups

    # ------------------------------------------------------------------
    # CorpusMetrics per group
    # ------------------------------------------------------------------

    def corpus_metrics(self) -> dict[str, CorpusMetrics]:
        """Return corpus metrics grouped by strategy (or strategy+model).

        Results are cached after the first call.
        """
        if self._corpus_metrics is not None:
            return self._corpus_metrics

        result: dict[str, CorpusMetrics] = {}
        for key, group in self._groups().items():
            strategy = group[0].strategy
            model = group[0].model_name
            result[key] = self.calc.compute_corpus_metrics(
                group, strategy=strategy, model=model
            )

        self._corpus_metrics = result
        return result

    # ------------------------------------------------------------------
    # Summary table (plain text)
    # ------------------------------------------------------------------

    def summary_table(self) -> str:
        """Return a formatted plain-text summary table.

        Columns: strategy | n | failed | EM | composite | extraction_rate |
                 CER | WER | latency | cost | throughput | ECE
        """
        metrics = self.corpus_metrics()
        if not metrics:
            return "No evaluation data."

        col_widths = {
            "strategy": max(12, max(len(k) for k in metrics)),
            "n": 5,
            "fail": 5,
            "em": 7,
            "comp": 7,
            "extr": 7,
            "cer": 7,
            "wer": 7,
            "lat_ms": 8,
            "cost_usd": 10,
            "tput": 8,
            "ece": 7,
        }

        def _fmt(v: float | None, decimals: int = 3) -> str:
            return f"{v:.{decimals}f}" if v is not None else "N/A"

        header_parts = [
            f"{'Strategy':<{col_widths['strategy']}}",
            f"{'N':>{col_widths['n']}}",
            f"{'Fail':>{col_widths['fail']}}",
            f"{'EM':>{col_widths['em']}}",
            f"{'Comp':>{col_widths['comp']}}",
            f"{'Extr':>{col_widths['extr']}}",
            f"{'CER':>{col_widths['cer']}}",
            f"{'WER':>{col_widths['wer']}}",
            f"{'Lat(ms)':>{col_widths['lat_ms']}}",
            f"{'Cost($)':>{col_widths['cost_usd']}}",
            f"{'Tput/s':>{col_widths['tput']}}",
            f"{'ECE':>{col_widths['ece']}}",
        ]
        header = "  ".join(header_parts)
        sep = "-" * len(header)

        lines = [header, sep]
        for key, cm in metrics.items():
            ece = cm.calibration.ece if cm.calibration else None
            row_parts = [
                f"{key:<{col_widths['strategy']}}",
                f"{cm.n_invoices:>{col_widths['n']}}",
                f"{cm.n_failed:>{col_widths['fail']}}",
                f"{_fmt(cm.overall_exact_accuracy):>{col_widths['em']}}",
                f"{_fmt(cm.overall_composite_accuracy):>{col_widths['comp']}}",
                f"{_fmt(cm.overall_extraction_rate):>{col_widths['extr']}}",
                f"{_fmt(cm.mean_cer):>{col_widths['cer']}}",
                f"{_fmt(cm.mean_wer):>{col_widths['wer']}}",
                f"{_fmt(cm.mean_latency_ms, 1):>{col_widths['lat_ms']}}",
                f"{_fmt(cm.total_cost_usd, 4):>{col_widths['cost_usd']}}",
                f"{_fmt(cm.throughput_invoices_per_sec, 2):>{col_widths['tput']}}",
                f"{_fmt(ece):>{col_widths['ece']}}",
            ]
            lines.append("  ".join(row_parts))

        return "\n".join(lines)

    def per_field_table(self, strategy_key: str) -> str:
        """Return a per-field breakdown table for one strategy.

        Columns: field | n | accuracy | precision | recall | F1 | CER | WER
        """
        metrics = self.corpus_metrics()
        if strategy_key not in metrics:
            return f"Strategy {strategy_key!r} not found."

        cm = metrics[strategy_key]
        col_w = max(20, max(len(fn) for fn in EVAL_FIELDS))

        def _fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else " N/A "

        header_parts = [
            f"{'Field':<{col_w}}",
            f"{'n':>5}",
            f"{'Acc':>7}",
            f"{'Prec':>7}",
            f"{'Rec':>7}",
            f"{'F1':>7}",
            f"{'CER':>7}",
            f"{'WER':>7}",
        ]
        header = "  ".join(header_parts)
        sep = "-" * len(header)

        lines = [f"Per-field metrics — {strategy_key}", header, sep]
        for fn in EVAL_FIELDS:
            fm = cm.per_field.get(fn)
            if fm is None:
                continue
            row_parts = [
                f"{fn:<{col_w}}",
                f"{fm.n_evaluable:>5}",
                f"{_fmt(fm.accuracy):>7}",
                f"{_fmt(fm.precision):>7}",
                f"{_fmt(fm.recall):>7}",
                f"{_fmt(fm.f1):>7}",
                f"{_fmt(fm.mean_cer):>7}",
                f"{_fmt(fm.mean_wer):>7}",
            ]
            lines.append("  ".join(row_parts))

        return "\n".join(lines)

    def effect_size_table(
        self,
        metric: str = "composite_accuracy",
    ) -> str:
        """Return a Cohen's d pairwise comparison table across strategies.

        Parameters
        ----------
        metric:
            Which per-invoice metric to compare.
            One of ``"exact_accuracy"``, ``"composite_accuracy"``,
            ``"extraction_rate"``.
        """
        valid_metrics = {"exact_accuracy", "composite_accuracy", "extraction_rate"}
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")

        groups = self._groups()
        keys = sorted(groups.keys())

        def _get_vals(inv_list: list[InvoiceEvaluation]) -> list[float]:
            ok = [inv for inv in inv_list if not inv.is_failed]
            return [getattr(inv, metric) for inv in ok]

        lines = [f"Cohen's d effect sizes for {metric!r}", ""]
        lines.append(f"{'Strategy A':<30}  {'Strategy B':<30}  {'d':>8}  {'|d| magnitude'}")
        lines.append("-" * 80)
        for i, ka in enumerate(keys):
            for kb in keys[i + 1 :]:
                va = _get_vals(groups[ka])
                vb = _get_vals(groups[kb])
                d = self.calc.cohens_d(va, vb)
                mag = self.calc.cohens_d_magnitude(d)
                d_str = f"{d:.4f}" if d is not None else "  N/A"
                lines.append(f"{ka:<30}  {kb:<30}  {d_str:>8}  {mag}")

        return "\n".join(lines)

    def ci_table(self) -> str:
        """Return a table of bootstrap confidence intervals per strategy."""
        metrics = self.corpus_metrics()
        lines = ["Bootstrap 95% Confidence Intervals", ""]

        for key, cm in metrics.items():
            lines.append(f"  Strategy: {key}")
            if not cm.ci:
                lines.append("    No CI data available.")
                continue
            for metric_name, ci in sorted(cm.ci.items()):
                lines.append(f"    {metric_name:<30} {ci}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_csv(
        self,
        path: str | Path,
        level: str = "invoice",
    ) -> Path:
        """Export evaluation data to CSV.

        Parameters
        ----------
        path:
            Output file path.
        level:
            ``"invoice"`` — one row per invoice (summary metrics).
            ``"field"``   — one row per (invoice × field).

        Returns
        -------
        Path
            Resolved output path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if level == "invoice":
            self._write_invoice_csv(out)
        elif level == "field":
            self._write_field_csv(out)
        else:
            raise ValueError(f"level must be 'invoice' or 'field', got {level!r}")

        return out

    def _write_invoice_csv(self, path: Path) -> None:
        fieldnames = [
            "filename",
            "strategy",
            "model_name",
            "confidence",
            "latency_ms",
            "cost_usd",
            "token_count",
            "exact_accuracy",
            "composite_accuracy",
            "extraction_rate",
            "mean_cer",
            "error",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for inv in self.evaluations:
                writer.writerow(
                    {
                        "filename": inv.filename,
                        "strategy": inv.strategy,
                        "model_name": inv.model_name,
                        "confidence": round(inv.confidence, 6),
                        "latency_ms": round(inv.latency_ms, 2),
                        "cost_usd": round(inv.cost_usd, 8),
                        "token_count": inv.token_count,
                        "exact_accuracy": round(inv.exact_accuracy, 6),
                        "composite_accuracy": round(inv.composite_accuracy, 6),
                        "extraction_rate": round(inv.extraction_rate, 6),
                        "mean_cer": (
                            round(inv.mean_cer, 6) if inv.mean_cer is not None else ""
                        ),
                        "error": inv.error or "",
                    }
                )

    def _write_field_csv(self, path: Path) -> None:
        fieldnames = [
            "filename",
            "strategy",
            "model_name",
            "field_name",
            "expected",
            "extracted",
            "is_exact_match",
            "is_fuzzy_match",
            "is_numeric_match",
            "is_date_match",
            "is_correct",
            "is_present",
            "cer",
            "confidence",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for inv in self.evaluations:
                for fr in inv.field_results:
                    writer.writerow(
                        {
                            "filename": inv.filename,
                            "strategy": inv.strategy,
                            "model_name": inv.model_name,
                            "field_name": fr.field_name,
                            "expected": fr.expected or "",
                            "extracted": fr.extracted or "",
                            "is_exact_match": int(fr.is_exact_match),
                            "is_fuzzy_match": int(fr.is_fuzzy_match),
                            "is_numeric_match": int(fr.is_numeric_match),
                            "is_date_match": int(fr.is_date_match),
                            "is_correct": int(fr.is_correct),
                            "is_present": int(fr.is_present),
                            "cer": (
                                round(fr.cer, 6) if fr.cer is not None else ""
                            ),
                            "confidence": (
                                round(fr.confidence, 6)
                                if fr.confidence is not None
                                else ""
                            ),
                        }
                    )

    def to_json(self, path: str | Path, indent: int = 2) -> Path:
        """Export the full evaluation dataset to JSON.

        The output is a list of invoice evaluation dicts, each containing
        the invoice-level metadata and a nested ``field_results`` list.
        Corpus-level metrics are included under a top-level ``"corpus"`` key.

        Parameters
        ----------
        path:
            Output file path.
        indent:
            JSON indentation level.

        Returns
        -------
        Path
            Resolved output path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        corpus = self.corpus_metrics()

        def _fr_to_dict(fr: FieldResult) -> dict[str, Any]:
            return {
                "field_name": fr.field_name,
                "expected": fr.expected,
                "extracted": fr.extracted,
                "is_exact_match": fr.is_exact_match,
                "is_fuzzy_match": fr.is_fuzzy_match,
                "is_numeric_match": fr.is_numeric_match,
                "is_date_match": fr.is_date_match,
                "is_correct": fr.is_correct,
                "is_present": fr.is_present,
                "cer": fr.cer,
                "confidence": fr.confidence,
            }

        def _inv_to_dict(inv: InvoiceEvaluation) -> dict[str, Any]:
            return {
                "filename": inv.filename,
                "strategy": inv.strategy,
                "model_name": inv.model_name,
                "confidence": inv.confidence,
                "latency_ms": inv.latency_ms,
                "cost_usd": inv.cost_usd,
                "token_count": inv.token_count,
                "exact_accuracy": inv.exact_accuracy,
                "composite_accuracy": inv.composite_accuracy,
                "extraction_rate": inv.extraction_rate,
                "mean_cer": inv.mean_cer,
                "error": inv.error,
                "field_results": [_fr_to_dict(fr) for fr in inv.field_results],
            }

        def _fm_to_dict(fm: FieldMetrics) -> dict[str, Any]:
            return {
                "field_name": fm.field_name,
                "n_evaluable": fm.n_evaluable,
                "n_correct": fm.n_correct,
                "n_present": fm.n_present,
                "n_true_positive": fm.n_true_positive,
                "precision": fm.precision,
                "recall": fm.recall,
                "f1": fm.f1,
                "accuracy": fm.accuracy,
                "extraction_rate": fm.extraction_rate,
                "mean_cer": fm.mean_cer,
                "mean_wer": fm.mean_wer,
            }

        def _ci_to_dict(ci: ConfidenceInterval) -> dict[str, Any]:
            return {
                "point_estimate": ci.point_estimate,
                "lower": ci.lower,
                "upper": ci.upper,
                "level": ci.level,
                "n_resamples": ci.n_resamples,
            }

        def _cal_to_dict(cal: CalibrationMetrics | None) -> dict[str, Any] | None:
            if cal is None:
                return None
            return {
                "ece": cal.ece,
                "max_calibration_error": cal.max_calibration_error,
                "n_bins": cal.n_bins,
                "bin_confidences": cal.bin_confidences,
                "bin_accuracies": cal.bin_accuracies,
                "bin_counts": cal.bin_counts,
            }

        def _cm_to_dict(cm: CorpusMetrics) -> dict[str, Any]:
            return {
                "strategy": cm.strategy,
                "model": cm.model,
                "n_invoices": cm.n_invoices,
                "n_failed": cm.n_failed,
                "overall_exact_accuracy": cm.overall_exact_accuracy,
                "overall_composite_accuracy": cm.overall_composite_accuracy,
                "overall_extraction_rate": cm.overall_extraction_rate,
                "mean_cer": cm.mean_cer,
                "mean_wer": cm.mean_wer,
                "mean_levenshtein_similarity": cm.mean_levenshtein_similarity,
                "mean_latency_ms": cm.mean_latency_ms,
                "total_cost_usd": cm.total_cost_usd,
                "throughput_invoices_per_sec": cm.throughput_invoices_per_sec,
                "cost_efficiency": cm.cost_efficiency,
                "calibration": _cal_to_dict(cm.calibration),
                "confidence_intervals": {
                    k: _ci_to_dict(v) for k, v in cm.ci.items()
                },
                "per_field": {
                    fn: _fm_to_dict(fm) for fn, fm in cm.per_field.items()
                },
            }

        payload: dict[str, Any] = {
            "evaluations": [_inv_to_dict(inv) for inv in self.evaluations],
            "corpus": {key: _cm_to_dict(cm) for key, cm in corpus.items()},
        }

        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, ensure_ascii=False, default=str)

        return out

    def print_summary(self) -> None:
        """Print the full summary report to stdout."""
        print(self.summary_table())
        print()
        print(self.ci_table())
        for key in self.corpus_metrics():
            print()
            print(self.per_field_table(key))
        if len(self.corpus_metrics()) > 1:
            print()
            print(self.effect_size_table())
