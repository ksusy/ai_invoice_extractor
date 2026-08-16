"""Evaluation sub-package for the AI Invoice Extractor.

Provides all data structures and metric calculations needed to assess
extraction quality against a ground-truth corpus of Czech utility invoices.

Public API
----------
Data structures
~~~~~~~~~~~~~~~
- :class:`~src.core.evaluation.metrics.FieldResult`
- :class:`~src.core.evaluation.metrics.InvoiceEvaluation`
- :class:`~src.core.evaluation.metrics.FieldMetrics`
- :class:`~src.core.evaluation.metrics.CorpusMetrics`
- :class:`~src.core.evaluation.metrics.CalibrationMetrics`
- :class:`~src.core.evaluation.metrics.ConfidenceInterval`

Calculators / reporters
~~~~~~~~~~~~~~~~~~~~~~~
- :class:`~src.core.evaluation.metrics.MetricsCalculator`
- :class:`~src.core.evaluation.metrics.EvaluationReport`

Constants
~~~~~~~~~
- :data:`~src.core.evaluation.metrics.EVAL_FIELDS`
- :data:`~src.core.evaluation.metrics.NUMERIC_FIELDS`
- :data:`~src.core.evaluation.metrics.DATE_FIELDS`
- :data:`~src.core.evaluation.metrics.DEFAULT_NUMERIC_TOLERANCE`
- :data:`~src.core.evaluation.metrics.BOOTSTRAP_N`
- :data:`~src.core.evaluation.metrics.BOOTSTRAP_CI`

Quickstart
----------
>>> from src.core.evaluation import (
...     MetricsCalculator, EvaluationReport, EVAL_FIELDS,
...     FieldResult, InvoiceEvaluation, CorpusMetrics,
... )
>>> calc = MetricsCalculator()
>>> inv_eval = calc.evaluate_invoice(
...     filename="invoice_001.pdf",
...     strategy="regex",
...     ground_truth={"invoice_number": "2024001", "issue_date": "01.03.2024"},
...     extracted={"invoice_number": "2024001", "issue_date": "2024-03-01"},
... )
>>> report = EvaluationReport([inv_eval])
>>> report.print_summary()
>>> report.to_csv("artifacts/results.csv")
>>> report.to_json("artifacts/results.json")
"""

from src.core.evaluation.metrics import (
    BOOTSTRAP_CI,
    BOOTSTRAP_N,
    DATE_FIELDS,
    DEFAULT_NUMERIC_TOLERANCE,
    EVAL_FIELDS,
    NUMERIC_FIELDS,
    CalibrationMetrics,
    ConfidenceInterval,
    CorpusMetrics,
    EvaluationReport,
    FieldMetrics,
    FieldResult,
    InvoiceEvaluation,
    MetricsCalculator,
)

__all__ = [
    # Data structures
    "FieldResult",
    "InvoiceEvaluation",
    "FieldMetrics",
    "CorpusMetrics",
    "CalibrationMetrics",
    "ConfidenceInterval",
    # Calculators / reporters
    "MetricsCalculator",
    "EvaluationReport",
    # Constants
    "EVAL_FIELDS",
    "NUMERIC_FIELDS",
    "DATE_FIELDS",
    "DEFAULT_NUMERIC_TOLERANCE",
    "BOOTSTRAP_N",
    "BOOTSTRAP_CI",
]
