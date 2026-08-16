"""Classification module – native PDF vs scanned image detection."""

from src.core.classification.base import BaseClassifier, DocumentKind
from src.core.classification.pdf_classifier import PDFClassifier, create_pdf_classifier

__all__ = [
    "BaseClassifier",
    "DocumentKind",
    "PDFClassifier",
    "create_pdf_classifier",
]
