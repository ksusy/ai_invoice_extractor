"""Extraction module – strategy pattern for data extraction methods.

Modul extrakce – vzor Strategie pro metody extrakce dat.
"""

from src.core.extraction.base import BaseExtractionStrategy, ExtractionContext
from src.core.extraction.regex_strategy import RegexExtractionStrategy, create_regex_strategy
from src.core.extraction.langchain_strategy import (
    LangChainExtractionStrategy,
    VisionLLMExtractionStrategy,
    LLMExtractedInvoice,
    create_langchain_strategy,
)

__all__ = [
    "BaseExtractionStrategy",
    "ExtractionContext",
    "RegexExtractionStrategy",
    "create_regex_strategy",
    "LangChainExtractionStrategy",
    "VisionLLMExtractionStrategy",
    "LLMExtractedInvoice",
    "create_langchain_strategy",
]
