"""Application-wide settings powered by pydantic-settings.

Environment variables are loaded from .env (if present) and can be
overridden by real environment variables at runtime.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PrintMode(StrEnum):
    """Toggle between color and grayscale output for graphs/plots."""

    COLOR = "color"
    GRAYSCALE = "grayscale"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Central configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────────
    app_name: str = "AI Invoice Extractor"
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO

    # ── Database ─────────────────────────────────────────────────
    # Supports PostgreSQL (postgresql+asyncpg://...) and SQLite (sqlite+aiosqlite://...)
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/invoices",
    )

    # ── External APIs ────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Upload limits ────────────────────────────────────────────
    max_upload_size_mb: int = 50  # Max file size in MB

    # ── Print / Plot mode (thesis requirement) ───────────────────
    print_mode: PrintMode = PrintMode.COLOR

    # ── Paths ────────────────────────────────────────────────────
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    ground_truth_dir: str = "data/ground_truth"
    artifacts_dir: str = "artifacts"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
