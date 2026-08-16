"""Direct query executor for running pre-built analytics queries.

Use this for known queries that don't need LLM interpretation,
saving API costs and latency.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.analytics.sql_agent import COMMON_QUERIES


class QueryExecutor:
    """Execute pre-defined analytics queries directly against the database."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run_query(self, query_name: str) -> list[dict[str, Any]]:
        """Execute a named query from COMMON_QUERIES.

        Args:
            query_name: Key from COMMON_QUERIES dict.

        Returns:
            List of row dictionaries.

        Raises:
            KeyError: If query_name is not found.
        """
        if query_name not in COMMON_QUERIES:
            available = ", ".join(COMMON_QUERIES.keys())
            raise KeyError(f"Unknown query '{query_name}'. Available: {available}")

        sql = COMMON_QUERIES[query_name]
        result = await self._session.execute(text(sql))
        rows = result.mappings().all()
        return [dict(row) for row in rows]

    async def error_summary(self) -> list[dict[str, Any]]:
        """Get transaction status distribution."""
        return await self.run_query("error_summary")

    async def common_errors(self) -> list[dict[str, Any]]:
        """Get top 10 most common error messages."""
        return await self.run_query("common_errors")

    async def frequently_corrected_fields(self) -> list[dict[str, Any]]:
        """Get most frequently user-corrected fields."""
        return await self.run_query("frequently_corrected_fields")

    async def ocr_engine_comparison(self) -> list[dict[str, Any]]:
        """Compare OCR engines by accuracy, speed, and cost."""
        return await self.run_query("ocr_engine_comparison")

    async def extraction_strategy_comparison(self) -> list[dict[str, Any]]:
        """Compare extraction strategies by accuracy and cost."""
        return await self.run_query("extraction_strategy_comparison")

    async def daily_processing_volume(self) -> list[dict[str, Any]]:
        """Get daily processing volume for the last 30 days."""
        return await self.run_query("daily_processing_volume")

    async def custom_query(self, sql: str) -> list[dict[str, Any]]:
        """Run a custom read-only query.

        Security:
            - Only a single SELECT statement is allowed.
            - Semicolons are stripped to prevent statement stacking.
            - Forbidden DDL / DML keywords are checked against tokenised words
              (not substrings) to reduce false positives while blocking
              ``DROP``, ``ALTER``, etc.
            - The database user should also have SELECT-only permissions.

        Args:
            sql: A single SELECT statement.

        Returns:
            List of row dictionaries.

        Raises:
            ValueError: If the query violates the safety rules.
        """
        import re

        # Strip trailing semicolons (prevents statement stacking)
        cleaned = sql.strip().rstrip(";").strip()
        if not cleaned:
            raise ValueError("Empty query")

        normalized = cleaned.upper()

        # Must start with SELECT (no CTEs allowed for simplicity)
        if not normalized.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")

        # Reject if multiple statements were concatenated
        if ";" in cleaned:
            raise ValueError("Multiple statements are not allowed")

        # Tokenise and check for forbidden SQL keywords as whole words
        forbidden = {
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
            "TRUNCATE", "CREATE", "GRANT", "REVOKE", "EXECUTE",
            "EXEC", "CALL", "MERGE", "REPLACE", "COPY",
        }
        # Match word boundaries to avoid false hits on column names
        # like "updated_at" or "created_by"
        tokens = set(re.findall(r"\b[A-Z_]+\b", normalized))
        found = tokens & forbidden
        if found:
            raise ValueError(f"Forbidden keywords in query: {', '.join(sorted(found))}")

        result = await self._session.execute(text(cleaned))
        rows = result.mappings().all()
        return [dict(row) for row in rows]
