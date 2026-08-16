"""LangChain SQL Agent with read-only access to analytics tables.

This agent can answer natural language questions about:
- Processing errors and failure rates
- Most frequently corrected fields
- Cost and accuracy comparisons between OCR engines and extraction strategies
"""

from __future__ import annotations

from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType

from src.config.settings import get_settings


# Tables the agent is allowed to query (read-only)
ALLOWED_TABLES = ["transactions", "ocr_results", "extraction_results"]


SYSTEM_PROMPT = """\
You are an expert data analyst for an invoice processing platform. You have \
READ-ONLY access to a PostgreSQL database containing processing results.

Available tables:
1. **transactions** – One row per uploaded invoice.
   - id (UUID), filename, commodity, status (pending|processing|completed|error), error_message, created_at, completed_at

2. **ocr_results** – One row per OCR engine run.
   - id, transaction_id (FK), engine_name (tesseract|paddleocr|easyocr), raw_text, confidence, duration_ms, cost_usd, error_message, created_at

3. **extraction_results** – One row per extraction strategy run.
   - id, transaction_id (FK), ocr_result_id (FK), strategy_name (regex|llm_text|vision_llm|hybrid), model_name
   - extracted_json (JSONB), user_corrected_json (JSONB), fields_corrected (JSONB array of field names)
   - confidence, accuracy, duration_ms, cost_usd, token_count, is_final, error_message, created_at

Key analytics you can provide:
- Error rates and common failure reasons
- Most frequently corrected fields (from fields_corrected column)
- OCR engine comparisons: accuracy vs cost vs speed
- Extraction strategy comparisons
- Processing trends over time
- Commodity-specific insights

IMPORTANT RULES:
1. ONLY use SELECT statements – never INSERT, UPDATE, DELETE, DROP, or any DDL.
2. Always LIMIT results to 100 rows unless the user explicitly requests more.
3. For cost comparisons, use cost_usd columns.
4. For accuracy, use the accuracy column (calculated after user corrections).
5. When comparing engines/strategies, always include sample sizes (COUNT).
6. Use clear column aliases in results for readability.

If you cannot answer a question with the available data, explain what data would be needed.
"""


class AnalyticsSQLAgent:
    """LangChain SQL Agent with read-only access to invoice processing analytics.

    Example usage:
        agent = AnalyticsSQLAgent()
        result = await agent.query("What are the most common processing errors?")
        print(result)
    """

    def __init__(
        self,
        model_name: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the SQL Agent.

        Args:
            model_name: Override the default model name.
            verbose: Enable debug output.
        """
        self._settings = get_settings()
        self._model_name = model_name
        self._verbose = verbose
        self._agent = None
        self._db = None

    def _get_llm(self):
        """Lazy-load the OpenAI LLM."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "Install langchain-openai: pip install langchain-openai"
            ) from exc
        return ChatOpenAI(
            model=self._model_name or "gpt-4o",
            api_key=self._settings.openai_api_key,
            temperature=0,
        )

    def _get_read_only_db(self) -> SQLDatabase:
        """Create a SQLDatabase instance restricted to allowed tables.

        The connection string is taken from settings but we enforce read-only
        semantics by restricting to specific tables.
        """
        if self._db is None:
            db_url = str(self._settings.database_url)
            # Convert async URL to sync for LangChain compatibility
            sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

            self._db = SQLDatabase.from_uri(
                sync_url,
                include_tables=ALLOWED_TABLES,
                sample_rows_in_table_info=3,
            )
        return self._db

    def _build_agent(self):
        """Construct the LangChain SQL agent with custom prompt."""
        if self._agent is None:
            llm = self._get_llm()
            db = self._get_read_only_db()

            self._agent = create_sql_agent(
                llm=llm,
                db=db,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                verbose=self._verbose,
                prefix=SYSTEM_PROMPT,
                agent_executor_kwargs={
                    "handle_parsing_errors": True,
                    "max_iterations": 10,
                },
            )
        return self._agent

    async def query(self, question: str) -> str:
        """Execute a natural language query against the analytics database.

        Args:
            question: A natural language question about invoice processing data.

        Returns:
            A human-readable answer based on the database query results.

        Example questions:
            - "What are the top 5 most common processing errors?"
            - "Which OCR engine has the best accuracy/cost ratio?"
            - "What fields are most frequently corrected by users?"
            - "Compare extraction strategies by accuracy and speed"
        """
        agent = self._build_agent()
        # LangChain agents are sync by default; run in executor for async interface
        import asyncio
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: agent.invoke({"input": question})
        )
        return result.get("output", str(result))

    def query_sync(self, question: str) -> str:
        """Synchronous version of query() for non-async contexts."""
        agent = self._build_agent()
        result = agent.invoke({"input": question})
        return result.get("output", str(result))


# ────────────────────────────────────────────────────────────────────────────
# Pre-built query templates for common analytics questions
# ────────────────────────────────────────────────────────────────────────────

COMMON_QUERIES = {
    "error_summary": """
        SELECT 
            status,
            COUNT(*) as count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM transactions
        GROUP BY status
        ORDER BY count DESC
    """,

    "common_errors": """
        SELECT 
            error_message,
            COUNT(*) as occurrences
        FROM transactions
        WHERE status = 'error' AND error_message IS NOT NULL
        GROUP BY error_message
        ORDER BY occurrences DESC
        LIMIT 10
    """,

    "frequently_corrected_fields": """
        SELECT 
            field_name,
            COUNT(*) as correction_count
        FROM extraction_results,
             LATERAL jsonb_array_elements_text(fields_corrected) AS field_name
        WHERE fields_corrected IS NOT NULL
        GROUP BY field_name
        ORDER BY correction_count DESC
        LIMIT 10
    """,

    "ocr_engine_comparison": """
        SELECT 
            engine_name,
            COUNT(*) as sample_size,
            ROUND(AVG(confidence)::numeric, 3) as avg_confidence,
            ROUND(AVG(duration_ms)::numeric, 1) as avg_duration_ms,
            ROUND(AVG(cost_usd)::numeric, 4) as avg_cost_usd,
            ROUND((AVG(confidence) / NULLIF(AVG(cost_usd), 0))::numeric, 2) as accuracy_per_dollar
        FROM ocr_results
        WHERE error_message IS NULL
        GROUP BY engine_name
        ORDER BY avg_confidence DESC
    """,

    "extraction_strategy_comparison": """
        SELECT 
            strategy_name,
            model_name,
            COUNT(*) as sample_size,
            ROUND(AVG(accuracy)::numeric, 3) as avg_accuracy,
            ROUND(AVG(confidence)::numeric, 3) as avg_confidence,
            ROUND(AVG(duration_ms)::numeric, 1) as avg_duration_ms,
            ROUND(AVG(cost_usd)::numeric, 4) as avg_cost_usd,
            ROUND(AVG(token_count)::numeric, 0) as avg_tokens
        FROM extraction_results
        WHERE error_message IS NULL AND accuracy IS NOT NULL
        GROUP BY strategy_name, model_name
        ORDER BY avg_accuracy DESC
    """,

    "daily_processing_volume": """
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as total,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
        FROM transactions
        GROUP BY DATE(created_at)
        ORDER BY date DESC
        LIMIT 30
    """,
}
