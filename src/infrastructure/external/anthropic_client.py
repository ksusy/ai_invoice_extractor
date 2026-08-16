"""Thin wrapper around the Anthropic API client.

Provides retry logic, timeout configuration, and structured logging.
"""

from __future__ import annotations

import logging
import time

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
TIMEOUT_SECONDS = 60.0


class AnthropicClient:
    """Lazy-initialised Anthropic API client with retries and logging."""

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise ImportError(
                    "Install the 'anthropic' package: pip install anthropic"
                ) from exc
            settings = get_settings()
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is not configured")
            self._client = AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                timeout=TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )
        return self._client

    async def complete(
        self,
        prompt: str,
        *,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Send a message request and return the assistant response."""
        client = self._get_client()

        extra_kwargs = {}
        if system_prompt:
            extra_kwargs["system"] = system_prompt

        start = time.perf_counter()
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **extra_kwargs,
                **kwargs,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0
            logger.info(
                "Anthropic %s completed in %.0fms (tokens: %d+%d)",
                model, duration_ms, input_tokens, output_tokens,
            )
            return response.content[0].text
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "Anthropic %s failed after %.0fms: %s",
                model, duration_ms, e,
            )
            raise
