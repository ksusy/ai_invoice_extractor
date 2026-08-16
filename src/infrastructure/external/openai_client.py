"""Thin wrapper around the OpenAI API client.

Provides retry logic, timeout configuration, and structured logging.
"""

from __future__ import annotations

import logging
import time

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = 60.0


class OpenAIClient:
    """Lazy-initialised OpenAI API client with retries and logging."""

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError(
                    "Install the 'openai' package: pip install openai"
                ) from exc
            settings = get_settings()
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not configured")
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )
        return self._client

    async def complete(
        self,
        prompt: str,
        *,
        model: str = "gpt-4o",
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Send a chat-completion request and return the assistant message."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                **kwargs,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            logger.info(
                "OpenAI %s completed in %.0fms (tokens: %d+%d)",
                model, duration_ms, prompt_tokens, completion_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "OpenAI %s failed after %.0fms: %s",
                model, duration_ms, e,
            )
            raise
