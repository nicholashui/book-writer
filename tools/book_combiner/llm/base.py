"""Provider-agnostic LLMClient.complete protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from book_combiner.models import CacheRecord


class LLMError(Exception):
    """LLM call failed (missing auth, HTTP error after retries, or bad payload)."""


class LLMClient(Protocol):
    """Provider-agnostic completion. Default backend is xAI chat completions."""

    model: str
    force: bool

    def complete(
        self,
        *,
        stage: str,
        prompt_hash: str,
        input_hashes: Sequence[str],
        user: str,
        temperature: float,
        max_tokens: int,
        max_input_chars: int,
        system: str = "",
        json_mode: bool = False,
        model: str | None = None,
        force: bool | None = None,
    ) -> CacheRecord:
        """Send one chat completion, with disk cache and 429/5xx retries.

        ``finish_reason='length'`` is returned to the caller and is not retried
        with the same payload. ``force=True`` skips reading the cache.
        """
        ...
