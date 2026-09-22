"""Provider-agnostic LLMClient.complete protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from book_combiner.models import CacheRecord


class LLMError(Exception):
    """LLM call failed (missing auth, HTTP error after retries, or bad payload)."""


def completed_text(record: CacheRecord) -> str:
    """Return non-empty completion text; reject truncated or blank payloads."""
    if record.finish_reason == "length":
        raise LLMError("truncated LLM output (finish_reason=length)")
    text = (record.response_text or "").strip()
    if not text:
        raise LLMError("empty LLM output")
    return text


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
