"""xAI and generic OpenAI-compatible chat-completions backends (httpx, no SDK)."""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import httpx

from book_combiner.llm.base import LLMError
from book_combiner.llm.cache import DiskCache
from book_combiner.models import CacheKey, CacheRecord

log = logging.getLogger(__name__)

DEFAULT_XAI_BASE_URL = "https://api.x.ai"
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = httpx.Timeout(300.0, connect=10.0)

_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() == "authorization":
            redacted[key] = "Bearer [REDACTED]"
        else:
            redacted[key] = value
    return redacted


def _completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/v1/chat/completions"


def _backoff_seconds(attempt: int) -> float:
    return min(8.0, 0.5 * (2 ** attempt))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_snippet(text: str, secret: str) -> str:
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    return text[:200]


class OpenAICompatClient:
    """POST {base_url}/v1/chat/completions with disk cache and 429/5xx retries."""

    def __init__(
        self,
        *,
        model: str,
        cache_dir: Path,
        api_key: str,
        base_url: str,
        force: bool = False,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        if not api_key:
            raise LLMError("API key is required")
        if not base_url:
            raise LLMError("Base URL is required")
        self.model = model
        self.force = force
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._url = _completions_url(self._base_url)
        self.cache = DiskCache(cache_dir)
        self._sleep = sleep if sleep is not None else time.sleep
        self._max_retries = max_retries
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(timeout=DEFAULT_TIMEOUT)

    @property
    def base_url(self) -> str:
        return self._base_url

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> OpenAICompatClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, "
            f"base_url={self._base_url!r}, force={self.force})"
        )

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
        model = self.model if model is None else model
        force = self.force if force is None else force
        params = {
            "json_mode": bool(json_mode),
            "max_input_chars": int(max_input_chars),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        key = CacheKey(
            stage=stage,
            model=model,
            prompt_hash=prompt_hash,
            params=params,
            input_hashes=sorted(input_hashes),
        )

        if not force:
            cached = self.cache.get(key)
            if cached is not None:
                log.info(
                    "LLM cache-hit stage=%s model=%s finish_reason=%s",
                    stage,
                    model,
                    cached.finish_reason,
                )
                return cached

        record = self._post_chat(
            key=key,
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        self.cache.put(key, record)
        log.info(
            "LLM http stage=%s model=%s finish_reason=%s input_tokens=%s output_tokens=%s",
            stage,
            model,
            record.finish_reason,
            record.input_tokens,
            record.output_tokens,
        )
        return record

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _post_chat(
        self,
        *,
        key: CacheKey,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> CacheRecord:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = self._headers()
        log.debug("POST %s headers=%s", self._url, _redact_headers(headers))

        last_status: int | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._http.post(self._url, headers=headers, json=payload)
            except httpx.RequestError as exc:
                if attempt >= self._max_retries:
                    raise LLMError(
                        f"LLM transport error after retries: {type(exc).__name__}"
                    ) from None
                self._sleep(_backoff_seconds(attempt))
                continue

            status = response.status_code
            last_status = status
            if status == 429 or status >= 500:
                if attempt >= self._max_retries:
                    raise LLMError(f"LLM HTTP {status} after retries") from None
                self._sleep(_backoff_seconds(attempt))
                continue
            if status >= 400:
                snippet = _safe_snippet(response.text, self._api_key)
                raise LLMError(f"LLM HTTP {status}: {snippet}") from None

            try:
                data = response.json()
            except ValueError:
                raise LLMError("LLM response is not JSON") from None
            return self._record_from_response(key, data)

        raise LLMError(f"LLM HTTP {last_status} after retries") from None

    def _record_from_response(self, key: CacheKey, data: object) -> CacheRecord:
        if not isinstance(data, dict):
            raise LLMError("LLM response is not a JSON object")
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("LLM response missing choices")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            message = {}
        text = message.get("content")
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)
        finish_reason = choice.get("finish_reason") or "stop"
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        return CacheRecord(
            key=key,
            response_text=text,
            finish_reason=str(finish_reason),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            created_at=_utc_now(),
        )


def build_client(
    *,
    provider: str,
    model: str,
    cache_dir: Path,
    force: bool = False,
    http_client: httpx.Client | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    sleep: Callable[[float], None] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> OpenAICompatClient:
    """Construct an xAI or OpenAI-compatible client from provider + env."""
    if provider == "xai":
        if api_key is None:
            api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        if base_url is None:
            base_url = os.environ.get("XAI_BASE_URL", DEFAULT_XAI_BASE_URL)
        if not api_key:
            raise LLMError("XAI_API_KEY or GROK_API_KEY is required")
    else:
        if api_key is None:
            api_key = os.environ.get("OPENAI_COMPAT_API_KEY")
        if base_url is None:
            base_url = os.environ.get("OPENAI_COMPAT_BASE_URL")
        if not api_key or not base_url:
            raise LLMError("OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_API_KEY are required")
    return OpenAICompatClient(
        model=model,
        cache_dir=cache_dir,
        api_key=api_key,
        base_url=base_url,
        force=force,
        http_client=http_client,
        sleep=sleep,
        max_retries=max_retries,
    )
