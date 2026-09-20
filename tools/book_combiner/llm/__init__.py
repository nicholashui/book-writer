"""Provider-agnostic LLM client, disk cache, and xAI/OpenAI-compatible backends."""

from book_combiner.llm.base import LLMClient, LLMError
from book_combiner.llm.cache import DiskCache, cache_key_hash
from book_combiner.llm.xai import OpenAICompatClient, build_client

__all__ = [
    "DiskCache",
    "LLMClient",
    "LLMError",
    "OpenAICompatClient",
    "build_client",
    "cache_key_hash",
]
