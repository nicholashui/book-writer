"""Keyed disk cache for LLM completions under artifacts/<topic>/cache/."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from book_combiner.models import CacheKey, CacheRecord

log = logging.getLogger(__name__)


def cache_key_dict(key: CacheKey) -> dict:
    """Dict form of CacheKey with sorted input_hashes for canonical JSON."""
    return {
        "input_hashes": sorted(key.input_hashes),
        "model": key.model,
        "params": key.params,
        "prompt_hash": key.prompt_hash,
        "stage": key.stage,
    }


def cache_key_hash(key: CacheKey) -> str:
    """sha256 of canonical CacheKey JSON (sort_keys, no ASCII escape)."""
    payload = json.dumps(
        cache_key_dict(key),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record_from_dict(data: dict) -> CacheRecord:
    key_data = data["key"]
    key = CacheKey(
        stage=key_data["stage"],
        model=key_data["model"],
        prompt_hash=key_data["prompt_hash"],
        params=key_data["params"],
        input_hashes=list(key_data["input_hashes"]),
    )
    return CacheRecord(
        key=key,
        response_text=data["response_text"],
        finish_reason=data["finish_reason"],
        input_tokens=int(data["input_tokens"]),
        output_tokens=int(data["output_tokens"]),
        created_at=data["created_at"],
    )


class DiskCache:
    """CacheRecord files named <sha256(canonical key)>.json."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    def path_for(self, key: CacheKey) -> Path:
        return self.cache_dir / f"{cache_key_hash(key)}.json"

    def get(self, key: CacheKey) -> CacheRecord | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            record = _record_from_dict(data)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            log.warning("Ignoring unreadable cache file %s: %s", path.name, type(exc).__name__)
            return None
        if cache_key_hash(record.key) != cache_key_hash(key):
            log.warning("Ignoring cache file %s: stored key does not match", path.name)
            return None
        return record

    def put(self, key: CacheKey, record: CacheRecord) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        tmp = path.with_name(path.name + ".partial")
        payload = asdict(record)
        payload["key"] = cache_key_dict(key)
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return path
