"""LLMClient + disk cache tests using a fake httpx transport (no live network)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from book_combiner.cli import build_llm_client, parse_args  # noqa: E402
from book_combiner.llm.base import LLMError  # noqa: E402
from book_combiner.llm.cache import DiskCache, cache_key_hash  # noqa: E402
from book_combiner.llm.xai import OpenAICompatClient, build_client  # noqa: E402
from book_combiner.models import CacheKey  # noqa: E402

SECRET = "sk-super-secret-test-key-12345"

OK_BODY = {
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "merged text"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3},
}

LENGTH_BODY = {
    "choices": [
        {
            "message": {"role": "assistant", "content": "truncated"},
            "finish_reason": "length",
        }
    ],
    "usage": {"prompt_tokens": 100, "completion_tokens": 8192},
}


class FakeLLM:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._queue: list[tuple[int, dict]] = []

    def queue(self, status: int, body: dict) -> None:
        self._queue.append((status, body))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._queue:
            status, body = self._queue.pop(0)
            return httpx.Response(status, json=body)
        return httpx.Response(200, json=OK_BODY)


def _complete(client: OpenAICompatClient, **overrides):
    kwargs = dict(
        stage="merge",
        prompt_hash="a" * 64,
        input_hashes=["bbb", "aaa"],
        user="hello",
        system="sys",
        temperature=0.2,
        max_tokens=1024,
        max_input_chars=6000,
        json_mode=False,
    )
    kwargs.update(overrides)
    return client.complete(**kwargs)


class TestCacheKeyHash(unittest.TestCase):
    def test_canonical_hash_is_stable(self) -> None:
        params_a = {
            "json_mode": False,
            "max_input_chars": 6000,
            "max_tokens": 1024,
            "temperature": 0.2,
        }
        params_b = {
            "temperature": 0.2,
            "max_tokens": 1024,
            "max_input_chars": 6000,
            "json_mode": False,
        }
        key_a = CacheKey(
            stage="merge",
            model="grok-4",
            prompt_hash="p" * 64,
            params=params_a,
            input_hashes=["b", "a"],
        )
        key_b = CacheKey(
            stage="merge",
            model="grok-4",
            prompt_hash="p" * 64,
            params=params_b,
            input_hashes=["a", "b"],
        )
        digest = cache_key_hash(key_a)
        self.assertEqual(digest, cache_key_hash(key_b))
        expected_payload = json.dumps(
            {
                "input_hashes": ["a", "b"],
                "model": "grok-4",
                "params": {
                    "json_mode": False,
                    "max_input_chars": 6000,
                    "max_tokens": 1024,
                    "temperature": 0.2,
                },
                "prompt_hash": "p" * 64,
                "stage": "merge",
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.assertEqual(digest, hashlib.sha256(expected_payload.encode("utf-8")).hexdigest())

        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskCache(Path(tmp) / "cache")
            from book_combiner.models import CacheRecord

            record = CacheRecord(
                key=key_a,
                response_text="same",
                finish_reason="stop",
                input_tokens=1,
                output_tokens=1,
                created_at="2026-09-18T00:00:00Z",
            )
            path_a = cache.put(key_a, record)
            path_b = cache.path_for(key_b)
            self.assertEqual(path_a, path_b)
            self.assertEqual(path_a.name, f"{digest}.json")
            self.assertTrue(path_a.is_file())
            hit = cache.get(key_b)
            self.assertIsNotNone(hit)
            self.assertEqual(hit.response_text, "same")


class TestLLMClient(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.fake = FakeLLM()
        self.http = httpx.Client(transport=httpx.MockTransport(self.fake))
        self.client = OpenAICompatClient(
            model="grok-4",
            cache_dir=self.tmp / "cache",
            api_key=SECRET,
            base_url="https://api.x.ai",
            http_client=self.http,
            sleep=lambda _dt: None,
        )

    def tearDown(self) -> None:
        self.http.close()
        self._tmp.cleanup()

    def test_cache_hit_skips_http(self) -> None:
        first = _complete(self.client)
        self.assertEqual(first.response_text, "merged text")
        self.assertEqual(len(self.fake.requests), 1)
        second = _complete(self.client)
        self.assertEqual(second.response_text, "merged text")
        self.assertEqual(len(self.fake.requests), 1)
        digest = cache_key_hash(first.key)
        self.assertTrue((self.tmp / "cache" / f"{digest}.json").is_file())

        rerun = OpenAICompatClient(
            model="grok-4",
            cache_dir=self.tmp / "cache",
            api_key=SECRET,
            base_url="https://api.x.ai",
            http_client=self.http,
            sleep=lambda _dt: None,
        )
        third = _complete(rerun)
        self.assertEqual(third.response_text, first.response_text)
        self.assertEqual(len(self.fake.requests), 1)

    def test_force_bypasses_cache(self) -> None:
        _complete(self.client)
        self.assertEqual(len(self.fake.requests), 1)
        _complete(self.client)
        self.assertEqual(len(self.fake.requests), 1)
        forced = _complete(self.client, force=True)
        self.assertEqual(forced.response_text, "merged text")
        self.assertEqual(len(self.fake.requests), 2)

        args = parse_args(
            [
                "demo",
                "--provider",
                "xai",
                "--model",
                "grok-4",
                "--force",
            ]
        )
        self.assertTrue(args.llm["force"])
        self.assertEqual(args.llm["provider"], "xai")
        self.assertEqual(args.llm["model"], "grok-4")
        cli_client = build_llm_client(
            args,
            artifacts_root=self.tmp,
            topic="topic",
            api_key=SECRET,
            base_url="https://api.x.ai",
            http_client=self.http,
            sleep=lambda _dt: None,
        )
        self.assertTrue(cli_client.force)
        _complete(cli_client)
        _complete(cli_client)
        self.assertEqual(len(self.fake.requests), 4)

        instance = OpenAICompatClient(
            model="grok-4",
            cache_dir=self.tmp / "cache",
            api_key=SECRET,
            base_url="https://api.x.ai",
            force=True,
            http_client=self.http,
            sleep=lambda _dt: None,
        )
        _complete(instance)
        self.assertEqual(len(self.fake.requests), 5)

    def test_finish_reason_length_not_retried(self) -> None:
        self.fake.queue(200, LENGTH_BODY)
        self.fake.queue(200, OK_BODY)
        record = _complete(self.client)
        self.assertEqual(record.finish_reason, "length")
        self.assertEqual(record.response_text, "truncated")
        self.assertEqual(len(self.fake.requests), 1)

    def test_429_then_success_retries(self) -> None:
        self.fake.queue(429, {"error": {"message": "rate limit"}})
        self.fake.queue(200, OK_BODY)
        record = _complete(self.client)
        self.assertEqual(record.finish_reason, "stop")
        self.assertEqual(record.response_text, "merged text")
        self.assertEqual(len(self.fake.requests), 2)

    def test_500_then_success_retries(self) -> None:
        self.fake.queue(503, {"error": {"message": "unavailable"}})
        self.fake.queue(200, OK_BODY)
        record = _complete(self.client)
        self.assertEqual(record.response_text, "merged text")
        self.assertEqual(len(self.fake.requests), 2)

    def test_retries_exhausted_raise(self) -> None:
        for _ in range(4):
            self.fake.queue(429, {"error": {"message": "rate limit"}})
        with self.assertRaises(LLMError) as ctx:
            _complete(self.client)
        self.assertIn("429", str(ctx.exception))
        self.assertEqual(len(self.fake.requests), 4)

    def test_secrets_not_present_in_log_output(self) -> None:
        logger = logging.getLogger("book_combiner.llm")
        prev_level = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            with self.assertLogs("book_combiner.llm", level="DEBUG") as cm:
                _complete(self.client)
        finally:
            logger.setLevel(prev_level)
        blob = "\n".join(cm.output)
        self.assertNotIn(SECRET, blob)
        self.assertNotIn(f"Bearer {SECRET}", blob)
        self.assertNotIn("XAI_API_KEY", blob)
        self.assertIn("[REDACTED]", blob)
        req = self.fake.requests[0]
        self.assertEqual(req.headers["Authorization"], f"Bearer {SECRET}")
        self.assertNotIn(SECRET, repr(self.client))

    def test_request_shape_and_json_mode(self) -> None:
        _complete(self.client, json_mode=True, temperature=0.3, max_tokens=4096)
        req = self.fake.requests[0]
        self.assertEqual(str(req.url), "https://api.x.ai/v1/chat/completions")
        body = json.loads(req.content.decode("utf-8"))
        self.assertEqual(body["model"], "grok-4")
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(body["messages"][1], {"role": "user", "content": "hello"})

    def test_build_client_openai_compat_env(self) -> None:
        env = {
            "OPENAI_COMPAT_BASE_URL": "https://compat.example",
            "OPENAI_COMPAT_API_KEY": "compat-secret",
        }
        with patch.dict(os.environ, env, clear=False):
            client = build_client(
                provider="openai-compat",
                model="other-model",
                cache_dir=self.tmp / "compat-cache",
                http_client=self.http,
                sleep=lambda _dt: None,
            )
        self.assertEqual(client.base_url, "https://compat.example")
        self.assertEqual(client.model, "other-model")
        _complete(client)
        self.assertEqual(str(self.fake.requests[-1].url), "https://compat.example/v1/chat/completions")
        self.assertEqual(
            self.fake.requests[-1].headers["Authorization"],
            "Bearer compat-secret",
        )

    def test_build_client_xai_requires_key(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XAI_API_KEY", None)
            os.environ.pop("GROK_API_KEY", None)
            with self.assertRaises(LLMError) as ctx:
                build_client(
                    provider="xai",
                    model="grok-4",
                    cache_dir=self.tmp / "no-key",
                    http_client=self.http,
                    sleep=lambda _dt: None,
                )
        self.assertIn("XAI_API_KEY", str(ctx.exception))
        self.assertNotIn(SECRET, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
