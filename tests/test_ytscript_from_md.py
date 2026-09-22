from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import unittest

from ytscript_from_md import split_chunks  # noqa: E402


class TestSplitChunks(unittest.TestCase):
    def test_merges_heading_only_parts(self) -> None:
        md = """---
a: 1
---

# 書

## 目錄

- 一

## 第一部

### 前言

內容甲。

### 第二章

內容乙。
"""
        chunks = split_chunks(md)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(any("目錄" in c for c in chunks))
        self.assertTrue(any("第一部" in c and "前言" in c for c in chunks))
        self.assertTrue(any("內容乙" in c for c in chunks))


class RejectLLM:
    def __init__(self, text: str, finish_reason: str) -> None:
        self.model = "fake"
        self.force = False
        self.text = text
        self.finish_reason = finish_reason

    def complete(self, **kwargs):
        class _Rec:
            response_text = self.text
            finish_reason = self.finish_reason

        return _Rec()


class TestConvertChunkRejectsBadCompletions(unittest.TestCase):
    def test_empty_output_raises(self) -> None:
        from book_combiner.llm.base import LLMError
        from ytscript_from_md import convert_chunk

        with self.assertRaises(LLMError):
            convert_chunk(
                client=RejectLLM("", "stop"),
                system="sys",
                chunk="原文一段。",
                index=0,
                total=1,
            )

    def test_length_finish_raises(self) -> None:
        from book_combiner.llm.base import LLMError
        from ytscript_from_md import convert_chunk

        with self.assertRaises(LLMError):
            convert_chunk(
                client=RejectLLM("truncated", "length"),
                system="sys",
                chunk="原文一段。",
                index=1,
                total=2,
            )
