"""Summarize compiled books: H2 split, chapter alignment, fake LLM."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
TESTS = Path(__file__).resolve().parent
for extra in (TOOLS, TESTS):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from book_combiner.models import Outline  # noqa: E402
from book_combiner.summarize import (  # noqa: E402
    RATIO,
    distribute_under_chapters,
    drop_toc,
    pack_chunks,
    run_summarize_pair,
    split_h2_sections,
)
from book_combiner.translate import heading_lines  # noqa: E402
from test_outline import SIMPLE_OUTLINE  # noqa: E402


class RatioLLM:
    model = "fake"
    force = False

    def complete(self, **kwargs):
        user = kwargs["user"]
        src = user.split("SOURCE:\n", 1)[-1]
        n = max(20, int(len(src) * RATIO))

        class _Rec:
            response_text = src[:n]
            finish_reason = "stop"

        return _Rec()


class TestSummarizeHelpers(unittest.TestCase):
    def test_split_and_drop_toc(self) -> None:
        md = "# T\n\n## Contents\n\n- a\n\n## Part I\n\nHello.\n\n## Part II\n\nWorld.\n"
        secs = drop_toc(split_h2_sections(md))
        self.assertEqual([t for t, _ in secs], ["Part I", "Part II"])

    def test_distribute_aligns_chapter_count(self) -> None:
        body = "p1\n\np2\n\np3\n\np4"
        out = distribute_under_chapters(body, ["One", "Two"])
        heads = [t for lv, t in heading_lines(out) if lv == 3]
        self.assertEqual(heads, ["One", "Two"])

    def test_pack_chunks_respects_size(self) -> None:
        text = "aaaa\n\n" * 50
        packs = pack_chunks(text, size=40)
        self.assertGreater(len(packs), 1)
        self.assertTrue(all(len(p) <= 50 for p in packs))


class TestSummarizePair(unittest.TestCase):
    def test_aligned_outputs(self) -> None:
        outline = Outline.model_validate(SIMPLE_OUTLINE)
        en = """# Facial Expression

## Contents

- skip

## Part One: Basics

Alpha paragraph about eyes and emotion.

More English facts and FACS AU1.

## Appendix A — Sources

| stem | bytes |
"""
        hk = """# 面部表情

## 目錄

- skip

## 第一部 基礎

粵語段落講眼睛同情緒。

更多書面粵語事實。

## 附錄 A · 來源

| stem | bytes |
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            en_path = root / "facial-expression.md"
            hk_path = root / "facial-expression.hk.md"
            en_path.write_text(en, encoding="utf-8")
            hk_path.write_text(hk, encoding="utf-8")
            out_en = root / "facial-expression.summary.md"
            out_hk = root / "facial-expression.summary.hk.md"
            run_summarize_pair(
                en_path=en_path,
                hk_path=hk_path,
                outline=outline,
                client=RatioLLM(),  # type: ignore[arg-type]
                out_en=out_en,
                out_hk=out_hk,
            )
            en_text = out_en.read_text(encoding="utf-8")
            hk_text = out_hk.read_text(encoding="utf-8")
            en_h2 = [t for lv, t in heading_lines(en_text) if lv == 2]
            hk_h2 = [t for lv, t in heading_lines(hk_text) if lv == 2]
            self.assertEqual(en_h2[0], "Contents")
            self.assertEqual(hk_h2[0], "目錄")
            self.assertEqual(len(en_h2), len(hk_h2))
            en_h3 = [t for lv, t in heading_lines(en_text) if lv == 3]
            hk_h3 = [t for lv, t in heading_lines(hk_text) if lv == 3]
            self.assertEqual(len(en_h3), len(hk_h3))
            self.assertIn("language: en", en_text[:200])
            self.assertIn("language: yue-Hant-HK", hk_text[:250])


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


class TestSummarizeRejectsBadCompletions(unittest.TestCase):
    def test_empty_output_raises(self) -> None:
        from book_combiner.llm.base import LLMError
        from book_combiner.summarize import summarize_chunk

        with self.assertRaises(LLMError):
            summarize_chunk(
                client=RejectLLM("", "stop"),
                text="x" * 900,
                language="English",
                stage="summarize-en",
            )

    def test_length_finish_raises(self) -> None:
        from book_combiner.llm.base import LLMError
        from book_combiner.summarize import summarize_chunk

        with self.assertRaises(LLMError):
            summarize_chunk(
                client=RejectLLM("partial", "length"),
                text="x" * 900,
                language="English",
                stage="summarize-en",
            )
