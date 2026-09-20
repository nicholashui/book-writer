"""MinHash signatures, Jaccard thresholds, and generic duplicate_of."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from book_combiner.extract import content_hash  # noqa: E402
from book_combiner.models import Section  # noqa: E402
from book_combiner.overlap import (  # noqa: E402
    EXACTISH_JACCARD,
    MIN_LEAF_CHARS,
    NUM_PERM,
    compute_overlap,
    estimated_jaccard,
    minhash_signature,
    normalize_text,
)


def _unique_run(n: int, offset: int = 0) -> str:
    return "".join(chr(0x4E00 + offset + i) for i in range(n))


def _section(
    sid: str,
    text: str,
    heading: list[str] | None = None,
    stem: str = "a",
) -> Section:
    return Section(
        id=sid,
        source_stem=stem,
        source_book=stem,
        heading_path=list(heading or ["泄密的眼睛"]),
        level=1,
        start_line=1,
        end_line=2,
        text=text,
        kind="body",
        char_count=len(text),
        content_hash=content_hash(text),
        duplicate_of=None,
    )


class TestMinHash(unittest.TestCase):
    def test_same_text_same_signature(self) -> None:
        text = _unique_run(120)
        a = minhash_signature(text)
        b = minhash_signature(text)
        self.assertIsNotNone(a)
        self.assertEqual(a, b)
        self.assertEqual(len(a or ()), NUM_PERM)

    def test_normalize_then_same_signature(self) -> None:
        plain = _unique_run(80)
        marked = f"**{plain[:20]}** {plain[20:]}"
        self.assertEqual(normalize_text(plain), normalize_text(marked))
        self.assertEqual(minhash_signature(plain), minhash_signature(marked))

    def test_short_leaves_skipped(self) -> None:
        short = "短文本不足四十字符。"
        self.assertLess(len(short), MIN_LEAF_CHARS)
        self.assertIsNone(minhash_signature(short))
        long_enough = _unique_run(MIN_LEAF_CHARS)
        self.assertEqual(len(long_enough), MIN_LEAF_CHARS)
        self.assertIsNotNone(minhash_signature(long_enough))

    def test_uses_hashlib_and_struct_not_builtin_hash(self) -> None:
        source = Path(inspect.getfile(sys.modules["book_combiner.overlap"])).read_text(
            encoding="utf-8"
        )
        self.assertIn("hashlib", source)
        self.assertIn("struct.pack", source)
        self.assertNotRegex(source, r"(?<![\w.])hash\(")


class TestDuplicateOf(unittest.TestCase):
    def test_near_identical_paragraphs_collapse(self) -> None:
        core = _unique_run(160)
        longer = _section("long:1:10", core + "额", heading=["眼睛"], stem="alpha")
        shorter = _section("short:1:8", core, heading=["眼睛"], stem="beta")
        self.assertGreater(longer.char_count, shorter.char_count)
        sig_a = minhash_signature(longer.text)
        sig_b = minhash_signature(shorter.text)
        self.assertIsNotNone(sig_a)
        self.assertIsNotNone(sig_b)
        self.assertGreaterEqual(estimated_jaccard(sig_a, sig_b), EXACTISH_JACCARD)

        result = compute_overlap([longer, shorter])
        self.assertEqual(result["duplicate_of"].get(shorter.id), longer.id)
        self.assertNotIn(longer.id, result["duplicate_of"])

    def test_different_body_same_heading_does_not_collapse(self) -> None:
        heading = ["第二章 眉眼真的会说话", "泄密的眼睛"]
        a = _section("a:1:20", _unique_run(160, offset=0), heading=heading, stem="one")
        other = "握手时掌心出汗属于精神性出汗而不是温热性出汗，还要区分生理性手抖。" * 4
        b = _section("b:1:20", other, heading=heading, stem="two")
        sig_a = minhash_signature(a.text)
        sig_b = minhash_signature(b.text)
        self.assertIsNotNone(sig_a)
        self.assertIsNotNone(sig_b)
        self.assertLess(estimated_jaccard(sig_a, sig_b), EXACTISH_JACCARD)
        result = compute_overlap([a, b])
        self.assertEqual(result["duplicate_of"], {})

    def test_generic_rule_catches_near_dup_pair_without_isbn_stems(self) -> None:
        opener = "我们分手吧——" + _unique_run(180)
        left = _section(
            "book-long:10:80",
            opener + "额",
            heading=["开篇"],
            stem="book-long",
        )
        right = _section(
            "book-short:10:40",
            opener,
            heading=["开篇"],
            stem="book-short",
        )
        sig_a = minhash_signature(left.text)
        sig_b = minhash_signature(right.text)
        self.assertGreaterEqual(estimated_jaccard(sig_a, sig_b), EXACTISH_JACCARD)
        result = compute_overlap([left, right])
        self.assertEqual(result["duplicate_of"].get(right.id), left.id)
        overlap_src = Path(inspect.getfile(sys.modules["book_combiner.overlap"])).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("9787554606841", overlap_src)
        self.assertNotIn("9789620756689", overlap_src)

    def test_identical_content_hash_collapses_even_if_short(self) -> None:
        text = "短句重复。"
        a = _section("aaa:1:2", text, stem="x")
        b = _section("bbb:1:2", text, stem="y")
        self.assertLess(len(text), MIN_LEAF_CHARS)
        self.assertEqual(a.content_hash, b.content_hash)
        result = compute_overlap([a, b])
        self.assertEqual(result["duplicate_of"][b.id], a.id)

    def test_canonical_tie_breaks_on_lexicographic_id(self) -> None:
        text = _unique_run(100)
        a = _section("m:1:2", text, stem="m")
        b = _section("z:1:2", text, stem="z")
        result = compute_overlap([a, b])
        self.assertEqual(result["duplicate_of"].get(b.id), a.id)
        self.assertNotIn(a.id, result["duplicate_of"])


if __name__ == "__main__":
    unittest.main()
