"""Full-pipeline wiring on testdata/two-book-overlap (FakeLLM only)."""

from __future__ import annotations

import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from book_combiner.assemble import (  # noqa: E402
    APPENDIX_B,
    APPENDIX_E,
    count_h1,
    heading_levels,
    heading_signature,
    toc_follows_title,
)
from book_combiner.cli import STAGE_CHOICES, STAGES, parse_args  # noqa: E402
from book_combiner.merge import MergeError  # noqa: E402
from book_combiner.translate import evaluate_cantonese  # noqa: E402
from test_merge import DIGEST, DigestLLM, SWEAT_FOUR, _section, _write_merge_artifacts  # noqa: E402
from test_outline import (  # noqa: E402
    SIMPLE_OUTLINE_JSON,
    TESTDATA,
    FakeLLMClient,
    _SOURCE_BLOCK_RE,
    _run_main,
    cleanup_compiled,
    fake_merge_from_user,
)

GOLD = TESTDATA / "two-book-overlap.gold"
TOPIC = "two-book-overlap"


def _help_text() -> str:
    captured = io.StringIO()
    with redirect_stdout(captured), redirect_stderr(io.StringIO()):
        try:
            parse_args(["--help"])
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    return captured.getvalue()


class ConflictMergeLLM(FakeLLMClient):
    """Union-of-facts fake that emits one Conflict when two sources share a heading."""

    def complete(self, **kwargs):
        if kwargs.get("stage") == "merge":
            user = str(kwargs.get("user") or "")
            sources = [b.strip() for b in _SOURCE_BLOCK_RE.findall(user) if b.strip()]
            body = fake_merge_from_user(user)
            if len(sources) >= 2:
                node_m = re.search(r"outline node `([^`]+)`", user)
                node_id = node_m.group(1) if node_m else "s-eyes"
                stems = re.findall(r"---SOURCE ([^\n]+)---", user)
                a = stems[0] if stems else "book-a"
                b = stems[1] if len(stems) > 1 else "book-b"
                cid = f"c-{node_id}-1"
                tag = f"[{a}#{cid}]"
                conflict = {
                    "id": cid,
                    "node_id": node_id,
                    "claim_a": {"source": a, "text": sources[0].splitlines()[0][:80]},
                    "claim_b": {"source": b, "text": sources[1].splitlines()[0][:80]},
                    "equivalent_hint": False,
                    "inline_tag": tag,
                }
                md = body.split("---CONFLICTS-JSON---", 1)[0].strip()
                text = (
                    f"{md} {tag}\n\n---CONFLICTS-JSON---\n"
                    f"{json.dumps([conflict], ensure_ascii=False)}\n"
                )
                self._queue.append(text)
        return super().complete(**kwargs)


def _strip_generated_at(text: str) -> str:
    return re.sub(r"^generated_at: .+$", "generated_at: <stamp>", text, flags=re.M, count=1)


# Appendix A | stem | bytes | sha256 | — bytes/hash follow working-tree newlines.
# Do not use \s*$: with re.M that swallows the newline after the last row.
_APPENDIX_A_ROW_RE = re.compile(
    r"^(\| [^\s|]+ \| )\d+( \| )[0-9a-f]{64}( \|)[ \t]*$",
    re.M | re.I,
)


def _comparable_book(text: str) -> str:
    """Drop stamps, local paths, and CRLF-sensitive Appendix A hashes."""
    text = _strip_generated_at(text).replace("\r\n", "\n")
    text = re.sub(
        r"^  - .*[\\/](book-[ab]\.md)\s*$",
        r"  - testdata/two-book-overlap/\1",
        text,
        flags=re.M,
    )
    return _APPENDIX_A_ROW_RE.sub(r"\1<bytes>\2<sha256>\3", text)


class TestComparableBook(unittest.TestCase):
    def test_appendix_a_hashes_ignore_crlf_vs_lf(self) -> None:
        crlf = (
            "| book-a | 707 | e5b1f9d1bcf608264a0c9d23b4b1fe9f35231bbcc567ce86e118632428256dbf |\r\n"
            "| book-b | 264 | 65b920062ecc9a4f9335ebc726e72455be24b73e71ddab584bc1d34df0c31e4f |\r\n"
        )
        lf = (
            "| book-a | 685 | 3c749d3c0123456789abcdef0123456789abcdef0123456789abcdef01234567 |\n"
            "| book-b | 252 | abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789 |\n"
        )
        self.assertEqual(_comparable_book(crlf), _comparable_book(lf))
        self.assertIn("| book-a | <bytes> | <sha256> |", _comparable_book(lf))
        self.assertIn("| book-b | <bytes> | <sha256> |", _comparable_book(lf))


class TestGitignoreTestdataCompiled(unittest.TestCase):
    def test_testdata_root_compiled_ignored_topic_sources_not(self) -> None:
        gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/testdata/*.md", gi)
        self.assertIn("!/testdata/noise_book.md", gi)
        self.assertTrue((TESTDATA / "two-book-overlap" / "book-a.md").is_file())
        self.assertTrue((TESTDATA / "noise_book.md").is_file())


class TestHelpRunbook(unittest.TestCase):
    def test_help_documents_live_operator_run(self) -> None:
        help_text = _help_text()
        self.assertIn("hours", help_text)
        self.assertIn("tens to low hundreds USD", help_text)
        self.assertIn("grok-4", help_text)
        self.assertIn("--stage outline", help_text)
        self.assertIn("--from-stage assign", help_text)
        self.assertIn("cache", help_text.lower())
        self.assertIn("resume", help_text.lower())
        self.assertIn("--force", help_text)
        self.assertIn("rebuilds outline only", help_text)
        self.assertIn("everything after", help_text)
        self.assertIn("no stage named cluster", help_text.lower())
        self.assertNotIn("cluster", STAGE_CHOICES)
        self.assertNotIn("cluster", STAGES)
        self.assertIn("two-book-overlap --book-root testdata", help_text)
        self.assertIn("/testdata/*.md", help_text)
        self.assertIn("Deterministic prefix only", help_text)
        self.assertNotIn("in this PR", help_text)


class TestDigestStillRejected(unittest.TestCase):
    def test_digest_fixture_still_rejected(self) -> None:
        pad = "手心出汗属于精神性反应，紧张时更为明显。" * 40
        catalogue = "温热性出汗、精神性出汗、味觉性出汗和运动性出汗"
        text = f"{pad}\n\n{catalogue}。\n\n{pad}"
        section = _section(
            "a:10:20",
            text,
            heading=["激动时，手会冒汗或者颤抖吗？"],
            stem="a",
        )
        other = _section(
            "b:10:20",
            "握手时掌心潮湿也可能是紧张，而不是室温过高。" * 8,
            heading=["激动时，手会冒汗或者颤抖吗？"],
            stem="b",
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _write_merge_artifacts(artifacts, "hands", [section, other])
            fake = DigestLLM()
            from book_combiner.merge import run_merge

            results = run_merge(
                topic="hands",
                artifacts_root=artifacts,
                client=fake,
                max_input_chars=6000,
            )
            self.assertFalse(results[0].failed)
            path = artifacts / "hands" / "merge" / "nodes" / "s-hands.zh.md"
            body = path.read_text(encoding="utf-8")
            for name in SWEAT_FOUR:
                self.assertIn(name, body)
            self.assertNotIn(DIGEST, body)
            self.assertFalse((artifacts / "hands" / "merge" / "nodes" / "s-hands.FAILED.md").is_file())


class TestTwoBookOverlapE2E(unittest.TestCase):
    def test_full_pipeline_on_testdata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            fake = ConflictMergeLLM(default=SIMPLE_OUTLINE_JSON)
            try:
                code, out, err = _run_main(
                    [
                        TOPIC,
                        "--book-root",
                        str(TESTDATA),
                        "--artifacts-root",
                        str(artifacts),
                        "--model",
                        "grok-4",
                    ],
                    llm_client=fake,
                )
            finally:
                en_path = TESTDATA / f"{TOPIC}.md"
                hk_path = TESTDATA / f"{TOPIC}.hk.md"
                en_text = en_path.read_text(encoding="utf-8") if en_path.is_file() else ""
                hk_text = hk_path.read_text(encoding="utf-8") if hk_path.is_file() else ""
                cleanup_compiled(TESTDATA, TOPIC)
            self.assertEqual(code, 0, err)
            self.assertNotIn("not implemented until a later PR", err)
            self.assertIn("[assemble]", out)
            zh_path = artifacts / TOPIC / "compiled.zh.md"
            self.assertTrue(zh_path.is_file(), "compiled.zh.md missing")
            self.assertTrue(en_text, f"{TOPIC}.md missing")
            self.assertTrue(hk_text, f"{TOPIC}.hk.md missing")
            zh_text = zh_path.read_text(encoding="utf-8")

            self.assertEqual(count_h1(zh_text), 1)
            self.assertEqual(count_h1(en_text), 1)
            self.assertEqual(count_h1(hk_text), 1)
            self.assertTrue(toc_follows_title(zh_text, "目录"))
            self.assertTrue(toc_follows_title(en_text, "Contents"))
            self.assertTrue(toc_follows_title(hk_text, "目錄"))
            self.assertEqual(heading_levels(zh_text), heading_levels(en_text))
            self.assertEqual(heading_levels(zh_text), heading_levels(hk_text))

            self.assertIn(APPENDIX_B["zh"], zh_text)
            self.assertIn(APPENDIX_B["en"], en_text)
            self.assertIn(APPENDIX_B["yue"], hk_text)
            self.assertIn("心理测试 你的情绪稳定吗", zh_text)
            self.assertIn("A. 忧郁", zh_text)
            self.assertIn("结果分析", zh_text)

            self.assertIn(APPENDIX_E["zh"], zh_text)
            self.assertIn(APPENDIX_E["en"], en_text)
            self.assertIn(APPENDIX_E["yue"], hk_text)
            self.assertIn("c-s-eyes-1", zh_text)
            self.assertIn("c-s-eyes-1", en_text)
            self.assertIn("#c-s-eyes-1]", zh_text)

            ok, reasons = evaluate_cantonese(hk_text)
            self.assertTrue(ok, reasons)

            gold_levels = json.loads((GOLD / "heading_levels.json").read_text(encoding="utf-8"))
            self.assertEqual(heading_levels(zh_text), gold_levels)
            must = json.loads((GOLD / "must_contain.json").read_text(encoding="utf-8"))
            for needle in must["zh"]:
                self.assertIn(needle, zh_text)
            for needle in must["en"]:
                self.assertIn(needle, en_text)
            for needle in must["hk"]:
                self.assertIn(needle, hk_text)

            for name, actual in (("compiled.zh.md", zh_text), ("en.md", en_text), ("hk.md", hk_text)):
                gold_path = GOLD / name
                self.assertTrue(gold_path.is_file(), gold_path)
                self.assertEqual(_comparable_book(actual), _comparable_book(gold_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
