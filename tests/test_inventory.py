"""Heading inventory, lexicon extras, and inventory CLI wiring."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from book_combiner.cli import main  # noqa: E402
from book_combiner.inventory import heading_row, make_excerpt  # noqa: E402
from book_combiner.lexicon import LEXICON, keywords_for, load_lexicon  # noqa: E402
from book_combiner.models import Section  # noqa: E402


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = main(argv)
        except SystemExit as exc:
            if isinstance(exc.code, int):
                code = exc.code
            elif exc.code is None:
                code = 0
            else:
                stderr.write(str(exc.code) + "\n")
                code = 1
    return code, stdout.getvalue(), stderr.getvalue()


def _unique_run(n: int, offset: int = 0) -> str:
    return "".join(chr(0x4E00 + offset + i) for i in range(n))


class TestLexicon(unittest.TestCase):
    def test_generic_defaults_are_exact(self) -> None:
        self.assertEqual(
            LEXICON,
            {
                "eyes": ["眼", "瞳孔", "眨眼", "目光", "视线"],
                "brows": ["眉"],
                "nose": ["鼻"],
                "mouth": ["嘴", "唇", "舌", "牙"],
                "chin": ["下巴", "颏"],
                "smile": ["笑", "微笑"],
                "micro": ["微表情", "微反应"],
                "hands": ["手", "掌", "指"],
                "arms": ["臂", "肩"],
                "legs": ["腿", "脚", "足", "坐姿", "站姿", "走"],
                "voice": ["声", "语速", "音调", "说话"],
                "lie": ["谎", "测谎", "欺骗"],
                "emotion_basic": ["惊讶", "厌恶", "愤怒", "恐惧", "悲伤", "愉悦", "轻蔑"],
                "workplace": ["职场", "面试", "上司", "客户"],
                "quiz": ["心理测试"],
            },
        )

    def test_unknown_topic_skips_extras_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lex = load_lexicon("no-such-topic", topics_dir=Path(tmp))
            self.assertEqual(lex, {k: list(v) for k, v in LEXICON.items()})
        lex_default = load_lexicon("definitely-unknown-topic-xyz")
        self.assertEqual(lex_default, {k: list(v) for k, v in LEXICON.items()})
        extras = (
            TOOLS / "book_combiner" / "topics" / "definitely-unknown-topic-xyz" / "lexicon.json"
        )
        self.assertFalse(extras.exists())

    def test_facial_expression_extras_are_empty(self) -> None:
        path = TOOLS / "book_combiner" / "topics" / "facial-expression" / "lexicon.json"
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {})
        self.assertEqual(
            load_lexicon("facial-expression"),
            {k: list(v) for k, v in LEXICON.items()},
        )

    def test_extras_are_additive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            topic_dir = Path(tmp) / "demo-topic"
            topic_dir.mkdir()
            (topic_dir / "lexicon.json").write_text(
                '{"eyes": ["眼泪"], "custom": ["foo"]}',
                encoding="utf-8",
            )
            lex = load_lexicon("demo-topic", topics_dir=Path(tmp))
        self.assertIn("眼泪", lex["eyes"])
        self.assertIn("眼", lex["eyes"])
        self.assertEqual(lex["custom"], ["foo"])

    def test_keywords_multi_bucket(self) -> None:
        hits = keywords_for(["第二章 眉眼真的会说话", "泄密的眼睛"])
        self.assertEqual(hits, ["eyes", "brows", "voice"])
        self.assertEqual(keywords_for(["愤怒时的眉毛"]), ["brows", "emotion_basic"])
        self.assertEqual(keywords_for(["心理测试　你的情绪稳定吗"]), ["quiz"])


class TestHeadingRows(unittest.TestCase):
    def test_excerpt_and_keywords(self) -> None:
        text = "线索一：目光方向可以透露兴趣。" + ("补充" * 20)
        sec = Section(
            id="9787538583373:221:268",
            source_stem="9787538583373",
            source_book="9787538583373",
            heading_path=["第二章 眉眼真的会说话", "泄密的眼睛"],
            level=2,
            start_line=221,
            end_line=268,
            text=text,
            kind="body",
            char_count=len(text),
            content_hash="h",
            duplicate_of=None,
        )
        row = heading_row(sec, LEXICON)
        self.assertEqual(row["section_id"], sec.id)
        self.assertEqual(row["keywords"], ["eyes", "brows", "voice"])
        self.assertEqual(row["kind"], "body")
        self.assertEqual(row["char_count"], len(text))
        self.assertTrue(str(row["excerpt"]).endswith("……"))
        self.assertEqual(len(make_excerpt(text, 40).replace("……", "")), 40)


class TestInventoryCli(unittest.TestCase):
    def test_stage_inventory_and_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book_root = Path(tmp) / "book"
            topic = book_root / "two-near"
            topic.mkdir(parents=True)
            shared = "我们分手吧——" + _unique_run(180)
            (topic / "book-long.md").write_text(
                f"# 开篇\n\n{shared}额\n",
                encoding="utf-8",
            )
            (topic / "book-short.md").write_text(
                f"# 开篇\n\n{shared}\n",
                encoding="utf-8",
            )
            (topic / "other.md").write_text(
                "# 泄密的眼睛\n\n"
                + ("握手时掌心出汗属于精神性出汗而不是温热性出汗。" * 6)
                + "\n",
                encoding="utf-8",
            )
            artifacts = Path(tmp) / "artifacts"
            code, out, err = _run_main(
                [
                    "two-near",
                    "--book-root",
                    str(book_root),
                    "--artifacts-root",
                    str(artifacts),
                    "--stage",
                    "inventory",
                ]
            )
            self.assertEqual(code, 0, err)
            self.assertIn("[discover]", out)
            self.assertIn("[extract]", out)
            self.assertIn("[inventory]", out)
            self.assertIn("overlap pairs=", out)
            self.assertNotIn("9787554606841", out)
            self.assertNotIn("9789620756689", out)

            headings_path = artifacts / "two-near" / "inventory" / "headings.json"
            overlap_path = artifacts / "two-near" / "inventory" / "overlap.json"
            self.assertTrue(headings_path.is_file())
            self.assertTrue(overlap_path.is_file())
            headings = json.loads(headings_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(headings), 3)
            for row in headings:
                self.assertIn("section_id", row)
                self.assertIn("heading_path", row)
                self.assertIn("keywords", row)
                self.assertIn("excerpt", row)
                self.assertIn("char_count", row)
                self.assertIn("kind", row)

            eyes = [r for r in headings if r["heading_path"] and "眼睛" in r["heading_path"][-1]]
            self.assertTrue(eyes)
            self.assertIn("eyes", eyes[0]["keywords"])

            dups = []
            extract_dir = artifacts / "two-near" / "extract"
            for path in extract_dir.glob("*.sections.json"):
                for sec in json.loads(path.read_text(encoding="utf-8")):
                    if sec.get("duplicate_of"):
                        dups.append(sec)
            self.assertTrue(dups)
            overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
            self.assertEqual(overlap["num_perm"], 128)
            self.assertEqual(overlap["seed"], 0xC0FFEE)
            self.assertTrue(overlap["components"])

            code2, out2, err2 = _run_main(
                [
                    "two-near",
                    "--book-root",
                    str(book_root),
                    "--artifacts-root",
                    str(artifacts),
                    "--dry-run",
                ]
            )
            self.assertEqual(code2, 0, err2)
            self.assertIn("[inventory]", out2)
            self.assertNotIn("not implemented until a later PR", err2)


if __name__ == "__main__":
    unittest.main()
