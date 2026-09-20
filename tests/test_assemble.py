"""Assemble ZH/EN/HK from node artifacts: TOC, appendices, atomic write, checks."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from book_combiner.assemble import (  # noqa: E402
    APPENDIX_A,
    AssembleError,
    BLOB_A,
    BLOB_B,
    cross_ref_line,
    format_appendix_a_table,
    format_front_matter,
    heading_levels,
    heading_signature,
    outline_heading_levels,
    prepare_appendices,
    retarget_headings,
    run_assemble,
    strip_cjk_from_headings,
    toc_follows_title,
    yue_heading_for,
)
from book_combiner.translate import evaluate_cantonese, traditionalize  # noqa: E402
from test_outline import (  # noqa: E402
    SIMPLE_OUTLINE_JSON,
    TESTDATA,
    FakeLLMClient,
    _run_main,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestAppendixA(unittest.TestCase):
    def test_table_is_deterministic(self) -> None:
        manifest = {
            "sources": [
                {"stem": "b-book", "bytes": 2, "sha256": "bb"},
                {"stem": "a-book", "bytes": 1, "sha256": "aa"},
            ]
        }
        table = format_appendix_a_table(manifest)
        lines = [ln for ln in table.splitlines() if ln.startswith("|") and "stem" not in ln and "---" not in ln]
        self.assertEqual(lines[0], "| a-book | 1 | aa |")
        self.assertEqual(lines[1], "| b-book | 2 | bb |")
        self.assertEqual(table, format_appendix_a_table(manifest))


class TestHeadingNesting(unittest.TestCase):
    def test_retarget_preserves_relative_levels(self) -> None:
        md = "### 眼睛\n\n#### 瞳孔\n\ntext\n"
        out = retarget_headings(md, 4)
        self.assertIn("#### 眼睛", out)
        self.assertIn("##### 瞳孔", out)


class TestCrossRef(unittest.TestCase):
    def test_templates_and_yue_fallback(self) -> None:
        self.assertEqual(cross_ref_line("zh", node_id="s-eyes", title_en="Eyes", title_zh="眼睛", yue_heading="眼"), "见眼睛（s-eyes）。")
        self.assertEqual(cross_ref_line("en", node_id="s-eyes", title_en="Eyes", title_zh="眼睛", yue_heading="眼"), "See Eyes (s-eyes).")
        self.assertEqual(cross_ref_line("yue", node_id="s-eyes", title_en="Eyes", title_zh="眼睛", yue_heading="眼睛"), "見眼睛（s-eyes）。")
        with tempfile.TemporaryDirectory() as tmp:
            topic_dir = Path(tmp)
            yue_path = topic_dir / "translate" / "yue" / "s-eyes.md"
            _write(yue_path, "### 眼眸\n\n眼睛係窗口。\n")
            self.assertEqual(yue_heading_for(topic_dir, "s-eyes", "眼睛"), "眼眸")
            self.assertEqual(yue_heading_for(topic_dir, "missing", "微反应"), traditionalize("微反应"))
            self.assertEqual(traditionalize("微反应"), "微反應")


class TestFrontMatter(unittest.TestCase):
    def test_en_and_hk_fields(self) -> None:
        en = format_front_matter(
            topic="facial-expression",
            title="Facial Expression",
            language="en",
            register=None,
            source_files=["book/facial-expression/a.md"],
            generated_at="2026-09-18T00:00:00Z",
            model="grok-4",
        )
        self.assertIn("language: en", en)
        self.assertIn("register: null", en)
        self.assertIn("combiner_version: 0.1.0", en)
        hk = format_front_matter(
            topic="facial-expression",
            title="面部表情",
            language="yue-Hant-HK",
            register="hk-written-cantonese",
            source_files=["book/facial-expression/a.md"],
            generated_at="2026-09-18T00:00:00Z",
            model="grok-4",
        )
        self.assertIn("language: yue-Hant-HK", hk)
        self.assertIn("register: hk-written-cantonese", hk)


class TestAssembleTwoBook(unittest.TestCase):
    def test_cli_assemble_writes_three_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts"
            book_root = Path(tmp) / "books"
            src = TESTDATA / "two-book-overlap"
            dest = book_root / "two-book-overlap"
            dest.mkdir(parents=True)
            for md in src.glob("*.md"):
                (dest / md.name).write_text(md.read_text(encoding="utf-8"), encoding="utf-8")
            fake = FakeLLMClient(default=SIMPLE_OUTLINE_JSON)
            code, out, err = _run_main(
                [
                    "two-book-overlap",
                    "--book-root",
                    str(book_root),
                    "--artifacts-root",
                    str(artifacts),
                    "--from-stage",
                    "outline",
                ],
                llm_client=fake,
            )
            self.assertEqual(code, 0, err)
            self.assertIn("[translate-en]", out)
            self.assertIn("[translate-yue]", out)
            self.assertIn("[assemble]", out)
            zh = artifacts / "two-book-overlap" / "compiled.zh.md"
            en = book_root / "two-book-overlap.md"
            hk = book_root / "two-book-overlap.hk.md"
            self.assertTrue(zh.is_file())
            self.assertTrue(en.is_file())
            self.assertTrue(hk.is_file())
            zh_text = zh.read_text(encoding="utf-8")
            en_text = en.read_text(encoding="utf-8")
            hk_text = hk.read_text(encoding="utf-8")
            self.assertNotIn("---\n", zh_text[:20])
            self.assertTrue(en_text.startswith("---\n"))
            self.assertIn("language: en", en_text)
            self.assertIn("register: null", en_text)
            self.assertIn("language: yue-Hant-HK", hk_text)
            self.assertIn("register: hk-written-cantonese", hk_text)
            self.assertEqual(heading_levels(zh_text), heading_levels(en_text))
            self.assertEqual(heading_levels(zh_text), heading_levels(hk_text))
            self.assertTrue(toc_follows_title(en_text, "Contents"))
            self.assertTrue(toc_follows_title(hk_text, "目錄"))
            self.assertTrue(toc_follows_title(zh_text, "目录"))
            self.assertIn(APPENDIX_A["en"], en_text)
            self.assertIn("| book-a |", en_text)
            self.assertIn("sha256", en_text)
            self.assertIn("附录 B", zh_text)
            self.assertIn("Appendix B", en_text)
            self.assertIn("心理测试", zh_text)
            ok, reasons = evaluate_cantonese(hk_text)
            self.assertTrue(ok, reasons)
            translate_calls = [c for c in fake.calls if c["stage"].startswith("translate")]
            self.assertTrue(translate_calls)

    def test_empty_appendix_omitted_from_toc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            topic = artifacts / "demo"
            outline = json.loads(SIMPLE_OUTLINE_JSON)
            _write(topic / "outline" / "outline.json", json.dumps(outline, ensure_ascii=False))
            _write(
                topic / "outline" / "assignment.json",
                json.dumps({"assignments": []}, ensure_ascii=False),
            )
            _write(
                topic / "manifest.json",
                json.dumps(
                    {
                        "topic": "demo",
                        "sources": [
                            {
                                "path": str(TESTDATA / "two-book-overlap" / "book-a.md"),
                                "stem": "book-a",
                                "sha256": "aa",
                                "bytes": 10,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
            )
            eyes = "### 眼睛\n\n眼睛是心灵的窗口。瞳孔放大表示兴趣。\n"
            hands = "### 握手\n\n握手时掌心出汗属于精神性出汗。\n"
            _write(topic / "merge" / "nodes" / "s-eyes.zh.md", eyes)
            _write(topic / "merge" / "nodes" / "s-handshake.zh.md", hands)
            _write(topic / "merge" / "nodes" / "s-eyes.meta.json", json.dumps({
                "unique_leaf_chars": 40, "remaining_overlap": 0.4
            }))
            _write(topic / "merge" / "nodes" / "s-handshake.meta.json", json.dumps({
                "unique_leaf_chars": 30, "remaining_overlap": 0.4
            }))
            from test_outline import fake_en_from_markdown, fake_yue_from_markdown

            _write(topic / "translate" / "en" / "s-eyes.md", fake_en_from_markdown(eyes))
            _write(topic / "translate" / "en" / "s-handshake.md", fake_en_from_markdown(hands))
            _write(topic / "translate" / "yue" / "s-eyes.md", fake_yue_from_markdown(eyes))
            _write(topic / "translate" / "yue" / "s-handshake.md", fake_yue_from_markdown(hands))
            written = prepare_appendices(topic="demo", artifacts_root=artifacts, strict_topic=False)
            self.assertIn(BLOB_A, written)
            self.assertNotIn(BLOB_B, written)
            book_root = Path(tmp) / "book"
            book_root.mkdir()
            paths = run_assemble(
                topic="demo",
                book_root=book_root,
                artifacts_root=artifacts,
                model="fake-model",
            )
            en_text = paths["en"].read_text(encoding="utf-8")
            self.assertIn("Appendix A", en_text)
            self.assertNotIn("Appendix B", en_text)
            toc = en_text.split("## Contents", 1)[1].split("\n\n", 1)[0]
            self.assertNotIn("Quizzes", toc)
            zh_text = paths["zh"].read_text(encoding="utf-8")
            levels_titles = heading_signature(zh_text)
            # Part ##, chapter ###, section #### — not chapter/section both ###.
            self.assertIn((2, "线索"), levels_titles)
            self.assertIn((3, "信号"), levels_titles)
            self.assertIn((4, "眼睛"), levels_titles)
            self.assertNotIn((3, "眼睛"), levels_titles)

    def test_yue_container_heading_and_part_chapter_crossrefs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            topic = artifacts / "demo"
            outline = json.loads(SIMPLE_OUTLINE_JSON)
            _write(topic / "outline" / "outline.json", json.dumps(outline, ensure_ascii=False))
            _write(
                topic / "outline" / "assignment.json",
                json.dumps(
                    {
                        "assignments": [
                            {
                                "section_id": "a:1:2",
                                "primary_node_id": "s-eyes",
                                "secondary_node_id": "c-cues",
                            },
                            {
                                "section_id": "a:3:4",
                                "primary_node_id": "s-handshake",
                                "secondary_node_id": "p-cues",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
            )
            _write(
                topic / "manifest.json",
                json.dumps(
                    {
                        "topic": "demo",
                        "sources": [
                            {
                                "path": str(TESTDATA / "two-book-overlap" / "book-a.md"),
                                "stem": "book-a",
                                "sha256": "aa",
                                "bytes": 10,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
            )
            eyes = "### 眼睛\n\n#### 瞳孔\n\n眼睛是心灵的窗口。\n"
            hands = "### 握手\n\n握手时掌心出汗属于精神性出汗。\n"
            _write(topic / "merge" / "nodes" / "s-eyes.zh.md", eyes)
            _write(topic / "merge" / "nodes" / "s-handshake.zh.md", hands)
            _write(
                topic / "merge" / "nodes" / "s-eyes.meta.json",
                json.dumps({"unique_leaf_chars": 40, "remaining_overlap": 0.4}),
            )
            _write(
                topic / "merge" / "nodes" / "s-handshake.meta.json",
                json.dumps({"unique_leaf_chars": 30, "remaining_overlap": 0.4}),
            )
            from test_outline import fake_en_from_markdown, fake_yue_from_markdown

            _write(topic / "translate" / "en" / "s-eyes.md", fake_en_from_markdown(eyes))
            _write(topic / "translate" / "en" / "s-handshake.md", fake_en_from_markdown(hands))
            _write(topic / "translate" / "yue" / "s-eyes.md", fake_yue_from_markdown(eyes))
            _write(topic / "translate" / "yue" / "s-handshake.md", fake_yue_from_markdown(hands))
            _write(topic / "translate" / "yue" / "c-cues.md", "### 訊號\n\n章節導言係粵語。\n")
            book_root = Path(tmp) / "book"
            book_root.mkdir()
            paths = run_assemble(
                topic="demo",
                book_root=book_root,
                artifacts_root=artifacts,
                model="fake-model",
            )
            zh_text = paths["zh"].read_text(encoding="utf-8")
            en_text = paths["en"].read_text(encoding="utf-8")
            hk_text = paths["hk"].read_text(encoding="utf-8")
            self.assertIn("见握手（s-handshake）。", zh_text)
            self.assertIn("见眼睛（s-eyes）。", zh_text)
            self.assertIn("See Handshake (s-handshake).", en_text)
            self.assertIn("See Eyes (s-eyes).", en_text)
            self.assertIn("### 訊號", hk_text)
            self.assertNotIn("### 信號", hk_text)
            self.assertIn((4, "眼睛"), heading_signature(zh_text))
            self.assertIn((5, "瞳孔"), heading_signature(zh_text))


class TestAtomicReplace(unittest.TestCase):
    def test_validation_failure_leaves_previous_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            topic = artifacts / "demo"
            outline = json.loads(SIMPLE_OUTLINE_JSON)
            _write(topic / "outline" / "outline.json", json.dumps(outline, ensure_ascii=False))
            _write(topic / "outline" / "assignment.json", json.dumps({"assignments": []}))
            _write(
                topic / "manifest.json",
                json.dumps(
                    {
                        "topic": "demo",
                        "sources": [
                            {
                                "path": str(TESTDATA / "two-book-overlap" / "book-a.md"),
                                "stem": "book-a",
                                "sha256": "aa",
                                "bytes": 10,
                            }
                        ],
                    }
                ),
            )
            _write(topic / "merge" / "nodes" / "s-eyes.zh.md", "### 眼睛\n\n好。\n")
            _write(topic / "merge" / "nodes" / "s-handshake.zh.md", "### 握手\n\n好。\n")
            _write(topic / "merge" / "nodes" / "s-eyes.FAILED.md", "FAILED\n")
            _write(topic / "translate" / "en" / "s-eyes.md", "### Eyes\n\nOk.\n")
            _write(topic / "translate" / "en" / "s-handshake.md", "### Hands\n\nOk.\n")
            _write(topic / "translate" / "yue" / "s-eyes.md", "### 眼睛\n\n眼睛係窗口。唔好交叉。\n")
            _write(topic / "translate" / "yue" / "s-handshake.md", "### 握手\n\n握手係線索。\n")
            book_root = Path(tmp) / "book"
            book_root.mkdir()
            previous = "PREVIOUS GOOD OUTPUT\n"
            en_path = book_root / "demo.md"
            hk_path = book_root / "demo.hk.md"
            zh_path = topic / "compiled.zh.md"
            en_path.write_text(previous, encoding="utf-8")
            hk_path.write_text(previous, encoding="utf-8")
            zh_path.write_text(previous, encoding="utf-8")
            with self.assertRaises(AssembleError):
                run_assemble(
                    topic="demo",
                    book_root=book_root,
                    artifacts_root=artifacts,
                    model="fake-model",
                )
            self.assertEqual(en_path.read_text(encoding="utf-8"), previous)
            self.assertEqual(hk_path.read_text(encoding="utf-8"), previous)
            self.assertEqual(zh_path.read_text(encoding="utf-8"), previous)
            self.assertTrue((en_path.with_name(en_path.name + ".partial")).is_file())

    def test_quizzes_concatenated_not_merged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            topic = artifacts / "quiztopic"
            extract = topic / "extract"
            extract.mkdir(parents=True)
            sections = [
                {
                    "id": "a:1:2",
                    "source_stem": "a",
                    "source_book": "a",
                    "heading_path": ["心理测试　情绪稳定吗"],
                    "level": 2,
                    "start_line": 1,
                    "end_line": 10,
                    "text": "A. 是\nB. 否\nC. 不确定\n",
                    "kind": "quiz",
                    "char_count": 20,
                    "content_hash": "h1",
                    "duplicate_of": None,
                },
                {
                    "id": "b:1:2",
                    "source_stem": "b",
                    "source_book": "b",
                    "heading_path": ["心理测试　易怒指数"],
                    "level": 2,
                    "start_line": 1,
                    "end_line": 8,
                    "text": "A. 高\nB. 中\nC. 低\n",
                    "kind": "quiz",
                    "char_count": 16,
                    "content_hash": "h2",
                    "duplicate_of": None,
                },
            ]
            _write(extract / "a.sections.json", json.dumps(sections, ensure_ascii=False))
            written = prepare_appendices(topic="quiztopic", artifacts_root=artifacts)
            self.assertIn(BLOB_B, written)
            blob = (topic / "merge" / "appendices" / f"{BLOB_B}.zh.md").read_text(encoding="utf-8")
            self.assertIn("情绪稳定吗", blob)
            self.assertIn("易怒指数", blob)
            self.assertIn("A. 是", blob)
            self.assertIn("A. 高", blob)


class TestAssembleLiveUnblock(unittest.TestCase):
    def test_strip_cjk_from_english_headings(self) -> None:
        text = "# Face 面部\n\n## Eyes 眼睛\n\nBody text 还在。\n"
        out = strip_cjk_from_headings(text)
        self.assertIn("# Face", out)
        self.assertIn("## Eyes", out)
        self.assertNotIn("# Face 面部", out)
        self.assertIn("Body text 还在。", out)

    def test_outline_heading_levels_ignore_h4(self) -> None:
        zh = "# T\n\n## Contents\n\n## Part\n\n### Ch\n\n#### Extra ZH\n"
        en = "# T\n\n## Contents\n\n## Part\n\n### Ch\n"
        self.assertNotEqual(heading_levels(zh), heading_levels(en))
        self.assertEqual(outline_heading_levels(zh), outline_heading_levels(en))

    def test_missing_outline_leaf_does_not_fail_assemble(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            topic = artifacts / "demo"
            outline = json.loads(SIMPLE_OUTLINE_JSON)
            outline["parts"][0]["chapters"][0]["sections"].append(
                {"id": "s-missing", "title_en": "Missing", "title_zh": "缺失"}
            )
            _write(topic / "outline" / "outline.json", json.dumps(outline, ensure_ascii=False))
            _write(topic / "outline" / "assignment.json", json.dumps({"assignments": []}))
            _write(
                topic / "manifest.json",
                json.dumps(
                    {
                        "topic": "demo",
                        "sources": [
                            {
                                "path": str(TESTDATA / "two-book-overlap" / "book-a.md"),
                                "stem": "book-a",
                                "sha256": "aa",
                                "bytes": 10,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
            )
            eyes = "### 眼睛\n\n瞳孔放大。\n"
            hands = "### 握手\n\n掌心出汗。\n"
            from test_outline import fake_en_from_markdown, fake_yue_from_markdown

            _write(topic / "merge" / "nodes" / "s-eyes.zh.md", eyes)
            _write(topic / "merge" / "nodes" / "s-handshake.zh.md", hands)
            _write(
                topic / "merge" / "nodes" / "s-eyes.meta.json",
                json.dumps({"unique_leaf_chars": 40, "remaining_overlap": 0.4}),
            )
            _write(
                topic / "merge" / "nodes" / "s-handshake.meta.json",
                json.dumps({"unique_leaf_chars": 30, "remaining_overlap": 0.4}),
            )
            _write(topic / "translate" / "en" / "s-eyes.md", fake_en_from_markdown(eyes))
            _write(topic / "translate" / "en" / "s-handshake.md", fake_en_from_markdown(hands))
            yue_pad = "呢段係書面粵語，唔係普通話，說明嘅內容要保留。\n"
            _write(
                topic / "translate" / "yue" / "s-eyes.md",
                fake_yue_from_markdown(eyes) + yue_pad,
            )
            _write(
                topic / "translate" / "yue" / "s-handshake.md",
                fake_yue_from_markdown(hands) + yue_pad,
            )
            book_root = Path(tmp) / "book"
            book_root.mkdir()
            paths = run_assemble(
                topic="demo",
                book_root=book_root,
                artifacts_root=artifacts,
                model="fake-model",
            )
            self.assertTrue(paths["en"].is_file())
            self.assertTrue(paths["hk"].is_file())
            self.assertFalse((topic / "merge" / "nodes" / "s-missing.zh.md").is_file())


class TestAssembleFailedNode(unittest.TestCase):
    def test_failed_node_blocks_assemble(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            topic = artifacts / "demo"
            _write(topic / "outline" / "outline.json", SIMPLE_OUTLINE_JSON)
            _write(topic / "outline" / "assignment.json", json.dumps({"assignments": []}))
            _write(
                topic / "manifest.json",
                json.dumps(
                    {
                        "sources": [
                            {
                                "path": str(TESTDATA / "two-book-overlap" / "book-a.md"),
                                "stem": "book-a",
                                "bytes": 1,
                                "sha256": "x",
                            }
                        ]
                    }
                ),
            )
            _write(topic / "merge" / "nodes" / "s-eyes.FAILED.md", "nope\n")
            book_root = Path(tmp) / "out"
            book_root.mkdir()
            with self.assertRaises(AssembleError) as ctx:
                run_assemble(topic="demo", book_root=book_root, artifacts_root=artifacts)
            self.assertIn("FAILED", str(ctx.exception))
            self.assertFalse((book_root / "demo.md").is_file())


if __name__ == "__main__":
    unittest.main()
