"""Extract stage: noise drop, omnibus split, quiz classifier, figures, CLI counts."""

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
from book_combiner.extract import (  # noqa: E402
    content_hash,
    extract_markdown,
    replace_figure_captions,
)

TESTDATA = ROOT / "testdata"
OMNIBUS_MD = TESTDATA / "omnibus-three-books" / "9787510438820.md"
NOISE_MD = TESTDATA / "noise_book.md"


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


def _by_heading(sections) -> dict[str, list]:
    out: dict[str, list] = {}
    for sec in sections:
        key = sec.heading_path[-1] if sec.heading_path else ""
        out.setdefault(key, []).append(sec)
    return out


class TestOmnibusSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = OMNIBUS_MD.read_text(encoding="utf-8")
        cls.result = extract_markdown(cls.text, "9787510438820")

    def test_fixture_has_two_return_toc_and_cip_after_book2_title(self) -> None:
        text = self.text
        self.assertEqual(text.count("返回总目录"), 2)
        self.assertIn("- 返回总目录", text)
        self.assertRegex(text, r"(?m)^返回总目录$")
        # Book-2 CIP sits after the title, not before it.
        title_at = text.index("# 5秒钟洞察人心")
        book2_cip = text.index("ISBN 978-7-5104-3625-3")
        self.assertGreater(book2_cip, title_at)
        next_h1 = text.find("\n# ", title_at + 1)
        self.assertGreater(next_h1, book2_cip)
        # Book-3 CIP is its own H1, then 前言, then 目录 ending in bare 返回总目录, then 上篇.
        book3_cip = text.index("# 图书在版编目（CIP）数据")
        book3_preface = text.index("# 前言")
        shangpian = text.index("# 上篇　面孔微反应")
        bare_toc = text.index("\n返回总目录\n")
        self.assertGreater(book3_preface, book3_cip)
        self.assertGreater(bare_toc, book3_preface)
        self.assertGreater(shangpian, bare_toc)

    def test_splits_into_three_sub_books(self) -> None:
        books = self.result.source_books
        self.assertEqual(len(books), 3, books)
        self.assertEqual(self.result.return_toc_count, 2)
        self.assertEqual(books[0], "如何像读书一样读人：微动作读心术")
        self.assertEqual(books[1], "5秒钟洞察人心")
        self.assertEqual(books[2], "9787510438820#3")
        used = {s.source_book for s in self.result.sections}
        self.assertEqual(used, set(books))
        book3 = [s for s in self.result.sections if s.source_book == books[2]]
        titles = [s.heading_path[-1] for s in book3 if s.heading_path]
        self.assertTrue(any("前言" in t for t in titles), titles)
        self.assertTrue(any("上篇" in t for t in titles), titles)
        # TOC-bullet 返回总目录 must not split book 2.
        book2 = [s for s in self.result.sections if s.source_book == books[1]]
        self.assertTrue(any("前言" in "".join(s.heading_path) for s in book2))
        self.assertTrue(any("第一章" in "".join(s.heading_path) or "视觉型" in "".join(s.heading_path) for s in book2))

    def test_forthcoming_from_master_toc(self) -> None:
        self.assertEqual(
            self.result.forthcoming_books,
            [
                "如何像读书一样读人：微动作读心术",
                "5秒钟洞察人心：微表情读心术",
                "FBI教你解读行为密码：微反应读心术",
            ],
        )

    def test_publisher_note_and_toc_dropped(self) -> None:
        reasons = {d.reason for d in self.result.dropped}
        self.assertIn("toc", reasons)
        self.assertIn("publisher_note", reasons)
        self.assertIn("copyright", reasons)
        joined = "\n".join(s.text for s in self.result.sections)
        self.assertNotIn("此段出版说明应当丢弃", joined)
        self.assertNotIn("ISBN 978-7-5104-3625-3", joined)

    def test_preface_kept(self) -> None:
        kinds = {s.kind for s in self.result.sections if "序言" in "".join(s.heading_path)}
        self.assertTrue(kinds)
        self.assertTrue(all(k == "preface" for k in kinds))


class TestNoiseBook(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = NOISE_MD.read_text(encoding="utf-8")
        cls.result = extract_markdown(cls.text, "noise_book")
        cls.by_h = _by_heading(cls.result.sections)

    def test_toc_cip_copyright_dropped(self) -> None:
        reasons = {d.reason for d in self.result.dropped}
        self.assertIn("toc", reasons)
        self.assertIn("copyright", reasons)
        joined = "\n".join(s.text for s in self.result.sections)
        self.assertNotIn("9787554610220", joined)
        self.assertNotIn("版权所有 侵权必究", joined)
        self.assertNotIn("- 封面", joined)
        self.assertFalse(
            any(
                p and p[-1] in {"目录", "版权信息"}
                for p in (s.heading_path for s in self.result.sections)
            )
        )

    def test_daodu_kept_as_preface(self) -> None:
        secs = self.by_h.get("导读", [])
        self.assertTrue(secs)
        self.assertTrue(all(s.kind == "preface" for s in secs))
        text = "\n".join(s.text for s in secs)
        self.assertIn("销售文案", text)
        keyword = [
            s
            for s in self.result.sections
            if s.heading_path and "本书关键词" in s.heading_path[-1]
        ]
        self.assertTrue(keyword)
        self.assertTrue(all(s.kind == "preface" for s in keyword), [s.kind for s in keyword])
        self.assertTrue(any("0.04秒" in s.text for s in keyword))

    def test_afterword_keeps_instruction_drops_colophon(self) -> None:
        secs = [s for s in self.result.sections if any("后记" in p for p in s.heading_path)]
        self.assertTrue(secs)
        text = "\n".join(s.text for s in secs)
        self.assertIn("7％", text)
        self.assertIn("38％", text)
        self.assertIn("55％", text)
        self.assertNotIn("限于时间和水平", text)
        self.assertNotIn("ISBN 978-7-5190-2331-7", text)
        self.assertNotIn("责任编辑", text)
        self.assertNotIn("职场达人", text)
        self.assertIn("印痕", text)
        self.assertIn("心与身体", text)
        self.assertTrue(all(s.kind == "preface" for s in secs))
        colophon = [d for d in self.result.dropped if d.reason == "colophon"]
        self.assertTrue(colophon)

    def test_glossary_xinli_ceshi_is_body_not_quiz(self) -> None:
        secs = [s for s in self.result.sections if s.heading_path and s.heading_path[-1] == "心理测试"]
        self.assertTrue(secs, "glossary ### 心理测试 should be kept")
        self.assertTrue(all(s.kind == "body" for s in secs), [s.kind for s in secs])
        self.assertTrue(any("比较大的概念" in s.text for s in secs))

    def test_glossary_siblings_are_not_nested(self) -> None:
        tested = [
            s
            for s in self.result.sections
            if s.heading_path and s.heading_path[-1] == "被测试人"
        ]
        self.assertTrue(tested)
        for s in tested:
            self.assertEqual(s.kind, "body")
            self.assertNotIn("心理测试", s.heading_path)
            self.assertEqual(s.heading_path[-1], "被测试人")

    def test_fear_micro_application_is_body(self) -> None:
        secs = [
            s
            for s in self.result.sections
            if s.heading_path and "恐惧微表情的心理测试应用" in s.heading_path[-1]
        ]
        self.assertTrue(secs)
        self.assertTrue(all(s.kind == "body" for s in secs), [s.kind for s in secs])

    def test_quiz_with_abc_is_quiz(self) -> None:
        secs = [
            s
            for s in self.result.sections
            if s.heading_path and "你的情绪稳定吗" in s.heading_path[-1]
        ]
        self.assertTrue(secs)
        self.assertTrue(all(s.kind == "quiz" for s in secs), [s.kind for s in secs])

    def test_quiz_descendants_inherit_quiz(self) -> None:
        wanted = ("测试内容", "问卷项目", "计分方法", "结果分析")
        found = {name: [] for name in wanted}
        for s in self.result.sections:
            if not s.heading_path:
                continue
            leaf = s.heading_path[-1]
            for name in wanted:
                if name in leaf:
                    found[name].append(s)
        for name in wanted:
            self.assertTrue(found[name], f"missing {name}")
            self.assertTrue(
                all(s.kind == "quiz" for s in found[name]),
                f"{name} kinds={[s.kind for s in found[name]]} paths={[s.heading_path for s in found[name]]}",
            )
            self.assertTrue(
                all(any("心理测试" in p for p in s.heading_path) for s in found[name]),
                name,
            )

    def test_skill_test_headings(self) -> None:
        titles = [s.heading_path[-1] for s in self.result.sections if s.heading_path]
        self.assertTrue(any("表情辨识能力测试" in t for t in titles))
        skill = [s for s in self.result.sections if s.kind == "skill_test"]
        self.assertTrue(skill)

    def test_figure_caption_placeholder(self) -> None:
        joined = "\n".join(s.text for s in self.result.sections)
        self.assertIn(
            "*[Figure omitted from source conversion: 图2.1　皱眉的例子]*",
            joined,
        )
        self.assertNotIn("**图2.1", joined)
        self.assertIn("皱眉肌位于眉毛内侧", joined)

    def test_bibliography_kind(self) -> None:
        biblio = [s for s in self.result.sections if s.kind == "bibliography"]
        self.assertTrue(biblio)
        titles = {s.heading_path[-1] for s in biblio if s.heading_path}
        self.assertTrue({"参考书目", "索引"} & titles)


class TestFiguresAndHash(unittest.TestCase):
    def test_replace_figure_captions(self) -> None:
        src = "**图1.1　紧张的女商人**\n她很紧张。\n"
        out = replace_figure_captions(src)
        self.assertEqual(
            out.splitlines()[0],
            "*[Figure omitted from source conversion: 图1.1　紧张的女商人]*",
        )
        self.assertIn("她很紧张。", out)

    def test_content_hash_strips_whitespace_and_punct(self) -> None:
        a = content_hash("额肌，左右对称。")
        b = content_hash("额肌  左右对称")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)


class TestOversizedLeaf(unittest.TestCase):
    def test_splits_on_paragraphs_with_overlap(self) -> None:
        paras = [f"段落{i}：" + ("字" * 80) for i in range(8)]
        body = "\n\n".join(paras)
        md = f"## 第四节 激动时，手会冒汗或者颤抖吗？\n\n{body}\n"
        result = extract_markdown(md, "sweat", max_input_chars=200)
        chunks = [s for s in result.sections if s.heading_path]
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all("#p" in s.id for s in chunks))
        self.assertTrue(all(s.heading_path[-1].startswith("第四节") for s in chunks))
        # Overlap: last paragraph of chunk i reappears as context in chunk i+1.
        cap = 200
        for i in range(len(chunks) - 1):
            prev_paras = [p for p in chunks[i].text.split("\n\n") if p.strip()]
            next_paras = [p for p in chunks[i + 1].text.split("\n\n") if p.strip()]
            self.assertEqual(prev_paras[-1], next_paras[0])
        for ch in chunks:
            paras = [p for p in ch.text.split("\n\n") if p.strip()]
            if len(paras) == 1:
                continue
            self.assertLessEqual(len(ch.text), cap)

    def test_overlap_does_not_exceed_cap(self) -> None:
        a = "甲" * 2500
        b = "乙" * 2500
        md = f"## 大节\n\n{a}\n\n{b}\n"
        result = extract_markdown(md, "big", max_input_chars=4000)
        chunks = [s for s in result.sections if s.heading_path]
        self.assertGreaterEqual(len(chunks), 2)
        for ch in chunks:
            paras = [p for p in ch.text.split("\n\n") if p.strip()]
            if len(paras) == 1:
                continue
            self.assertLessEqual(len(ch.text), 4000)


class TestHierarchyRepair(unittest.TestCase):
    def test_chapter_hash_depth_not_trusted(self) -> None:
        md = """# 目录

## 第一章 不知道微表情，你就“Out”了
- 第一节 表情
## 第二章 会说话的脸
- 第一节 鼻子

# 第一章 不知道微表情，你就“Out”了

微表情读心术真的这么神奇吗？只要你学会观察，它就不再深不可测了。这是足够长的正文。

## 第二章 会说话的脸

相由心生，一个人的性情可以从脸上看出来。鼻子与眼睛一样可以看出性格。

### 第一节 最性情的鼻子

嗤之以鼻是形容轻视的情绪，这段正文用于确认节的层级。
"""
        result = extract_markdown(md, "levels")
        ch2 = [
            s
            for s in result.sections
            if s.heading_path and compact_join(s.heading_path).find("第二章") >= 0
        ]
        self.assertTrue(ch2)
        for s in ch2:
            self.assertEqual(s.heading_path[0].startswith("第二章") or "第二章" in s.heading_path[0], True)
            self.assertLessEqual(s.level, 2 if "节" not in "".join(s.heading_path) else 3)

    def test_promoted_h1_shares_path_with_nested_heading(self) -> None:
        md = """# 目录

- 第二章　眉眼真的会说话
  - 眉毛告诉你的事

# 第二章　眉眼真的会说话

眉毛可以反映出一个人的心理状态。下面这段足够长，用来结束目录扫描。

## 眉毛告诉你的事

一旦情绪大爆发，眉毛就会变得跳跃起来。继续写一些说明文字凑够长度。

### 愤怒时的眉毛

愤怒时的眉毛比较容易辨识，眉头向眉心集中，同时下压。

---

# 愤怒时的眉毛

在准备对付对方的时候，当事者会发挥极大的能量。额肌和皱眉肌同时作用。
"""
        result = extract_markdown(md, "brows")
        angry = [
            s
            for s in result.sections
            if s.heading_path and s.heading_path[-1] == "愤怒时的眉毛"
        ]
        self.assertTrue(angry)
        paths = {tuple(s.heading_path) for s in angry}
        self.assertEqual(len(paths), 1, paths)
        path = angry[0].heading_path
        self.assertGreaterEqual(len(path), 2)
        self.assertTrue(any("第二章" in p for p in path))
        joined = "\n".join(s.text for s in angry)
        self.assertIn("眉头向眉心集中", joined)
        self.assertIn("额肌和皱眉肌", joined)


def compact_join(path: list[str]) -> str:
    return ">".join(path)


class TestCliExtractAndDryRun(unittest.TestCase):
    def test_stage_extract_prints_section_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts"
            code, out, err = _run_main(
                [
                    "omnibus-three-books",
                    "--book-root",
                    str(TESTDATA),
                    "--artifacts-root",
                    str(artifacts),
                    "--stage",
                    "extract",
                ]
            )
            self.assertEqual(code, 0, err)
            self.assertIn("[discover]", out)
            self.assertIn("[extract]", out)
            self.assertIn("sections=", out)
            self.assertIn("omnibus_books=3", out)
            sections_path = artifacts / "omnibus-three-books" / "extract" / "9787510438820.sections.json"
            dropped_path = artifacts / "omnibus-three-books" / "extract" / "9787510438820.dropped.json"
            self.assertTrue(sections_path.is_file())
            self.assertTrue(dropped_path.is_file())
            sections = json.loads(sections_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len({s["source_book"] for s in sections}), 3)

    def test_dry_run_runs_extract_and_prints_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book_root = Path(tmp) / "book"
            topic = book_root / "noise-book"
            topic.mkdir(parents=True)
            (topic / "noise_book.md").write_text(NOISE_MD.read_text(encoding="utf-8"), encoding="utf-8")
            artifacts = Path(tmp) / "artifacts"
            code, out, err = _run_main(
                [
                    "noise-book",
                    "--book-root",
                    str(book_root),
                    "--artifacts-root",
                    str(artifacts),
                    "--dry-run",
                ]
            )
            self.assertEqual(code, 0, err)
            self.assertIn("[extract]", out)
            self.assertIn("sections=", out)
            self.assertNotIn("not implemented until a later PR", err)
            data = json.loads(
                (artifacts / "noise-book" / "extract" / "noise_book.sections.json").read_text(
                    encoding="utf-8"
                )
            )
            kinds = {s["kind"] for s in data}
            self.assertIn("preface", kinds)
            self.assertIn("quiz", kinds)
            self.assertIn("body", kinds)


if __name__ == "__main__":
    unittest.main()
