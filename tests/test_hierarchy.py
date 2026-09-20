"""Unit tests for TOC-guided heading repair."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from book_combiner.hierarchy import (  # noqa: E402
    compact_key,
    is_magazine_quiz_heading,
    parse_toc_entries,
    repair_heading_paths,
    structural_level,
)


class TestHierarchy(unittest.TestCase):
    def test_chapter_overrides_hash_depth(self) -> None:
        self.assertEqual(structural_level("第二章 会说话的脸", 2), 1)
        self.assertEqual(structural_level("第一节 最性情的鼻子", 3), 2)
        self.assertEqual(structural_level("1.1 普通表情的辨识", 1), 2)

    def test_toc_then_numbering_then_short_h1(self) -> None:
        toc = parse_toc_entries(
            [
                "# 目录",
                "- 第二章　眉眼真的会说话",
                "  - 眉毛告诉你的事",
            ]
        )
        paths = repair_heading_paths(
            [
                (1, "第二章　眉眼真的会说话"),
                (2, "眉毛告诉你的事"),
                (3, "愤怒时的眉毛"),
                (1, "愤怒时的眉毛"),
            ],
            toc,
        )
        self.assertEqual(compact_key(paths[0][0]), compact_key("第二章　眉眼真的会说话"))
        self.assertEqual(paths[2], paths[3])
        self.assertEqual(paths[3][-1], "愤怒时的眉毛")
        self.assertGreaterEqual(len(paths[3]), 2)

    def test_compact_key_ignores_spaces(self) -> None:
        self.assertEqual(compact_key("导 读"), compact_key("导读"))
        self.assertEqual(compact_key("第一章　面部"), compact_key("第一章 面部"))

    def test_magazine_quiz_heading(self) -> None:
        self.assertTrue(is_magazine_quiz_heading("心理测试　你的情绪稳定吗"))
        self.assertTrue(is_magazine_quiz_heading("心理测试：测测你有多乐观"))
        self.assertFalse(is_magazine_quiz_heading("心理测试"))
        self.assertFalse(is_magazine_quiz_heading("3. 恐惧微表情的心理测试应用"))

    def test_glossary_does_not_swallow_siblings(self) -> None:
        paths = repair_heading_paths(
            [
                (2, "第一章 面部"),
                (3, "心理测试"),
                (3, "被测试人"),
                (2, "第4章　悲伤的微表情"),
            ]
        )
        self.assertEqual(paths[1][-1], "心理测试")
        self.assertEqual(paths[2][-1], "被测试人")
        self.assertNotIn("心理测试", paths[2][:-1])
        self.assertEqual(paths[1][:-1], paths[2][:-1])
        self.assertTrue(paths[3][0].startswith("第4章"))


if __name__ == "__main__":
    unittest.main()
