"""Frozen coverage extractors: sweat catalogues, names, numbers, muscles."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from book_combiner.coverage import (  # noqa: E402
    extract_tokens,
    flatten_tokens,
    missing_tokens,
    sweat_compounds,
)

SWEAT_CATALOGUE = "温热性出汗、精神性出汗、味觉性出汗和运动性出汗"
SWEAT_FOUR = ["温热性出汗", "精神性出汗", "味觉性出汗", "运动性出汗"]


class TestSweatCompounds(unittest.TestCase):
    def test_l1196_catalogue_four_bare_type_names(self) -> None:
        self.assertEqual(sweat_compounds(SWEAT_CATALOGUE), SWEAT_FOUR)
        self.assertNotIn("和运动性出汗", sweat_compounds(SWEAT_CATALOGUE))

    def test_full_sentence_does_not_emit_conjunction_prefix(self) -> None:
        sentence = (
            "出汗根据刺激的种类可分为三种，即"
            + SWEAT_CATALOGUE
            + "。"
        )
        found = sweat_compounds(sentence)
        self.assertIn("精神性出汗", found)
        self.assertIn("味觉性出汗", found)
        self.assertIn("运动性出汗", found)
        self.assertNotIn("和运动性出汗", found)
        tokens = flatten_tokens(extract_tokens(SWEAT_CATALOGUE + "。"))
        for name in SWEAT_FOUR:
            self.assertIn(name, tokens)
        self.assertNotIn("和运动性出汗", tokens)


class TestFrozenExtractors(unittest.TestCase):
    def test_names_require_middle_dot_not_jiang(self) -> None:
        tokens = extract_tokens("姜振宇认为微反应不同于微表情，保罗·艾克曼提出了FACS。")
        self.assertNotIn("姜振宇", tokens.names)
        self.assertIn("保罗·艾克曼", tokens.names)
        self.assertIn("FACS", tokens.names)

    def test_single_digit_list_indices_are_not_numbers(self) -> None:
        tokens = extract_tokens("1. 第一项说明\n2. 第二项说明\n只有10%的人可以察觉，时值1/25秒。")
        self.assertNotIn("1", tokens.numbers)
        self.assertNotIn("2", tokens.numbers)
        self.assertIn("10%", tokens.numbers)
        self.assertIn("1/25", tokens.numbers)

    def test_figure_placeholder_is_not_a_name_token(self) -> None:
        text = "眉毛上扬。\n\n*[Figure omitted from source conversion: 图1.1　紧张的女商人]*\n\nFACS 编码。"
        tokens = extract_tokens(text)
        self.assertNotIn("Figure", tokens.names)
        self.assertIn("FACS", tokens.names)

    def test_outline_section_numbers_are_not_coverage_numbers(self) -> None:
        text = "10.2 眉毛\n10.2.1 上扬\n0.04秒的微表情，约10%的人能察觉。"
        tokens = extract_tokens(text)
        self.assertNotIn("10.2", tokens.numbers)
        self.assertNotIn("10.2.1", tokens.numbers)
        self.assertIn("0.04", tokens.numbers)
        self.assertIn("10%", tokens.numbers)

    def test_outline_numbers_in_bold_are_not_required(self) -> None:
        tokens = flatten_tokens(extract_tokens("**10.2.1** 眉毛上扬。参考书目见文末。"))
        self.assertNotIn("10.2.1", tokens)
        self.assertNotIn("参考书目", tokens)

    def test_partial_muscle_spans_are_not_tokens(self) -> None:
        tokens = extract_tokens("只由两块肌和令这两组肌发会造成肌肉紧张，颧大肌和眼轮匝肌除外。")
        self.assertNotIn("两块肌", tokens.muscle_or_au)
        self.assertNotIn("这两组肌", tokens.muscle_or_au)
        self.assertNotIn("发会造成肌", tokens.muscle_or_au)
        self.assertIn("颧大肌", tokens.muscle_or_au)
        self.assertIn("眼轮匝肌", tokens.muscle_or_au)

    def test_traditional_muscle_token_matches_simplified_output(self) -> None:
        required = flatten_tokens(extract_tokens("鼻子周圍肌肉也会抽动。"))
        self.assertTrue(any("肌" in t for t in required))
        missing = missing_tokens("鼻子周围肌肉也会抽动。", required)
        self.assertEqual(missing, [])

    def test_bold_and_muscle(self) -> None:
        tokens = extract_tokens("额肌左右对称。**皱眉肌**与眼轮匝肌一起工作，AU12 上提。")
        self.assertIn("皱眉肌", tokens.bold_terms)
        self.assertIn("额肌", tokens.muscle_or_au)
        self.assertIn("眼轮匝肌", tokens.muscle_or_au)
        self.assertIn("AU12", tokens.muscle_or_au)


if __name__ == "__main__":
    unittest.main()
