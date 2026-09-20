"""Translate stage and 書面粵語 / EN Han-density checks."""

from __future__ import annotations

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

from book_combiner.translate import (  # noqa: E402
    HAN_DENSITY_CAP,
    STAGE_EN,
    STAGE_YUE,
    SIMPLIFIED_ONLY,
    TECH_TERM_S2T,
    TRANSLATE_YUE_PROMPT_PATH,
    en_han_density,
    evaluate_cantonese,
    split_translate_chunks,
    traditionalize,
    translate_markdown,
)
from book_combiner.models import CacheRecord  # noqa: E402
from test_outline import FakeLLMClient  # noqa: E402

GOLD_YUE = ROOT / "testdata" / "cantonese" / "written_yue.md"
GOLD_MANDARIN = ROOT / "testdata" / "cantonese" / "trad_mandarin.md"

FEWSHOTS = (
    ("微表情是人们日益关注的一个新词。", "微表情係近年嚟愈嚟愈多人關注嘅一個新詞。"),
    ("只有10%的人可以察觉到微表情变化。", "只有10%嘅人可以察覺到微表情變化。"),
    ("眼睛是心灵的窗口。", "眼睛係心靈嘅窗口。"),
    ("不要交叉手臂。", "唔好交叉手臂。"),
    ("他说：“我很好。”", "佢講：「我很好。」"),
    ("保罗·艾克曼提出了FACS。", "保羅·艾克曼提出咗FACS。"),
    ("额肌左右对称，位于眉毛上方。", "額肌左右對稱，喺眉毛上方。"),
    ("微反应并不仅仅是面部微表情。", "微反應唔單單係面部微表情。"),
)


class TestCantoneseGoldFiles(unittest.TestCase):
    def test_written_yue_passes_without_llm(self) -> None:
        text = GOLD_YUE.read_text(encoding="utf-8")
        self.assertGreaterEqual(len(text), 360)
        ok, reasons = evaluate_cantonese(text)
        self.assertTrue(ok, reasons)

    def test_trad_mandarin_fails_without_llm(self) -> None:
        text = GOLD_MANDARIN.read_text(encoding="utf-8")
        ok, reasons = evaluate_cantonese(text)
        self.assertFalse(ok)
        self.assertTrue(reasons)


class TestCantoneseHeuristics(unittest.TestCase):
    def test_simplified_denylist_covers_seed_examples(self) -> None:
        for ch in "们这过来对时会么为无发":
            self.assertIn(ch, SIMPLIFIED_ONLY)

    def test_shared_traditional_glyphs_pass_yue_eval(self) -> None:
        for ch in "里后松云准余采":
            self.assertNotIn(ch, SIMPLIFIED_ONLY)
        samples = (
            "公里係長度單位，唔好同市里撈亂。",
            "皇后係古代嘅稱號，唔係簡化字。",
            "松樹係常綠喬木，呢句係書面粵語。",
        )
        for sample in samples:
            ok, reasons = evaluate_cantonese(sample)
            self.assertTrue(ok, (sample, reasons))

    def test_simplified_body_fails(self) -> None:
        ok, reasons = evaluate_cantonese("呢句有们字所以一定唔合格。係嘅唔。")
        self.assertFalse(ok)
        self.assertTrue(any(r.startswith("simplified:") for r in reasons))

    def test_particle_ratio_and_density(self) -> None:
        ok, _ = evaluate_cantonese("眼睛係心靈嘅窗口。唔好交叉手臂。")
        self.assertTrue(ok)
        ok2, reasons = evaluate_cantonese("眼睛是心靈的窗口。不要交叉手臂。這是的是不是的說明。" * 8)
        self.assertFalse(ok2, reasons)
        self.assertTrue(any("particle" in r for r in reasons))

    def test_eval_strips_yaml_and_source_tags(self) -> None:
        text = (
            "---\nlanguage: yue-Hant-HK\n---\n\n"
            "眼睛係心靈嘅窗口。[book-a#c-s-eyes-1] 唔好交叉手臂。\n"
        )
        ok, reasons = evaluate_cantonese(text)
        self.assertTrue(ok, reasons)


class TestTraditionalize(unittest.TestCase):
    def test_technical_terms(self) -> None:
        self.assertEqual(traditionalize("微反应"), "微反應")
        self.assertEqual(traditionalize("额肌"), "額肌")
        for src, dst in TECH_TERM_S2T:
            self.assertIn(dst, traditionalize(src))


class TestEnHanDensity(unittest.TestCase):
    def test_parentheticals_and_source_tags_stripped(self) -> None:
        text = (
            "The eyes are a window (心灵的窗口). "
            "See the tag [9787538583373#c-s-micro-1] for the dispute.\n"
        )
        self.assertLessEqual(en_han_density(text), HAN_DENSITY_CAP)
        leaked = "The 眼睛 are 心灵的窗口 and more 汉字 " * 10
        self.assertGreater(en_han_density(leaked), HAN_DENSITY_CAP)


class TestTranslateYuePrompt(unittest.TestCase):
    def test_eight_fewshots_present(self) -> None:
        blob = TRANSLATE_YUE_PROMPT_PATH.read_text(encoding="utf-8")
        for zh, yue in FEWSHOTS:
            self.assertIn(zh, blob)
            self.assertIn(yue, blob)
        self.assertIn("香港書面粵語", blob)
        self.assertIn("微表情是人們日益關注的一個新詞", blob)


class LengthTranslateLLM(FakeLLMClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._calls = 0

    def complete(self, **kwargs) -> CacheRecord:
        if kwargs.get("stage") in ("translate-en", "translate-yue"):
            self._calls += 1
            if self._calls == 1:
                self._finish_queue.append("length")
                self._queue.append("TRUNCATED")
        return super().complete(**kwargs)


class TestTranslateSplit(unittest.TestCase):
    def test_split_on_h4_then_paragraphs(self) -> None:
        text = "#### A\n\n" + ("甲" * 40) + "\n\n#### B\n\n" + ("乙" * 40)
        chunks = split_translate_chunks(text, max_chars=50)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(c) <= 50 for c in chunks), [len(c) for c in chunks])
        self.assertEqual("".join(c.replace("\n", "") for c in chunks).count("甲"), 40)

    def test_hard_splits_oversize_paragraph_after_h4(self) -> None:
        text = "#### A\n\n" + ("甲" * 80) + "\n\n#### B\n\nshort"
        chunks = split_translate_chunks(text, max_chars=50)
        self.assertTrue(all(len(c) <= 50 for c in chunks), [len(c) for c in chunks])
        self.assertGreater(len(chunks), 2)
        self.assertEqual("".join(chunks).count("甲"), 80)

    def test_length_finish_splits_and_does_not_keep_truncated(self) -> None:
        zh = "#### 瞳孔\n\n瞳孔放大表示兴趣。\n\n#### 眨眼\n\n频繁眨眼表示紧张。"
        fake = LengthTranslateLLM()
        templates = {
            "system": "t",
            "user": "Translate\n\n[[SOURCE]]\n{markdown}\n[[END SOURCE]]",
        }
        out = translate_markdown(
            zh,
            stage=STAGE_EN,
            client=fake,
            templates=templates,
            prompt_hash="p",
            source_hash="s",
            max_input_chars=80,
            chars_per_token=1.0,
            n_samples=5,
            model="fake-model",
            force=True,
        )
        self.assertNotIn("TRUNCATED", out)
        self.assertTrue(out.strip())


class TestTranslateCacheKey(unittest.TestCase):
    def test_input_hash_is_source_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            fake = FakeLLMClient(cache_dir=cache)
            templates = {
                "system": "t",
                "user": "[[SOURCE]]\n{markdown}\n[[END SOURCE]]",
            }
            translate_markdown(
                "眼睛是心灵的窗口。",
                stage=STAGE_YUE,
                client=fake,
                templates=templates,
                prompt_hash="prompt",
                source_hash="abc123",
                max_input_chars=6000,
                chars_per_token=1.0,
                n_samples=0,
                model="fake-model",
                force=False,
            )
            self.assertEqual(len(fake.calls), 1)
            self.assertIn("abc123", fake.calls[0]["input_hashes"])
            self.assertEqual(fake.calls[0]["stage"], STAGE_YUE)
            self.assertEqual(fake.calls[0]["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
