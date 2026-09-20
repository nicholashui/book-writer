"""Outline stage: bucket-split, Pydantic Outline, keyword-bucket fallback, CLI stop."""

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

from book_combiner.cli import main  # noqa: E402
from book_combiner.llm.cache import DiskCache  # noqa: E402
from book_combiner.models import (  # noqa: E402
    CacheKey,
    CacheRecord,
    Outline,
    OutlineSection,
    OutlineTitle,
)
from book_combiner.outline import (  # noqa: E402
    HEADING_TOKEN_CAP,
    OUTLINE_MAX_TOKENS,
    collect_outline_ids,
    estimate_tokens,
    fallback_outline_for_bucket,
    format_heading_line,
    load_fewshot,
    pack_heading_rows,
    parse_outline,
    partition_by_bucket,
    primary_bucket,
    run_outline,
    sha256_file,
    stitch_outlines,
)

FEWSHOT_PART_TITLES = (
    "Part I — Foundations",
    "Part II — The face by region",
    "Part III — Basic emotions on the face",
    "Part IV — Smiles",
    "Part V — Voice and speech",
    "Part VI — Body",
    "Part VII — Deception",
    "Part VIII — Applications",
    "Part IX — Working with your own emotions",
)

TESTDATA = ROOT / "testdata"

SIMPLE_OUTLINE = {
    "title": {"en": "Two Book Overlap", "zh": "两书重叠", "yue_hint": "两书重叠"},
    "parts": [
        {
            "id": "p-cues",
            "title_en": "Cues",
            "title_zh": "线索",
            "chapters": [
                {
                    "id": "c-cues",
                    "title_en": "Signals",
                    "title_zh": "信号",
                    "sections": [
                        {"id": "s-eyes", "title_en": "Eyes", "title_zh": "眼睛"},
                        {"id": "s-handshake", "title_en": "Handshake", "title_zh": "握手"},
                    ],
                }
            ],
        }
    ],
}

SIMPLE_OUTLINE_JSON = json.dumps(SIMPLE_OUTLINE, ensure_ascii=False)

_SOURCE_BLOCK_RE = re.compile(
    r"---SOURCE [^\n]+---\n(.*?)\n---END SOURCE---",
    re.S,
)
_UNIQUE_FROM_RE = re.compile(
    r"UNIQUE FROM [^\n]+:\n(.*?)(?=\n(?:UNIQUE FROM |---SOURCE |\Z))",
    re.S,
)
_TRANSLATE_SOURCE_RE = re.compile(
    r"\[\[SOURCE\]\]\n(.*?)\n\[\[END SOURCE\]\]",
    re.S,
)
_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def fake_merge_from_user(user: str) -> str:
    """Union-of-facts stand-in: concatenate source / UNIQUE FROM blocks."""
    bodies = [b.strip() for b in _SOURCE_BLOCK_RE.findall(user) if b.strip()]
    bodies.extend(b.strip() for b in _UNIQUE_FROM_RE.findall(user) if b.strip())
    body = "\n\n".join(bodies) if bodies else user.strip()
    return f"{body}\n\n---CONFLICTS-JSON---\n[]\n"


def fake_en_from_markdown(md: str) -> str:
    """Wire assemble: no CJK in headings; leftover Han lives in parentheticals."""
    lines: list[str] = []
    for line in md.splitlines():
        match = _HEADING_LINE_RE.match(line)
        if match:
            hashes, title = match.group(1), match.group(2)
            cleaned = _CJK_RUN_RE.sub("Section", title).strip() or "Section"
            cleaned = re.sub(r"\s+", " ", cleaned)
            lines.append(f"{hashes} {cleaned}")
        else:
            lines.append(_CJK_RUN_RE.sub(lambda m: f"({m.group(0)})", line))
    return "\n".join(lines) + "\n"


def fake_yue_from_markdown(md: str) -> str:
    """Wire assemble: Traditionalize and swap 的/是/不 for 嘅/係/唔."""
    from book_combiner.translate import traditionalize

    text = traditionalize(md)
    text = text.replace("不是", "唔係").replace("不要", "唔好")
    text = text.replace("的", "嘅").replace("是", "係").replace("不", "唔")
    return text if text.endswith("\n") else text + "\n"


def fake_translate_from_user(user: str, stage: str) -> str:
    match = _TRANSLATE_SOURCE_RE.search(user)
    payload = match.group(1) if match else user
    if stage == "translate-en":
        return fake_en_from_markdown(payload)
    return fake_yue_from_markdown(payload)


class FakeLLMClient:
    """LLMClient stand-in: queued JSON, optional DiskCache, no network."""

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        default: str = SIMPLE_OUTLINE_JSON,
        cache_dir: Path | None = None,
        model: str = "fake-model",
        finish_reasons: list[str] | None = None,
    ) -> None:
        self.model = model
        self.force = False
        self.calls: list[dict] = []
        self._queue = list(responses or [])
        self._finish_queue = list(finish_reasons or [])
        self.default = default
        self.cache = DiskCache(cache_dir) if cache_dir else None

    def complete(self, **kwargs) -> CacheRecord:
        model = kwargs.get("model") or self.model
        params = {
            "json_mode": bool(kwargs.get("json_mode", False)),
            "max_input_chars": int(kwargs["max_input_chars"]),
            "max_tokens": int(kwargs["max_tokens"]),
            "temperature": float(kwargs["temperature"]),
        }
        key = CacheKey(
            stage=kwargs["stage"],
            model=model,
            prompt_hash=kwargs["prompt_hash"],
            params=params,
            input_hashes=list(kwargs["input_hashes"]),
        )
        force = self.force if kwargs.get("force") is None else kwargs["force"]
        if self.cache is not None and not force:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        self.calls.append(kwargs)
        if self._queue:
            text = self._queue.pop(0)
        elif kwargs.get("stage") == "merge":
            text = fake_merge_from_user(str(kwargs.get("user") or ""))
        elif kwargs.get("stage") in ("translate-en", "translate-yue"):
            text = fake_translate_from_user(str(kwargs.get("user") or ""), str(kwargs.get("stage")))
        else:
            text = self.default
        finish_reason = self._finish_queue.pop(0) if self._finish_queue else "stop"
        record = CacheRecord(
            key=key,
            response_text=text,
            finish_reason=finish_reason,
            input_tokens=1,
            output_tokens=1,
            created_at="2026-09-18T00:00:00Z",
        )
        if self.cache is not None:
            self.cache.put(key, record)
        return record


def cleanup_compiled(book_root: Path, topic: str = "two-book-overlap") -> None:
    for name in (f"{topic}.md", f"{topic}.hk.md"):
        path = Path(book_root) / name
        if path.is_file():
            path.unlink()


def _run_main(argv: list[str], llm_client=None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = main(argv, llm_client=llm_client)
        except SystemExit as exc:
            if isinstance(exc.code, int):
                code = exc.code
            elif exc.code is None:
                code = 0
            else:
                stderr.write(str(exc.code) + "\n")
                code = 1
    return code, stdout.getvalue(), stderr.getvalue()


def _inventory_two_book(artifacts: Path) -> None:
    fake = FakeLLMClient()
    code, _out, err = _run_main(
        [
            "two-book-overlap",
            "--book-root",
            str(TESTDATA),
            "--artifacts-root",
            str(artifacts),
            "--stage",
            "inventory",
        ],
        llm_client=fake,
    )
    if code != 0:
        raise AssertionError(err or "inventory failed")


class TestOutlineModels(unittest.TestCase):
    def test_section_has_no_title_yue(self) -> None:
        self.assertNotIn("title_yue", OutlineSection.model_fields)
        self.assertIn("yue_hint", OutlineTitle.model_fields)
        self.assertIn("en", OutlineTitle.model_fields)
        self.assertIn("zh", OutlineTitle.model_fields)
        parsed = parse_outline(
            json.dumps(
                {
                    **SIMPLE_OUTLINE,
                    "parts": [
                        {
                            **SIMPLE_OUTLINE["parts"][0],
                            "chapters": [
                                {
                                    **SIMPLE_OUTLINE["parts"][0]["chapters"][0],
                                    "sections": [
                                        {
                                            "id": "s-eyes",
                                            "title_en": "Eyes",
                                            "title_zh": "眼睛",
                                            "title_yue": "眼睛",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )
        self.assertIsNotNone(parsed)
        section = parsed.parts[0].chapters[0].sections[0]
        self.assertFalse(hasattr(section, "title_yue") and "title_yue" in section.model_fields)
        dumped = section.model_dump()
        self.assertNotIn("title_yue", dumped)

    def test_fewshot_is_shape_example_not_on_disk_fallback(self) -> None:
        fewshot = load_fewshot("facial-expression")
        self.assertIsNotNone(fewshot)
        Outline.model_validate(fewshot)
        titles = [part["title_en"] for part in fewshot["parts"]]
        for expected in FEWSHOT_PART_TITLES:
            self.assertIn(expected, titles)
        self.assertIsNone(load_fewshot("two-book-overlap"))
        self.assertIsNone(load_fewshot("no-such-topic-xyz"))


class TestBucketSplitAndFallback(unittest.TestCase):
    def test_primary_bucket_first_lexicon_key(self) -> None:
        from book_combiner.lexicon import LEXICON

        self.assertEqual(primary_bucket(["泄密的眼睛"], LEXICON), "eyes")
        self.assertEqual(primary_bucket(["握手"], LEXICON), "hands")
        self.assertEqual(primary_bucket(["量子纠缠"], LEXICON), "other")

    def test_pack_heading_rows_respects_token_cap(self) -> None:
        rows = [
            {
                "section_id": f"stem:{i}:{i}",
                "heading_path": [f"标题{i}"],
                "excerpt": "摘" * 20,
                "char_count": 80,
            }
            for i in range(12)
        ]
        line_tokens = estimate_tokens(format_heading_line(rows[0]), 1.5)
        packs = pack_heading_rows(rows, chars_per_token=1.5, token_cap=max(1, line_tokens * 3))
        self.assertGreater(len(packs), 1)
        self.assertEqual(sum(len(p) for p in packs), 12)
        self.assertEqual(HEADING_TOKEN_CAP, 24_000)

    def test_keyword_bucket_fallback_shape(self) -> None:
        headings = [
            {"heading_path": ["泄密的眼睛"], "section_id": "a:1:2"},
            {"heading_path": ["眼睛会说话", "瞳孔"], "section_id": "b:1:2"},
            {"heading_path": ["泄密的眼睛"], "section_id": "c:1:2"},
        ]
        outline = fallback_outline_for_bucket("eyes", headings, "two-book-overlap")
        self.assertEqual(len(outline.parts), 1)
        part = outline.parts[0]
        self.assertEqual(part.id, "p-eyes")
        self.assertEqual(part.title_en, "Eyes")
        self.assertEqual(len(part.chapters), 2)
        titles = {ch.title_zh for ch in part.chapters}
        self.assertEqual(titles, {"泄密的眼睛", "眼睛会说话"})
        eyes_ch = next(ch for ch in part.chapters if ch.title_zh == "泄密的眼睛")
        self.assertEqual([s.title_zh for s in eyes_ch.sections], ["body"])
        talk_ch = next(ch for ch in part.chapters if ch.title_zh == "眼睛会说话")
        self.assertEqual([s.title_zh for s in talk_ch.sections], ["瞳孔"])
        for title in FEWSHOT_PART_TITLES:
            self.assertNotEqual(part.title_en, title)

    def test_parse_outline_rejects_duplicate_ids(self) -> None:
        dup = json.loads(SIMPLE_OUTLINE_JSON)
        dup["parts"][0]["chapters"][0]["sections"].append(
            {"id": "s-eyes", "title_en": "Eyes again", "title_zh": "又是眼睛"}
        )
        self.assertIsNone(parse_outline(json.dumps(dup, ensure_ascii=False)))
        self.assertIsNotNone(parse_outline(SIMPLE_OUTLINE_JSON))

    def test_stitch_uniquifies_part_chapter_section_ids(self) -> None:
        packs = [
            fallback_outline_for_bucket(
                "eyes",
                [{"heading_path": [f"章{i}"]}],
                "two-book-overlap",
                pack_index=0,
            )
            for i in range(3)
        ]
        # Same pack_index on purpose so ids collide before stitch.
        raw_ids = [id_ for ol in packs for id_ in collect_outline_ids(ol)]
        self.assertNotEqual(len(raw_ids), len(set(raw_ids)))
        stitched = stitch_outlines("two-book-overlap", [("eyes", ol) for ol in packs])
        ids = collect_outline_ids(stitched)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(stitched.parts), 3)
        for title in FEWSHOT_PART_TITLES:
            self.assertNotIn(title, [p.title_en for p in stitched.parts])

    def test_invalid_json_falls_back_not_fewshot_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _inventory_two_book(artifacts)
            fake = FakeLLMClient(default="NOT JSON {{{")
            with redirect_stdout(io.StringIO()):
                outline = run_outline(
                    topic="two-book-overlap",
                    artifacts_root=artifacts,
                    client=fake,
                )
            titles = [part.title_en for part in outline.parts]
            for banned in FEWSHOT_PART_TITLES:
                self.assertNotIn(banned, titles)
            ids = {part.id for part in outline.parts}
            self.assertIn("p-eyes", ids)
            self.assertIn("p-hands", ids)
            md = (artifacts / "two-book-overlap" / "outline" / "outline.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("p-eyes", md)
            self.assertNotIn("Part I — Foundations", md)

    def test_finish_reason_length_uses_keyword_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _inventory_two_book(artifacts)
            fake = FakeLLMClient(
                default=SIMPLE_OUTLINE_JSON,
                finish_reasons=["length", "length", "length"],
            )
            with redirect_stdout(io.StringIO()):
                outline = run_outline(
                    topic="two-book-overlap",
                    artifacts_root=artifacts,
                    client=fake,
                )
            ids = {part.id for part in outline.parts}
            self.assertIn("p-eyes", ids)
            self.assertIn("p-hands", ids)
            for title in FEWSHOT_PART_TITLES:
                self.assertNotIn(title, [p.title_en for p in outline.parts])
            self.assertEqual(len(collect_outline_ids(outline)), len(set(collect_outline_ids(outline))))

    def test_bucket_split_one_call_per_bucket_plus_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _inventory_two_book(artifacts)
            headings = json.loads(
                (artifacts / "two-book-overlap" / "inventory" / "headings.json").read_text(
                    encoding="utf-8"
                )
            )
            from book_combiner.lexicon import LEXICON

            buckets = partition_by_bucket(headings, LEXICON)
            self.assertIn("eyes", buckets)
            self.assertIn("hands", buckets)
            fake = FakeLLMClient(default=SIMPLE_OUTLINE_JSON)
            with redirect_stdout(io.StringIO()):
                run_outline(
                    topic="two-book-overlap",
                    artifacts_root=artifacts,
                    client=fake,
                )
            outline_calls = [c for c in fake.calls if c["stage"] == "outline"]
            self.assertEqual(len(outline_calls), len(buckets) + 1)
            for call in outline_calls:
                self.assertTrue(call["json_mode"])
                self.assertEqual(call["max_tokens"], OUTLINE_MAX_TOKENS)
                self.assertEqual(call["temperature"], 0.3)
            self.assertNotIn("Part I — Foundations", outline_calls[0]["user"])

    def test_facial_expression_fewshot_in_prompt_not_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book_root = Path(tmp) / "book"
            topic_dir = book_root / "facial-expression"
            topic_dir.mkdir(parents=True)
            for src in (TESTDATA / "two-book-overlap").glob("*.md"):
                (topic_dir / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            artifacts = Path(tmp) / "artifacts"
            fake = FakeLLMClient(default="<<<not-json>>>")
            code, out, err = _run_main(
                [
                    "facial-expression",
                    "--book-root",
                    str(book_root),
                    "--artifacts-root",
                    str(artifacts),
                    "--stage",
                    "outline",
                ],
                llm_client=fake,
            )
            self.assertEqual(code, 0, err)
            self.assertTrue(fake.calls)
            self.assertIn("Part I — Foundations", fake.calls[0]["user"])
            outline = json.loads(
                (artifacts / "facial-expression" / "outline" / "outline.json").read_text(
                    encoding="utf-8"
                )
            )
            titles = [p["title_en"] for p in outline["parts"]]
            for banned in FEWSHOT_PART_TITLES:
                self.assertNotIn(banned, titles)
            self.assertTrue(any(p["id"].startswith("p-eyes") or p["id"] == "p-eyes" for p in outline["parts"]))
            self.assertIn("[outline]", out)
            self.assertFalse((artifacts / "facial-expression" / "outline" / "assignment.json").exists())


class TestOutlineCli(unittest.TestCase):
    def test_stage_outline_writes_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            fake = FakeLLMClient(default=SIMPLE_OUTLINE_JSON)
            code, out, err = _run_main(
                [
                    "two-book-overlap",
                    "--book-root",
                    str(TESTDATA),
                    "--artifacts-root",
                    str(artifacts),
                    "--stage",
                    "outline",
                ],
                llm_client=fake,
            )
            self.assertEqual(code, 0, err)
            self.assertIn("[outline]", out)
            self.assertNotIn("[assign]", out)
            self.assertNotIn("not implemented until a later PR", err)
            outline_dir = artifacts / "two-book-overlap" / "outline"
            self.assertTrue((outline_dir / "outline.json").is_file())
            self.assertTrue((outline_dir / "outline.md").is_file())
            self.assertFalse((outline_dir / "assignment.json").exists())
            data = json.loads((outline_dir / "outline.json").read_text(encoding="utf-8"))
            Outline.model_validate(data)
            ids = {
                sec["id"]
                for part in data["parts"]
                for ch in part["chapters"]
                for sec in ch["sections"]
            }
            self.assertIn("s-eyes", ids)
            self.assertIn("s-handshake", ids)

    def test_from_stage_assign_keeps_edited_outline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            fake = FakeLLMClient(default=SIMPLE_OUTLINE_JSON)
            code, _out, err = _run_main(
                [
                    "two-book-overlap",
                    "--book-root",
                    str(TESTDATA),
                    "--artifacts-root",
                    str(artifacts),
                    "--stage",
                    "outline",
                ],
                llm_client=fake,
            )
            self.assertEqual(code, 0, err)
            outline_path = artifacts / "two-book-overlap" / "outline" / "outline.json"
            edited = json.loads(outline_path.read_text(encoding="utf-8"))
            edited["title"]["en"] = "Edited Title"
            outline_path.write_text(
                json.dumps(edited, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            before = sha256_file(outline_path)
            fake2 = FakeLLMClient(default=SIMPLE_OUTLINE_JSON)
            try:
                code2, out2, err2 = _run_main(
                    [
                        "two-book-overlap",
                        "--book-root",
                        str(TESTDATA),
                        "--artifacts-root",
                        str(artifacts),
                        "--from-stage",
                        "assign",
                    ],
                    llm_client=fake2,
                )
            finally:
                cleanup_compiled(TESTDATA)
            self.assertEqual(code2, 0, err2)
            self.assertIn("[assign]", out2)
            self.assertNotIn("not implemented until a later PR", err2)
            self.assertEqual(sha256_file(outline_path), before)
            self.assertEqual(
                json.loads(outline_path.read_text(encoding="utf-8"))["title"]["en"],
                "Edited Title",
            )
            self.assertTrue(
                (artifacts / "two-book-overlap" / "outline" / "assignment.json").is_file()
            )
            outline_calls = [c for c in fake2.calls if c["stage"] == "outline"]
            self.assertEqual(outline_calls, [])


if __name__ == "__main__":
    unittest.main()
