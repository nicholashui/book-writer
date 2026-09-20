"""Same-heading union-of-facts merge, complementary concat, coverage checks."""

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

from book_combiner.coverage import extract_tokens, flatten_tokens, required_tokens  # noqa: E402
from book_combiner.extract import content_hash  # noqa: E402
from book_combiner.merge import (  # noqa: E402
    MODEL_OUTPUT_CAP,
    MergeError,
    expected_out_chars,
    max_tokens_for,
    merge_input_hashes,
    needed_tokens,
    normalize_heading_path,
    pack_graft_payload,
    partition_by_heading,
    run_merge,
    union_metrics,
    unique_members,
)
from book_combiner.models import CacheRecord, Section  # noqa: E402

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from test_outline import (  # noqa: E402
    SIMPLE_OUTLINE_JSON,
    TESTDATA,
    FakeLLMClient,
    _run_main,
    cleanup_compiled,
    fake_merge_from_user,
)

SWEAT_CATALOGUE = "温热性出汗、精神性出汗、味觉性出汗和运动性出汗"
SWEAT_FOUR = ["温热性出汗", "精神性出汗", "味觉性出汗", "运动性出汗"]
DIGEST = "Hands may sweat or tremble when nervous."


def _unique_run(n: int, offset: int = 0) -> str:
    return "".join(chr(0x4E00 + offset + i) for i in range(n))


def _section(
    sid: str,
    text: str,
    heading: list[str] | None = None,
    stem: str = "a",
    duplicate_of: str | None = None,
    kind: str = "body",
) -> Section:
    return Section(
        id=sid,
        source_stem=stem,
        source_book=stem,
        heading_path=list(heading or ["手汗"]),
        level=2,
        start_line=1,
        end_line=2,
        text=text,
        kind=kind,
        char_count=len(text),
        content_hash=content_hash(text),
        duplicate_of=duplicate_of,
    )


def _hands_outline() -> dict:
    return {
        "title": {"en": "Hands", "zh": "手", "yue_hint": "手"},
        "parts": [
            {
                "id": "p-body",
                "title_en": "Body",
                "title_zh": "身体",
                "chapters": [
                    {
                        "id": "c-hands",
                        "title_en": "Hands",
                        "title_zh": "手",
                        "sections": [
                            {"id": "s-hands", "title_en": "Hand sweat", "title_zh": "手汗"},
                            {"id": "s-eyes", "title_en": "Eyes", "title_zh": "眼睛"},
                        ],
                    }
                ],
            }
        ],
    }


def _write_merge_artifacts(
    artifacts: Path,
    topic: str,
    sections: list[Section],
    *,
    outline: dict | None = None,
    node_id: str = "s-hands",
) -> Path:
    topic_dir = artifacts / topic
    extract_dir = topic_dir / "extract"
    outline_dir = topic_dir / "outline"
    extract_dir.mkdir(parents=True, exist_ok=True)
    outline_dir.mkdir(parents=True, exist_ok=True)
    by_stem: dict[str, list[dict]] = {}
    for section in sections:
        by_stem.setdefault(section.source_stem, []).append(
            {
                "id": section.id,
                "source_stem": section.source_stem,
                "source_book": section.source_book,
                "heading_path": section.heading_path,
                "level": section.level,
                "start_line": section.start_line,
                "end_line": section.end_line,
                "text": section.text,
                "kind": section.kind,
                "char_count": section.char_count,
                "content_hash": section.content_hash,
                "duplicate_of": section.duplicate_of,
            }
        )
    for stem, rows in by_stem.items():
        (extract_dir / f"{stem}.sections.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    payload = outline if outline is not None else _hands_outline()
    (outline_dir / "outline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    assignments = [
        {
            "section_id": section.id,
            "primary_node_id": node_id,
            "secondary_node_id": None,
        }
        for section in sections
    ]
    (outline_dir / "assignment.json").write_text(
        json.dumps({"assignments": assignments}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return topic_dir


class DigestLLM(FakeLLMClient):
    def complete(self, **kwargs) -> CacheRecord:
        kwargs = dict(kwargs)
        self.model = kwargs.get("model") or self.model
        if kwargs.get("stage") == "merge":
            self._queue.append(
                f"{DIGEST}\n\n---CONFLICTS-JSON---\n[]\n"
            )
        return super().complete(**kwargs)


class TerminatorlessLLM(FakeLLMClient):
    def complete(self, **kwargs) -> CacheRecord:
        self._queue.append("merged body with no terminator at all")
        return super().complete(**kwargs)


class LengthLLM(FakeLLMClient):
    def complete(self, **kwargs) -> CacheRecord:
        if kwargs.get("stage") == "merge":
            self._queue.append("TRUNCATED ONLY")
            self._finish_queue.append("length")
        return super().complete(**kwargs)


class TestUnionMetrics(unittest.TestCase):
    def test_expected_out_chars_is_union_not_longest_times_1_3(self) -> None:
        members = [
            _section("a:1:2", _unique_run(1500, 0), stem="a"),
            _section("b:1:2", _unique_run(1500, 2000), stem="b"),
            _section("c:1:2", _unique_run(1500, 4000), stem="c"),
        ]
        metrics = union_metrics(members)
        self.assertEqual(metrics.unique_leaf_chars, 4500)
        self.assertGreater(metrics.union_chars, 1.3 * 1500)
        expected = expected_out_chars(
            union_chars=metrics.union_chars,
            graft_payload_chars=0,
            graft_applied=False,
        )
        self.assertEqual(expected, int(metrics.union_chars))
        self.assertNotEqual(expected, int(1.3 * 1500))
        needed = needed_tokens(expected, 1.0)
        self.assertGreater(needed, int(1.3 * 1500))
        self.assertEqual(max_tokens_for(expected, 1.0, 0), MODEL_OUTPUT_CAP)
        self.assertEqual(max_tokens_for(expected, 1.0, 5), min(MODEL_OUTPUT_CAP, needed))
        self.assertAlmostEqual(metrics.size_floor, 0.7 * metrics.union_chars)

    def test_graft_expected_out_chars_uses_payload(self) -> None:
        expected = expected_out_chars(
            union_chars=9000,
            graft_payload_chars=1200,
            graft_applied=True,
        )
        self.assertEqual(expected, 1200)

    def test_same_heading_partition_ignores_jaccard(self) -> None:
        members = [
            _section("a:1:2", _unique_run(80, 0), heading=["瞳孔"], stem="a"),
            _section("b:1:2", _unique_run(80, 200), heading=["瞳孔"], stem="b"),
            _section("c:1:2", _unique_run(80, 400), heading=["眨眼"], stem="c"),
        ]
        groups = partition_by_heading(members)
        keys = {k for k, _ in groups}
        self.assertEqual(len(groups), 2)
        self.assertEqual(normalize_heading_path(["瞳孔"]), normalize_heading_path(["瞳孔"]))
        pupil = next(g for k, g in groups if "瞳孔" in k or normalize_heading_path(["瞳孔"]) == k)
        self.assertEqual(len(pupil), 2)


class TestCoverageMerge(unittest.TestCase):
    def test_digest_is_rejected_and_four_sweat_names_are_required(self) -> None:
        pad = "手心出汗属于精神性反应，紧张时更为明显。" * 40
        text = f"{pad}\n\n{SWEAT_CATALOGUE}。\n\n{pad}"
        tokens = flatten_tokens(extract_tokens(text))
        for name in SWEAT_FOUR:
            self.assertIn(name, tokens)
        self.assertNotIn("和运动性出汗", tokens)
        self.assertTrue(all(name not in DIGEST for name in SWEAT_FOUR))

        section = _section("a:10:20", text, heading=["激动时，手会冒汗或者颤抖吗？"], stem="a")
        other = _section(
            "b:10:20",
            "握手时掌心潮湿也可能是紧张，而不是室温过高。" * 8,
            heading=["激动时，手会冒汗或者颤抖吗？"],
            stem="b",
        )
        required = required_tokens([section, other])
        for name in SWEAT_FOUR:
            self.assertIn(name, required)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _write_merge_artifacts(artifacts, "hands", [section, other])
            fake = DigestLLM()
            results = run_merge(
                topic="hands",
                artifacts_root=artifacts,
                client=fake,
                max_input_chars=6000,
            )
            self.assertFalse(results[0].failed)
            self.assertTrue(results[0].nonconverged)
            path = artifacts / "hands" / "merge" / "nodes" / "s-hands.zh.md"
            body = path.read_text(encoding="utf-8")
            for name in SWEAT_FOUR:
                self.assertIn(name, body)
            self.assertNotIn(DIGEST, body)
            self.assertFalse((artifacts / "hands" / "merge" / "nodes" / "s-hands.FAILED.md").is_file())
            self.assertTrue(any(c["stage"] == "merge" for c in fake.calls))

    def test_concat_unique_sentences_passes_coverage(self) -> None:
        a = _section(
            "a:1:2",
            "瞳孔放大表示兴趣。" + SWEAT_CATALOGUE + "。额肌也会参与。",
            heading=["手汗"],
            stem="a",
        )
        b = _section(
            "b:1:2",
            "握手时用力过猛可能表示支配欲，而不是礼貌客气。FACS 可编码 AU12。",
            heading=["手汗"],
            stem="b",
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _write_merge_artifacts(artifacts, "hands", [a, b])
            fake = FakeLLMClient()
            results = run_merge(
                topic="hands",
                artifacts_root=artifacts,
                client=fake,
                max_input_chars=6000,
            )
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].failed)
            path = artifacts / "hands" / "merge" / "nodes" / "s-hands.zh.md"
            self.assertTrue(path.is_file())
            text = path.read_text(encoding="utf-8")
            for name in SWEAT_FOUR:
                self.assertIn(name, text)
            self.assertIn("支配欲", text)
            self.assertIn("### 手汗", text)
            self.assertFalse((artifacts / "hands" / "merge" / "nodes" / "s-hands.FAILED.md").is_file())
            merge_calls = [c for c in fake.calls if c["stage"] == "merge"]
            self.assertEqual(len(merge_calls), 1)
            user = merge_calls[0]["user"]
            self.assertIn("---SOURCE a---", user)
            self.assertIn("---SOURCE b---", user)
            self.assertGreater(merge_calls[0]["max_tokens"], int(1.3 * max(a.char_count, b.char_count)))
            leftover = list((artifacts / "hands" / "merge" / "nodes").glob("*.partial"))
            self.assertEqual(leftover, [])


class TestHeadingPartition(unittest.TestCase):
    def test_same_heading_low_jaccard_still_llm_merged(self) -> None:
        a = _section("a:1:2", "瞳孔放大通常表示兴趣与唤醒。" * 8, heading=["瞳孔"], stem="a")
        b = _section("b:1:2", "瞳孔在厌恶或剧痛时可能缩小。" * 8, heading=["瞳孔"], stem="b")
        metrics = union_metrics([a, b])
        self.assertLess(metrics.remaining_overlap, 0.55)
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _write_merge_artifacts(artifacts, "eyes", [a, b], node_id="s-eyes")
            fake = FakeLLMClient()
            run_merge(topic="eyes", artifacts_root=artifacts, client=fake)
            calls = [c for c in fake.calls if c["stage"] == "merge"]
            self.assertEqual(len(calls), 1)
            user = calls[0]["user"]
            self.assertIn("兴趣与唤醒", user)
            self.assertIn("厌恶或剧痛", user)
            out = (artifacts / "eyes" / "merge" / "nodes" / "s-eyes.zh.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("兴趣与唤醒", out)
            self.assertIn("厌恶或剧痛", out)
            self.assertNotIn("#### 瞳孔", out)

    def test_different_heading_concat_not_llm_stitched(self) -> None:
        pupil = _section(
            "a:1:2",
            "瞳孔放大表示兴趣，这是独特的瞳孔事实。",
            heading=["眼睛", "瞳孔"],
            stem="a",
        )
        pupil_b = _section(
            "a2:1:2",
            "黑暗中瞳孔也会放大，这是另一条瞳孔事实。",
            heading=["眼睛", "瞳孔"],
            stem="c",
        )
        blink = _section(
            "b:1:2",
            "眨眼频率升高可能表示紧张，这是独特的眨眼事实。",
            heading=["眼睛", "眨眼"],
            stem="b",
        )
        blink_b = _section(
            "b2:1:2",
            "眨眼过少可能表示专注，这是另一条眨眼事实。",
            heading=["眼睛", "眨眼"],
            stem="d",
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _write_merge_artifacts(artifacts, "eyes", [pupil, pupil_b, blink, blink_b], node_id="s-eyes")
            fake = FakeLLMClient()
            run_merge(topic="eyes", artifacts_root=artifacts, client=fake)
            calls = [c for c in fake.calls if c["stage"] == "merge"]
            self.assertEqual(len(calls), 2)
            users = [c["user"] for c in calls]
            self.assertTrue(any("独特的瞳孔事实" in u for u in users))
            self.assertTrue(any("独特的眨眼事实" in u for u in users))
            self.assertFalse(
                any("独特的瞳孔事实" in u and "独特的眨眼事实" in u for u in users)
            )
            out = (artifacts / "eyes" / "merge" / "nodes" / "s-eyes.zh.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("#### 瞳孔", out)
            self.assertIn("#### 眨眼", out)
            self.assertIn("独特的瞳孔事实", out)
            self.assertIn("独特的眨眼事实", out)

    def test_complementary_groups_disk_cache_keeps_distinct_completions(self) -> None:
        pupil = _section(
            "a:1:2",
            "瞳孔放大表示兴趣，这是独特的瞳孔事实。",
            heading=["眼睛", "瞳孔"],
            stem="a",
        )
        pupil_b = _section(
            "a2:1:2",
            "黑暗中瞳孔也会放大，这是另一条瞳孔事实。",
            heading=["眼睛", "瞳孔"],
            stem="c",
        )
        blink = _section(
            "b:1:2",
            "眨眼频率升高可能表示紧张，这是独特的眨眼事实。",
            heading=["眼睛", "眨眼"],
            stem="b",
        )
        blink_b = _section(
            "b2:1:2",
            "眨眼过少可能表示专注，这是另一条眨眼事实。",
            heading=["眼睛", "眨眼"],
            stem="d",
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _write_merge_artifacts(
                artifacts, "eyes", [pupil, pupil_b, blink, blink_b], node_id="s-eyes"
            )
            fake = FakeLLMClient(cache_dir=artifacts / "eyes" / "cache")
            run_merge(topic="eyes", artifacts_root=artifacts, client=fake)
            calls = [c for c in fake.calls if c["stage"] == "merge"]
            self.assertEqual(len(calls), 2)
            hash_sets = [tuple(c["input_hashes"]) for c in calls]
            self.assertEqual(len(set(hash_sets)), 2)
            users = [c["user"] for c in calls]
            self.assertTrue(any("独特的瞳孔事实" in u for u in users))
            self.assertTrue(any("独特的眨眼事实" in u for u in users))
            self.assertFalse(
                any("独特的瞳孔事实" in u and "独特的眨眼事实" in u for u in users)
            )
            out = (artifacts / "eyes" / "merge" / "nodes" / "s-eyes.zh.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("独特的瞳孔事实", out)
            self.assertIn("独特的眨眼事实", out)
            n_calls = len(fake.calls)
            run_merge(topic="eyes", artifacts_root=artifacts, client=fake)
            self.assertEqual(len(fake.calls), n_calls)

    def test_duplicate_of_short_circuit_canonical_only(self) -> None:
        canon = _section("a:1:2", "完整的手汗说明包含精神性出汗。" * 6, stem="a")
        dup = _section(
            "b:1:2",
            "较短的重复手汗说明。" * 3,
            stem="b",
            duplicate_of=canon.id,
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _write_merge_artifacts(artifacts, "hands", [canon, dup])
            fake = FakeLLMClient()
            run_merge(topic="hands", artifacts_root=artifacts, client=fake)
            calls = [c for c in fake.calls if c["stage"] == "merge"]
            self.assertEqual(len(calls), 0)
            out = (artifacts / "hands" / "merge" / "nodes" / "s-hands.zh.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("完整的手汗说明", out)
            self.assertNotIn("较短的重复手汗说明", out)

    def test_single_unique_leaf_skips_llm(self) -> None:
        section = _section("a:1:2", "颏肌收缩会抬高下巴。" * 8, heading=["颏肌"], stem="a")
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _write_merge_artifacts(artifacts, "hands", [section], node_id="s-hands")
            fake = DigestLLM()
            results = run_merge(topic="hands", artifacts_root=artifacts, client=fake)
            self.assertFalse(results[0].failed)
            self.assertEqual([c for c in fake.calls if c["stage"] == "merge"], [])
            out = (artifacts / "hands" / "merge" / "nodes" / "s-hands.zh.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("颏肌收缩会抬高下巴", out)
            self.assertFalse((artifacts / "hands" / "merge" / "nodes" / "s-hands.FAILED.md").is_file())

    def test_figure_placeholder_does_not_fail_coverage(self) -> None:
        text = (
            "眉毛上扬表示惊讶。" * 20
            + "\n\n*[Figure omitted from source conversion: 图1.1　紧张的女商人]*\n\n"
            + "FACS 可编码 AU1。"
        )
        a = _section("a:1:2", text, heading=["眉毛"], stem="a")
        b = _section("b:1:2", "皱眉肌收缩表示困惑或专注思考。" * 12, heading=["眉毛"], stem="b")
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _write_merge_artifacts(artifacts, "hands", [a, b], node_id="s-hands")
            fake = FakeLLMClient()
            results = run_merge(topic="hands", artifacts_root=artifacts, client=fake)
            self.assertFalse(results[0].failed)
            self.assertFalse((artifacts / "hands" / "merge" / "nodes" / "s-hands.FAILED.md").is_file())


class TestMergeFailureModes(unittest.TestCase):
    def test_missing_terminator_retries_then_failed_md(self) -> None:
        a = _section("a:1:2", "瞳孔放大表示兴趣。" * 10, heading=["瞳孔"], stem="a")
        b = _section("b:1:2", "频繁眨眼表示紧张不安。" * 10, heading=["瞳孔"], stem="b")
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _write_merge_artifacts(artifacts, "eyes", [a, b], node_id="s-eyes")
            fake = TerminatorlessLLM()
            results = run_merge(topic="eyes", artifacts_root=artifacts, client=fake)
            self.assertFalse(results[0].failed)
            self.assertTrue(results[0].nonconverged)
            failed = artifacts / "eyes" / "merge" / "nodes" / "s-eyes.FAILED.md"
            self.assertFalse(failed.is_file())
            out = (artifacts / "eyes" / "merge" / "nodes" / "s-eyes.zh.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("瞳孔放大表示兴趣", out)
            self.assertIn("频繁眨眼表示紧张不安", out)
            merge_calls = [c for c in fake.calls if c["stage"] == "merge"]
            self.assertEqual(len(merge_calls), 2)
            self.assertIn("---CONFLICTS-JSON---", merge_calls[1]["user"])

    def test_finish_reason_length_never_accepts_truncated(self) -> None:
        a = _section("a:1:2", "瞳孔放大表示兴趣。" * 10, heading=["瞳孔"], stem="a")
        b = _section("b:1:2", "频繁眨眼表示紧张不安。" * 10, heading=["瞳孔"], stem="b")
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _write_merge_artifacts(artifacts, "eyes", [a, b], node_id="s-eyes")
            fake = LengthLLM()
            run_merge(topic="eyes", artifacts_root=artifacts, client=fake)
            out = (artifacts / "eyes" / "merge" / "nodes" / "s-eyes.zh.md").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("TRUNCATED ONLY", out)
            self.assertIn("瞳孔放大表示兴趣", out)
            self.assertIn("频繁眨眼表示紧张不安", out)
            meta = json.loads(
                (artifacts / "eyes" / "merge" / "nodes" / "s-eyes.meta.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(meta["merge_nonconverged"])

    def test_over_budget_three_sources_concat_nonconverged(self) -> None:
        members = [
            _section("a:1:2", _unique_run(200, 0), heading=["手汗"], stem="a"),
            _section("b:1:2", _unique_run(200, 400), heading=["手汗"], stem="b"),
            _section("c:1:2", _unique_run(200, 800), heading=["手汗"], stem="c"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _write_merge_artifacts(artifacts, "hands", members)
            fake = FakeLLMClient()
            run_merge(
                topic="hands",
                artifacts_root=artifacts,
                client=fake,
                max_input_chars=1,
            )
            meta = json.loads(
                (artifacts / "hands" / "merge" / "nodes" / "s-hands.meta.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(meta["merge_nonconverged"])
            out = (artifacts / "hands" / "merge" / "nodes" / "s-hands.zh.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("####", out)
            self.assertIn(members[0].text[:20], out)

    def test_graft_packing_uses_unique_from_blocks(self) -> None:
        shared = "共同的手汗导言段落，说明紧张时手心可能出汗。" * 4
        extra = "独特的味觉性出汗例子：吃辣椒后面部出汗。"
        a = _section("a:1:2", shared + "温热性出汗也会出现。" + shared, stem="a")
        b = _section("b:1:2", shared + "\n\n" + extra, stem="b")
        raw, grafted_flag, _ = pack_graft_payload(
            unique_members([a, b]),
            max_input_chars=len(shared),
        )
        self.assertTrue(grafted_flag)
        self.assertIn("UNIQUE FROM b:", raw)
        self.assertIn("味觉性出汗", raw)


class TestMergeCli(unittest.TestCase):
    def test_stage_merge_requires_assignment(self) -> None:
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
            code2, _out2, err2 = _run_main(
                [
                    "two-book-overlap",
                    "--book-root",
                    str(TESTDATA),
                    "--artifacts-root",
                    str(artifacts),
                    "--stage",
                    "merge",
                ],
                llm_client=fake,
            )
            self.assertEqual(code2, 2)
            self.assertIn("assignment.json not found", err2)

    def test_stage_merge_writes_nodes_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            fake = FakeLLMClient(default=SIMPLE_OUTLINE_JSON)
            try:
                code, out, err = _run_main(
                    [
                        "two-book-overlap",
                        "--book-root",
                        str(TESTDATA),
                        "--artifacts-root",
                        str(artifacts),
                        "--from-stage",
                        "outline",
                    ],
                    llm_client=fake,
                )
            finally:
                cleanup_compiled(TESTDATA)
            self.assertEqual(code, 0, err)
            self.assertIn("[merge]", out)
            nodes = artifacts / "two-book-overlap" / "merge" / "nodes"
            eyes = nodes / "s-eyes.zh.md"
            hands = nodes / "s-handshake.zh.md"
            self.assertTrue(eyes.is_file())
            self.assertTrue(hands.is_file())
            eyes_text = eyes.read_text(encoding="utf-8")
            self.assertIn("####", eyes_text)
            self.assertIn("瞳孔", eyes_text)
            self.assertIn("眨眼", eyes_text or (nodes / "s-eyes.zh.md").read_text(encoding="utf-8"))
            self.assertFalse(list(nodes.glob("*.partial")))
            self.assertNotIn("not implemented until a later PR", err)
            self.assertIn("[assemble]", out)

    def test_merge_cache_hashes_include_node_title(self) -> None:
        a = _section("a:1:2", "瞳孔放大表示兴趣。", heading=["瞳孔"])
        hashes = merge_input_hashes([a], "s-eyes", "Eyes", "眼睛")
        self.assertEqual(hashes[0], a.content_hash)
        self.assertIn("s-eyes", hashes)
        pupil = merge_input_hashes(
            [a],
            "s-eyes",
            "Eyes",
            "眼睛",
            payload="pupil-payload",
            heading_key="眼睛>瞳孔",
            round_id="depth:0",
        )
        blink = merge_input_hashes(
            [a],
            "s-eyes",
            "Eyes",
            "眼睛",
            payload="blink-payload",
            heading_key="眼睛>眨眼",
            round_id="depth:0",
        )
        self.assertNotEqual(pupil, blink)

    def test_conflicts_jsonl_replaced_across_runs(self) -> None:
        a = _section("a:1:2", "瞳孔放大表示兴趣。" * 8, heading=["瞳孔"], stem="a")
        b = _section("b:1:2", "频繁眨眼表示紧张不安。" * 8, heading=["瞳孔"], stem="b")
        conflict = {
            "id": "c-s-eyes-1",
            "node_id": "s-eyes",
            "claim_a": {"source": "a", "text": "兴趣"},
            "claim_b": {"source": "b", "text": "紧张"},
            "equivalent_hint": False,
            "inline_tag": "[a#c-s-eyes-1]",
        }

        class _ConflictFake(FakeLLMClient):
            def complete(self, **kwargs):
                if kwargs.get("stage") == "merge":
                    body = fake_merge_from_user(str(kwargs.get("user") or ""))
                    md = body.split("---CONFLICTS-JSON---", 1)[0].strip()
                    text = (
                        md
                        + " [a#c-s-eyes-1]\n\n---CONFLICTS-JSON---\n"
                        + json.dumps([conflict], ensure_ascii=False)
                        + "\n"
                    )
                    self._queue.append(text)
                return super().complete(**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _write_merge_artifacts(artifacts, "eyes", [a, b], node_id="s-eyes")
            path = artifacts / "eyes" / "merge" / "conflicts.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"id": "stale"}\n{"id": "stale-2"}\n', encoding="utf-8")
            fake = _ConflictFake()
            run_merge(topic="eyes", artifacts_root=artifacts, client=fake)
            first = path.read_text(encoding="utf-8")
            self.assertNotIn("stale", first)
            first_rows = [ln for ln in first.splitlines() if ln.strip()]
            self.assertEqual(len(first_rows), 1)
            self.assertIn("c-s-eyes-1", first_rows[0])
            run_merge(topic="eyes", artifacts_root=artifacts, client=fake)
            second = path.read_text(encoding="utf-8")
            self.assertNotIn("stale", second)
            second_rows = [ln for ln in second.splitlines() if ln.strip()]
            self.assertEqual(len(second_rows), 1)
            self.assertIn("c-s-eyes-1", second_rows[0])


if __name__ == "__main__":
    unittest.main()
