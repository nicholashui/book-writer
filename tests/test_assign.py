"""Assign stage: pre-seed, batches, cache hashes follow outline.json edits."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from book_combiner.assign import (  # noqa: E402
    ASSIGN_BATCH_SIZE,
    assign_batch_limit,
    assign_stage_input_hashes,
    first_section_id,
    flatten_outline_nodes,
    preseed_assignment,
    run_assign,
)
from book_combiner.lexicon import LEXICON, load_lexicon  # noqa: E402
from book_combiner.llm.cache import cache_key_hash  # noqa: E402
from book_combiner.models import AssignmentItem, CacheKey, Outline  # noqa: E402
from book_combiner.outline import sha256_file  # noqa: E402

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from test_outline import (  # noqa: E402
    SIMPLE_OUTLINE,
    SIMPLE_OUTLINE_JSON,
    TESTDATA,
    FakeLLMClient,
    _run_main,
    cleanup_compiled,
)


class TestAssignSchema(unittest.TestCase):
    def test_assignment_has_no_title_yue(self) -> None:
        self.assertNotIn("title_yue", AssignmentItem.model_fields)
        self.assertTrue(all("yue" not in name for name in AssignmentItem.model_fields))
        self.assertLessEqual(assign_batch_limit(), ASSIGN_BATCH_SIZE)
        self.assertLessEqual(assign_batch_limit(), 40)

    def test_preseed_unique_bucket_match(self) -> None:
        outline = Outline.model_validate(SIMPLE_OUTLINE)
        nodes = flatten_outline_nodes(outline)
        eyes = preseed_assignment(
            {"section_id": "a:1:2", "heading_path": ["泄密的眼睛"]},
            nodes,
            LEXICON,
        )
        hands = preseed_assignment(
            {"section_id": "a:3:4", "heading_path": ["握手"]},
            nodes,
            LEXICON,
        )
        none = preseed_assignment(
            {"section_id": "a:5:6", "heading_path": ["量子纠缠"]},
            nodes,
            LEXICON,
        )
        self.assertIsNotNone(eyes)
        self.assertEqual(eyes.primary_node_id, "s-eyes")
        self.assertIsNotNone(hands)
        self.assertEqual(hands.primary_node_id, "s-handshake")
        self.assertIsNone(none)


class TestAssignTwoBook(unittest.TestCase):
    def test_two_book_overlap_eyes_and_handshake_nodes(self) -> None:
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
            self.assertIn("[outline]", out)
            self.assertIn("[assign]", out)
            assignment = json.loads(
                (artifacts / "two-book-overlap" / "outline" / "assignment.json").read_text(
                    encoding="utf-8"
                )
            )
            items = assignment["assignments"]
            self.assertTrue(items)
            headings = json.loads(
                (artifacts / "two-book-overlap" / "inventory" / "headings.json").read_text(
                    encoding="utf-8"
                )
            )
            by_id = {row["section_id"]: row for row in headings}
            eyes_ids = [
                item["primary_node_id"]
                for item in items
                if any("眼" in p for p in by_id[item["section_id"]]["heading_path"])
            ]
            hand_ids = [
                item["primary_node_id"]
                for item in items
                if any("握手" in p for p in by_id[item["section_id"]]["heading_path"])
            ]
            self.assertTrue(eyes_ids)
            self.assertTrue(hand_ids)
            self.assertEqual(set(eyes_ids), {"s-eyes"})
            self.assertEqual(set(hand_ids), {"s-handshake"})
            blob = json.dumps(assignment)
            self.assertNotIn("title_yue", blob)
            assign_calls = [c for c in fake.calls if c["stage"] == "assign"]
            self.assertEqual(assign_calls, [])

    def test_unknown_topic_with_no_hits_sends_every_row_to_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book_root = Path(tmp) / "book"
            topic_dir = book_root / "quantum-notes"
            topic_dir.mkdir(parents=True)
            (topic_dir / "a.md").write_text(
                "# 量子纠缠\n\n"
                "量子纠缠是物理学中两个粒子无论相距多远都会保持关联的现象，测量其中一个立即确定另一个。\n",
                encoding="utf-8",
            )
            (topic_dir / "b.md").write_text(
                "# 波函数坍缩\n\n"
                "波函数坍缩描述观测如何把叠加态变成确定结果，这与表情或肢体语言完全无关。\n",
                encoding="utf-8",
            )
            artifacts = Path(tmp) / "artifacts"
            outline_json = json.dumps(
                {
                    "title": {"en": "Quantum Notes", "zh": "量子笔记", "yue_hint": "量子笔记"},
                    "parts": [
                        {
                            "id": "p-other",
                            "title_en": "Other",
                            "title_zh": "其他",
                            "chapters": [
                                {
                                    "id": "c-other",
                                    "title_en": "Notes",
                                    "title_zh": "笔记",
                                    "sections": [
                                        {
                                            "id": "s-body",
                                            "title_en": "Body",
                                            "title_zh": "正文",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
            )
            class _AssignFake(FakeLLMClient):
                def complete(self, **kwargs):
                    if kwargs.get("stage") == "assign":
                        headings = json.loads(
                            (artifacts / "quantum-notes" / "inventory" / "headings.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        payload = {
                            "assignments": [
                                {
                                    "section_id": row["section_id"],
                                    "primary_node_id": "s-body",
                                    "secondary_node_id": None,
                                }
                                for row in headings
                            ]
                        }
                        self.default = json.dumps(payload)
                    return super().complete(**kwargs)

            fake = _AssignFake(default=outline_json)
            code, out, err = _run_main(
                [
                    "quantum-notes",
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
            self.assertIn("[assign]", out)
            headings = json.loads(
                (artifacts / "quantum-notes" / "inventory" / "headings.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertGreaterEqual(len(headings), 2)
            assign_calls = [c for c in fake.calls if c["stage"] == "assign"]
            self.assertEqual(len(assign_calls), 1)
            user = assign_calls[0]["user"]
            for row in headings:
                self.assertIn(row["section_id"], user)
            self.assertTrue(assign_calls[0]["json_mode"])
            self.assertEqual(assign_calls[0]["max_tokens"], 2048)
            self.assertLessEqual(len(headings), ASSIGN_BATCH_SIZE)


class TestAssignCacheHashes(unittest.TestCase):
    def test_outline_json_edit_changes_assign_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts"
            fake = FakeLLMClient(default=SIMPLE_OUTLINE_JSON, cache_dir=artifacts / "cache")
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
            topic_art = artifacts / "two-book-overlap"
            outline_path = topic_art / "outline" / "outline.json"
            headings_path = topic_art / "inventory" / "headings.json"
            lexicon = load_lexicon("two-book-overlap")
            hashes_before = assign_stage_input_hashes(outline_path, headings_path, lexicon)
            self.assertIn(sha256_file(outline_path), hashes_before)

            data = json.loads(outline_path.read_text(encoding="utf-8"))
            data["title"]["en"] = "Edited For Cache"
            outline_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            hashes_after = assign_stage_input_hashes(outline_path, headings_path, lexicon)
            self.assertIn(sha256_file(outline_path), hashes_after)
            self.assertNotEqual(hashes_before, hashes_after)
            self.assertNotEqual(hashes_before[0], hashes_after[0])

            key_before = CacheKey(
                stage="assign",
                model="fake-model",
                prompt_hash="p" * 64,
                params={
                    "json_mode": True,
                    "max_input_chars": 12000,
                    "max_tokens": 2048,
                    "temperature": 0.2,
                },
                input_hashes=hashes_before,
            )
            key_after = CacheKey(
                stage="assign",
                model="fake-model",
                prompt_hash="p" * 64,
                params=key_before.params,
                input_hashes=hashes_after,
            )
            self.assertNotEqual(cache_key_hash(key_before), cache_key_hash(key_after))

            book_root = Path(tmp) / "book"
            topic_dir = book_root / "quantum-notes"
            topic_dir.mkdir(parents=True)
            (topic_dir / "a.md").write_text(
                "# 量子纠缠\n\n两个粒子无论相距多远都会保持关联，测量其中一个立即确定另一个状态。\n",
                encoding="utf-8",
            )
            q_art = Path(tmp) / "qart"
            q_fake = FakeLLMClient(
                default=json.dumps(
                    {
                        "title": {"en": "Q", "zh": "量", "yue_hint": "量"},
                        "parts": [
                            {
                                "id": "p-other",
                                "title_en": "Other",
                                "title_zh": "其他",
                                "chapters": [
                                    {
                                        "id": "c-other",
                                        "title_en": "Notes",
                                        "title_zh": "笔记",
                                        "sections": [
                                            {"id": "s-body", "title_en": "Body", "title_zh": "正文"}
                                        ],
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                cache_dir=q_art / "quantum-notes" / "cache",
            )

            def _assign_response(headings_file: Path) -> str:
                rows = json.loads(headings_file.read_text(encoding="utf-8"))
                return json.dumps(
                    {
                        "assignments": [
                            {
                                "section_id": row["section_id"],
                                "primary_node_id": "s-body",
                                "secondary_node_id": None,
                            }
                            for row in rows
                        ]
                    }
                )

            code_q, _o, err_q = _run_main(
                [
                    "quantum-notes",
                    "--book-root",
                    str(book_root),
                    "--artifacts-root",
                    str(q_art),
                    "--stage",
                    "outline",
                ],
                llm_client=q_fake,
            )
            self.assertEqual(code_q, 0, err_q)
            headings_q = q_art / "quantum-notes" / "inventory" / "headings.json"
            q_fake.default = _assign_response(headings_q)
            with redirect_stdout(io.StringIO()):
                run_assign(
                    topic="quantum-notes",
                    artifacts_root=q_art,
                    client=q_fake,
                )
            first_assign = [c for c in q_fake.calls if c["stage"] == "assign"]
            self.assertEqual(len(first_assign), 1)
            outline_q = q_art / "quantum-notes" / "outline" / "outline.json"
            self.assertIn(sha256_file(outline_q), first_assign[0]["input_hashes"])
            n_calls = len(q_fake.calls)
            with redirect_stdout(io.StringIO()):
                run_assign(
                    topic="quantum-notes",
                    artifacts_root=q_art,
                    client=q_fake,
                )
            self.assertEqual(len(q_fake.calls), n_calls)

            edited = json.loads(outline_q.read_text(encoding="utf-8"))
            edited["title"]["en"] = "Cache Buster"
            outline_q.write_text(json.dumps(edited, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            q_fake.default = _assign_response(headings_q)
            with redirect_stdout(io.StringIO()):
                run_assign(
                    topic="quantum-notes",
                    artifacts_root=q_art,
                    client=q_fake,
                )
            assign_calls = [c for c in q_fake.calls if c["stage"] == "assign"]
            self.assertEqual(len(assign_calls), 2)
            self.assertIn(sha256_file(outline_q), assign_calls[1]["input_hashes"])
            self.assertNotEqual(assign_calls[0]["input_hashes"], assign_calls[1]["input_hashes"])

    def test_stage_assign_without_outline_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            fake = FakeLLMClient()
            code, _out, err = _run_main(
                [
                    "two-book-overlap",
                    "--book-root",
                    str(TESTDATA),
                    "--artifacts-root",
                    str(artifacts),
                    "--stage",
                    "assign",
                ],
                llm_client=fake,
            )
            self.assertEqual(code, 2)
            self.assertIn("outline.json not found", err)


class TestAssignLengthFallback(unittest.TestCase):
    def _quantum_topic(self, tmp: Path) -> tuple[Path, Path]:
        book_root = tmp / "book"
        topic_dir = book_root / "quantum-notes"
        topic_dir.mkdir(parents=True)
        (topic_dir / "a.md").write_text(
            "# 量子纠缠\n\n两个粒子无论相距多远都会保持关联，测量其中一个立即确定另一个状态。\n",
            encoding="utf-8",
        )
        (topic_dir / "b.md").write_text(
            "# 波函数坍缩\n\n波函数坍缩描述观测如何把叠加态变成确定结果，这与表情或肢体语言完全无关。\n",
            encoding="utf-8",
        )
        artifacts = tmp / "artifacts"
        return book_root, artifacts

    def _outline_json(self) -> str:
        return json.dumps(
            {
                "title": {"en": "Q", "zh": "量", "yue_hint": "量"},
                "parts": [
                    {
                        "id": "p-other",
                        "title_en": "Other",
                        "title_zh": "其他",
                        "chapters": [
                            {
                                "id": "c-other",
                                "title_en": "Notes",
                                "title_zh": "笔记",
                                "sections": [
                                    {"id": "s-body", "title_en": "Body", "title_zh": "正文"}
                                ],
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        )

    def test_length_retries_smaller_batch_then_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book_root, artifacts = self._quantum_topic(Path(tmp))
            fake = FakeLLMClient(default=self._outline_json())
            code, _out, err = _run_main(
                [
                    "quantum-notes",
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
            headings = json.loads(
                (artifacts / "quantum-notes" / "inventory" / "headings.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertGreaterEqual(len(headings), 2)

            payload = json.dumps(
                {
                    "assignments": [
                        {
                            "section_id": row["section_id"],
                            "primary_node_id": "s-body",
                            "secondary_node_id": None,
                        }
                        for row in headings
                    ]
                }
            )
            retry_fake = FakeLLMClient(
                default=payload,
                finish_reasons=["length", "stop", "stop"],
            )
            with redirect_stdout(io.StringIO()):
                items = run_assign(
                    topic="quantum-notes",
                    artifacts_root=artifacts,
                    client=retry_fake,
                )
            assign_calls = [c for c in retry_fake.calls if c["stage"] == "assign"]
            self.assertEqual(len(assign_calls), 3)
            self.assertEqual({item.primary_node_id for item in items}, {"s-body"})
            self.assertEqual(len(items), len(headings))

    def test_length_then_invalid_uses_first_section_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book_root, artifacts = self._quantum_topic(Path(tmp))
            fake = FakeLLMClient(default=self._outline_json())
            code, _out, err = _run_main(
                [
                    "quantum-notes",
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
            outline = Outline.model_validate(
                json.loads(
                    (artifacts / "quantum-notes" / "outline" / "outline.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
            fallback = first_section_id(outline)
            headings = json.loads(
                (artifacts / "quantum-notes" / "inventory" / "headings.json").read_text(
                    encoding="utf-8"
                )
            )
            fail_fake = FakeLLMClient(
                default="NOT-JSON",
                finish_reasons=["length", "length", "length"],
            )
            with redirect_stdout(io.StringIO()):
                items = run_assign(
                    topic="quantum-notes",
                    artifacts_root=artifacts,
                    client=fail_fake,
                )
            self.assertTrue(items)
            self.assertEqual({item.primary_node_id for item in items}, {fallback})
            self.assertEqual(len(items), len(headings))
            self.assertGreaterEqual(len([c for c in fail_fake.calls if c["stage"] == "assign"]), 2)


if __name__ == "__main__":
    unittest.main()

