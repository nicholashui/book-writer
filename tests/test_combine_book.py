"""Unit tests for combine-book CLI scaffold and discover."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from book_combiner.cli import _load_dotenv, main, parse_args, stages_to_run  # noqa: E402
from book_combiner.discover import DiscoverError, discover, validate_topic  # noqa: E402
from book_combiner.models import CacheKey, CacheRecord, Section  # noqa: E402


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


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


class TestLoadDotenv(unittest.TestCase):
    def test_loads_missing_keys_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            env_file = Path(raw) / ".env"
            env_file.write_text("XAI_MODEL=from-file\n", encoding="utf-8")
            os.environ.pop("DOTENV_TEST_KEY", None)
            previous = os.environ.get("XAI_MODEL")
            os.environ["XAI_MODEL"] = "already-set"
            try:
                _load_dotenv(env_file)
                self.assertEqual(os.environ.get("XAI_MODEL"), "already-set")
                env_file.write_text("DOTENV_TEST_KEY=hello\n", encoding="utf-8")
                _load_dotenv(env_file)
                self.assertEqual(os.environ.get("DOTENV_TEST_KEY"), "hello")
            finally:
                os.environ.pop("DOTENV_TEST_KEY", None)
                if previous is None:
                    os.environ.pop("XAI_MODEL", None)
                else:
                    os.environ["XAI_MODEL"] = previous


class TestParseArgs(unittest.TestCase):
    def test_flag_parsing(self) -> None:
        ns = parse_args(
            [
                "facial-expression",
                "--stage",
                "translate-yue",
                "--from-stage",
                "assign",
                "--strict-topic",
                "--max-input-chars",
                "123",
                "--force",
                "--concurrency",
                "2",
                "--provider",
                "xai",
                "--model",
                "grok-4",
            ]
        )
        self.assertEqual(ns.topic, "facial-expression")
        self.assertEqual(ns.stage, "translate-yue")
        self.assertEqual(ns.from_stage, "assign")
        self.assertTrue(ns.strict_topic)
        self.assertEqual(ns.max_input_chars, 123)
        self.assertTrue(ns.force)
        self.assertEqual(ns.concurrency, 2)
        self.assertEqual(ns.provider, "xai")
        self.assertEqual(ns.model, "grok-4")

    def test_defaults(self) -> None:
        ns = parse_args(["demo"])
        self.assertEqual(ns.book_root, Path("book"))
        self.assertEqual(ns.artifacts_root, Path("artifacts"))
        self.assertEqual(ns.stage, "all")
        self.assertIsNone(ns.from_stage)
        self.assertFalse(ns.dry_run)
        self.assertFalse(ns.strict_topic)
        self.assertEqual(ns.provider, "xai")
        self.assertEqual(ns.max_input_chars, 6000)
        self.assertFalse(ns.force)
        self.assertEqual(ns.concurrency, 1)
        expected_model = os.environ.get("XAI_MODEL", "grok-4")
        self.assertEqual(ns.model, expected_model)

    def test_stage_all_is_from_discover(self) -> None:
        ns = parse_args(["demo"])
        self.assertEqual(stages_to_run(ns)[0], "discover")
        self.assertEqual(stages_to_run(ns), list(stages_to_run(ns)))
        ns_from = parse_args(["demo", "--from-stage", "discover"])
        self.assertEqual(stages_to_run(ns), stages_to_run(ns_from))

    def test_no_cluster_stage(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            parse_args(["demo", "--stage", "cluster"])
        self.assertEqual(ctx.exception.code, 2)


class TestDiscover(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.book_root = self.tmp / "book"
        self.artifacts = self.tmp / "artifacts"
        self.topic = "demo"
        self.topic_dir = self.book_root / self.topic
        self.topic_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_finds_two_markdown_sources_and_skips_epub(self) -> None:
        a = _write(self.topic_dir / "a.md", "hello a\n")
        b = _write(self.topic_dir / "b.md", "hello b\nmore\n")
        _write(self.topic_dir / "skipped.epub", "not markdown")
        _write(self.book_root / "outside.md", "should not be ingested")
        nested = self.topic_dir / "nested"
        nested.mkdir()
        _write(nested / "deep.md", "not top-level")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            manifest = discover(self.topic, self.book_root, self.artifacts)

        self.assertEqual({s.stem for s in manifest.sources}, {"a", "b"})
        self.assertEqual(manifest.skipped_epub, ["skipped.epub"])
        self.assertIn("skipped epub=skipped.epub", stdout.getvalue())

        manifest_path = self.artifacts / self.topic / "manifest.json"
        self.assertTrue(manifest_path.is_file())
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(data["topic"], "demo")
        self.assertEqual(len(data["sources"]), 2)
        self.assertEqual(data["skipped_epub"], ["skipped.epub"])
        stems = {s["stem"] for s in data["sources"]}
        self.assertEqual(stems, {"a", "b"})

        by_stem = {s["stem"]: s for s in data["sources"]}
        for path, stem in ((a, "a"), (b, "b")):
            entry = by_stem[stem]
            text = path.read_text(encoding="utf-8")
            self.assertEqual(entry["chars"], len(text))
            self.assertEqual(entry["bytes"], path.stat().st_size)
            self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(entry["physical_lines"], len(text.splitlines()))
            self.assertEqual(entry["path"], (self.book_root / self.topic / path.name).as_posix())

        code, out, err = _run_main(
            [
                self.topic,
                "--book-root",
                str(self.book_root),
                "--artifacts-root",
                str(self.artifacts),
                "--stage",
                "discover",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("[discover]", out)
        self.assertEqual(err, "")

    def test_empty_folder_exits_2(self) -> None:
        code, _out, err = _run_main(
            [
                self.topic,
                "--book-root",
                str(self.book_root),
                "--artifacts-root",
                str(self.artifacts),
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("No markdown sources in", err)
        self.assertFalse((self.artifacts / self.topic / "manifest.json").exists())

    def test_missing_md_exits_2(self) -> None:
        _write(self.topic_dir / "only.epub", "epub")
        code, out, err = _run_main(
            [
                self.topic,
                "--book-root",
                str(self.book_root),
                "--artifacts-root",
                str(self.artifacts),
                "--stage",
                "discover",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("No markdown sources in", err)
        self.assertIn("skipped epub=only.epub", out)

    def test_missing_topic_dir_exits_2(self) -> None:
        code, _out, err = _run_main(
            [
                "absent",
                "--book-root",
                str(self.book_root),
                "--artifacts-root",
                str(self.artifacts),
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("No markdown sources in", err)

    def test_invalid_topics_exit_2(self) -> None:
        for topic in ("../etc", "foo/bar", "..", ".hidden", "has space"):
            with self.subTest(topic=topic):
                code, _out, err = _run_main(
                    [
                        topic,
                        "--book-root",
                        str(self.book_root),
                        "--artifacts-root",
                        str(self.artifacts),
                    ]
                )
                self.assertEqual(code, 2)
                self.assertIn("Invalid topic name", err)
                with self.assertRaises(DiscoverError):
                    validate_topic(topic, self.book_root)

    def test_excludes_compiled_looking_names(self) -> None:
        _write(self.topic_dir / "keep.md", "keep me\n")
        _write(self.topic_dir / f"{self.topic}.md", "compiled sibling")
        _write(self.topic_dir / f"{self.topic}.hk.md", "compiled hk")
        _write(self.topic_dir / "compiled.zh.md", "compiled zh")
        _write(self.topic_dir / "foo.en.md", "english dump")
        _write(self.topic_dir / "compiled.md", "compiled prefix")
        _write(self.topic_dir / ".hidden.md", "dotfile")
        _write(self.book_root / f"{self.topic}.md", "book-root compiled")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            manifest = discover(self.topic, self.book_root, self.artifacts)
        self.assertEqual([s.stem for s in manifest.sources], ["keep"])

    def test_does_not_pick_up_files_outside_topic_subfolder(self) -> None:
        _write(self.topic_dir / "inside.md", "in")
        other = self.book_root / "other-topic"
        other.mkdir()
        _write(other / "other.md", "out")
        _write(self.book_root / "demo.md", "compiled-looking at root")
        with redirect_stdout(io.StringIO()):
            manifest = discover(self.topic, self.book_root, self.artifacts)
        self.assertEqual([s.stem for s in manifest.sources], ["inside"])

    def test_stage_inventory_completes_prefix(self) -> None:
        _write(self.topic_dir / "a.md", "one")
        _write(self.topic_dir / "b.md", "two")
        code, out, err = _run_main(
            [
                self.topic,
                "--book-root",
                str(self.book_root),
                "--artifacts-root",
                str(self.artifacts),
                "--stage",
                "inventory",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("[extract]", out)
        self.assertIn("[inventory]", out)
        self.assertNotIn("not implemented until a later PR", err)
        self.assertTrue((self.artifacts / self.topic / "manifest.json").is_file())
        self.assertTrue((self.artifacts / self.topic / "inventory" / "headings.json").is_file())

    def test_translate_en_without_merge_nodes_exits_3(self) -> None:
        tests_dir = str(Path(__file__).resolve().parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        from test_outline import FakeLLMClient

        _write(self.topic_dir / "a.md", "one")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                [
                    self.topic,
                    "--book-root",
                    str(self.book_root),
                    "--artifacts-root",
                    str(self.artifacts),
                    "--stage",
                    "translate-en",
                ],
                llm_client=FakeLLMClient(),
            )
        err = stderr.getvalue()
        self.assertEqual(code, 3)
        self.assertIn("merge nodes not found", err)
        self.assertNotIn("not implemented until a later PR", err)

    def test_dry_run_translate_skips_llm(self) -> None:
        _write(self.topic_dir / "a.md", "one")
        for extra in (["--stage", "translate-en"], ["--from-stage", "translate-en"]):
            with self.subTest(extra=extra):
                code, out, err = _run_main(
                    [
                        self.topic,
                        "--book-root",
                        str(self.book_root),
                        "--artifacts-root",
                        str(self.artifacts),
                        "--dry-run",
                        *extra,
                    ]
                )
                self.assertEqual(code, 0, err)
                self.assertIn("[inventory]", out)
                self.assertNotIn("[translate-en]", out)
                self.assertNotIn("not implemented until a later PR", err)

    def test_dry_run_outline_skips_llm(self) -> None:
        _write(self.topic_dir / "a.md", "one")
        code, out, err = _run_main(
            [
                self.topic,
                "--book-root",
                str(self.book_root),
                "--artifacts-root",
                str(self.artifacts),
                "--dry-run",
                "--stage",
                "outline",
            ]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("[inventory]", out)
        self.assertNotIn("[outline]", out)
        self.assertFalse((self.artifacts / self.topic / "outline" / "outline.json").exists())

    def test_dry_run_all_runs_discover_extract_inventory(self) -> None:
        _write(self.topic_dir / "a.md", "one")
        code, out, err = _run_main(
            [
                self.topic,
                "--book-root",
                str(self.book_root),
                "--artifacts-root",
                str(self.artifacts),
                "--dry-run",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("[discover]", out)
        self.assertIn("[extract]", out)
        self.assertIn("[inventory]", out)
        self.assertNotIn("not implemented until a later PR", err)
        self.assertTrue((self.artifacts / self.topic / "manifest.json").is_file())
        self.assertTrue((self.artifacts / self.topic / "inventory" / "headings.json").is_file())

    def test_dry_run_help_omits_this_pr(self) -> None:
        captured = io.StringIO()
        with redirect_stdout(captured), self.assertRaises(SystemExit) as ctx:
            parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        help_text = captured.getvalue()
        self.assertNotIn("in this PR", help_text)
        self.assertIn("Deterministic prefix only", help_text)
        self.assertIn("hours", help_text)
        self.assertIn("tens to low hundreds USD", help_text)
        self.assertIn("grok-4", help_text)
        self.assertIn("--stage outline", help_text)
        self.assertIn("--from-stage assign", help_text)
        self.assertIn("no stage named cluster", help_text.lower())

    def test_ancestor_artifacts_root_still_ingests(self) -> None:
        _write(self.topic_dir / "a.md", "keep me\n")
        _write(self.topic_dir / "skipped.epub", "epub")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            manifest = discover(self.topic, self.book_root, self.tmp)
        self.assertEqual([s.stem for s in manifest.sources], ["a"])
        self.assertEqual(manifest.skipped_epub, ["skipped.epub"])
        self.assertTrue((self.tmp / self.topic / "manifest.json").is_file())

        code, _out, _err = _run_main(
            [
                self.topic,
                "--book-root",
                str(self.book_root),
                "--artifacts-root",
                str(self.book_root),
                "--stage",
                "discover",
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue((self.book_root / self.topic / "manifest.json").is_file())
        data = json.loads(
            (self.book_root / self.topic / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual([s["stem"] for s in data["sources"]], ["a"])

    def test_non_utf8_markdown_exits_2(self) -> None:
        (self.topic_dir / "bad.md").write_bytes(b"\xff\xfe not utf-8")
        code, _out, err = _run_main(
            [
                self.topic,
                "--book-root",
                str(self.book_root),
                "--artifacts-root",
                str(self.artifacts),
                "--stage",
                "discover",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("bad.md", err)
        self.assertIn("UTF-8", err)
        self.assertFalse((self.artifacts / self.topic / "manifest.json").exists())

    def test_models_importable(self) -> None:
        section = Section(
            id="a:1:2",
            source_stem="a",
            source_book="a",
            heading_path=["Ch"],
            level=1,
            start_line=1,
            end_line=2,
            text="x",
            kind="body",
            char_count=1,
            content_hash="h",
            duplicate_of=None,
        )
        key = CacheKey(
            stage="discover",
            model="grok-4",
            prompt_hash="p",
            params={},
            input_hashes=[],
        )
        record = CacheRecord(
            key=key,
            response_text="",
            finish_reason="stop",
            input_tokens=0,
            output_tokens=0,
            created_at="2026-09-18T00:00:00Z",
        )
        self.assertEqual(section.kind, "body")
        self.assertEqual(record.key.stage, "discover")


class TestShim(unittest.TestCase):
    def test_shim_discover(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            topic_dir = tmp_path / "book" / "demo"
            topic_dir.mkdir(parents=True)
            (topic_dir / "a.md").write_text("one\n", encoding="utf-8")
            (topic_dir / "b.md").write_text("two\n", encoding="utf-8")
            artifacts = tmp_path / "artifacts"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "combine_book.py"),
                    "demo",
                    "--book-root",
                    str(tmp_path / "book"),
                    "--artifacts-root",
                    str(artifacts),
                    "--stage",
                    "discover",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads((artifacts / "demo" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["sources"]), 2)


class TestFromStageSkipsExtract(unittest.TestCase):
    def test_from_stage_assemble_does_not_rewrite_fresh_extract(self) -> None:
        import os
        import time

        with tempfile.TemporaryDirectory() as tmp:
            book_root = Path(tmp) / "book"
            topic_dir = book_root / "demo"
            topic_dir.mkdir(parents=True)
            src = topic_dir / "a.md"
            src.write_text("# Hi\n\nbody text for extract.\n", encoding="utf-8")
            artifacts = Path(tmp) / "artifacts"
            extract_dir = artifacts / "demo" / "extract"
            extract_dir.mkdir(parents=True)
            sections = extract_dir / "a.sections.json"
            sections.write_text("[]\n", encoding="utf-8")
            headings = artifacts / "demo" / "inventory" / "headings.json"
            headings.parent.mkdir(parents=True)
            headings.write_text("[]\n", encoding="utf-8")
            time.sleep(0.05)
            os.utime(sections, None)
            os.utime(headings, None)
            sec_mtime = sections.stat().st_mtime
            head_mtime = headings.stat().st_mtime
            code, out, err = _run_main(
                [
                    "demo",
                    "--book-root",
                    str(book_root),
                    "--artifacts-root",
                    str(artifacts),
                    "--from-stage",
                    "assemble",
                ]
            )
            self.assertNotIn("sections=", out)
            self.assertEqual(sections.read_text(encoding="utf-8"), "[]\n")
            self.assertEqual(sections.stat().st_mtime, sec_mtime)
            self.assertEqual(headings.stat().st_mtime, head_mtime)
            self.assertEqual(headings.read_text(encoding="utf-8"), "[]\n")


if __name__ == "__main__":
    unittest.main()
