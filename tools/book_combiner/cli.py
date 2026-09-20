"""combine-book CLI: parse flags and run implemented pipeline stages."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from book_combiner.assemble import AssembleError, run_assemble
from book_combiner.assign import run_assign
from book_combiner.discover import DiscoverError, discover, validate_topic
from book_combiner.extract import extract_from_manifest
from book_combiner.inventory import run_inventory
from book_combiner.llm.base import LLMError
from book_combiner.merge import MergeError, run_merge
from book_combiner.outline import run_outline
from book_combiner.translate import TranslateError, run_translate

IMPLEMENTED_STAGES = (
    "discover",
    "extract",
    "inventory",
    "outline",
    "assign",
    "merge",
    "translate-en",
    "translate-yue",
    "assemble",
)
DRY_RUN_STAGES = ("discover", "extract", "inventory")
LLM_STAGES = ("outline", "assign", "merge", "translate-en", "translate-yue")

STAGES = (
    "discover",
    "extract",
    "inventory",
    "outline",
    "assign",
    "merge",
    "translate-en",
    "translate-yue",
    "assemble",
)

STAGE_CHOICES = (*STAGES, "all")


def _load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ without overriding."""
    env_path = Path(path) if path is not None else Path(".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.lower().startswith("export "):
            key = key[7:].strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value

# Operator runbook for --help. Live facial-expression (~3.3 MB) is not a CI job.
HELP_EPILOG = """
Operator runbook (live facial-expression corpus, ~3.3 MB / eight markdown sources):

  Wall time: hours at --concurrency 1 (250-400+ LLM calls × 5-20s).
  Cost: tens to low hundreds USD on grok-4 (public list prices, unverified;
  cache hits reduce spend). The live compile is an operator run, not CI.

  Recommended path (inspect outline before merge spend):
    1. python tools/combine_book.py facial-expression --stage outline
       Inspect artifacts/facial-expression/outline/outline.md
       (edit outline.json if needed; that busts downstream cache keys).
    2. python tools/combine_book.py facial-expression --from-stage assign
       Runs assign through assemble. Cache hits skip LLM work (resume).

  Testdata (no live spend):
    python tools/combine_book.py two-book-overlap --book-root testdata --artifacts-root <tmp>
    Compiled testdata/<topic>.md and .hk.md are gitignored (/testdata/*.md);
    sources stay in testdata/<topic>/*.md. Do not commit compiled siblings.

  Cache / resume / --force:
    Each executed stage skips LLM work on cache hit unless --force.
    --force with --stage outline rebuilds outline only (downstream files go
    stale; a later --from-stage assign misses cache and rebuilds).
    --force with --from-stage outline rebuilds outline and everything after.
    --stage all (default) is --from-stage discover.
    Byte-identical compiled outputs happen only on a full cache hit.
    Re-run the same command to resume after a mid-pipeline LLM failure.

  There is no stage named cluster.
""".strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="combine-book",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Combine markdown sources for one topic into compiled books.",
        epilog=HELP_EPILOG,
    )
    parser.add_argument(
        "topic",
        help="Single path segment; subfolder name under --book-root",
    )
    parser.add_argument(
        "--book-root",
        type=Path,
        default=Path("book"),
        help="Root that contains <topic>/ source folders (default: book)",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Intermediate/cache root (default: artifacts)",
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        default="all",
        help=(
            "Run exactly one stage, or all (default: all). "
            "Stages: discover, extract, inventory, outline, assign, merge, "
            "translate-en, translate-yue, assemble. There is no cluster stage. "
            "all is --from-stage discover."
        ),
    )
    parser.add_argument(
        "--from-stage",
        choices=STAGES,
        default=None,
        help="Run this stage through assemble (cache hits skip LLM work unless --force)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Deterministic prefix only (no LLM calls)",
    )
    parser.add_argument(
        "--strict-topic",
        action="store_true",
        help="Park off-topic unique instruction in Appendix C",
    )
    parser.add_argument(
        "--provider",
        default="xai",
        help="LLM provider (default: xai)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("XAI_MODEL", "grok-4"),
        help="Model id (default: env XAI_MODEL or grok-4)",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=6000,
        help="Soft cap per merge/translate LLM call (default: 6000)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore cache for the stages being executed. "
            "--force --stage outline rebuilds outline only; "
            "--force --from-stage outline rebuilds outline and everything after"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max parallel LLM calls (default: 1)",
    )
    args = parser.parse_args(argv)
    args.llm = {
        "provider": args.provider,
        "model": args.model,
        "force": bool(args.force),
        "max_input_chars": args.max_input_chars,
    }
    return args


def build_llm_client(args: argparse.Namespace, **kwargs):
    """Build an LLMClient from CLI flags; cache lives under artifacts/<topic>/cache/."""
    from book_combiner.llm import build_client

    artifacts_root = Path(kwargs.pop("artifacts_root", args.artifacts_root))
    topic = kwargs.pop("topic", args.topic)
    cache_dir = artifacts_root / topic / "cache"
    return build_client(
        provider=args.provider,
        model=args.model,
        cache_dir=cache_dir,
        force=args.force,
        **kwargs,
    )


def stages_to_run(args: argparse.Namespace) -> list[str]:
    if args.from_stage:
        start = STAGES.index(args.from_stage)
        return list(STAGES[start:])
    if args.stage == "all":
        return list(STAGES)
    return [args.stage]


def main(argv: list[str] | None = None, llm_client=None) -> int:
    _load_dotenv()
    args = parse_args(argv)
    try:
        topic_dir = validate_topic(args.topic, args.book_root)
    except DiscoverError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    requested = stages_to_run(args)
    if args.dry_run:
        llm_requested = [s for s in requested if s not in DRY_RUN_STAGES]
        stages = [s for s in requested if s in DRY_RUN_STAGES]
        if llm_requested and llm_requested[0] not in IMPLEMENTED_STAGES:
            print(
                f"Stage {llm_requested[0]!r} is not implemented until a later PR.",
                file=sys.stderr,
            )
            return 1
        if not stages:
            stages = list(DRY_RUN_STAGES)
    else:
        stages = requested

    if not stages or stages[0] not in IMPLEMENTED_STAGES:
        first = stages[0] if stages else (args.from_stage or args.stage)
        print(
            f"Stage {first!r} is not implemented until a later PR.",
            file=sys.stderr,
        )
        return 1

    try:
        manifest = discover(
            topic=args.topic,
            book_root=args.book_root,
            artifacts_root=args.artifacts_root,
            topic_dir=topic_dir,
        )
    except DiscoverError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    last_done = "discover"
    need_extract = any(
        s in stages
        for s in (
            "extract",
            "inventory",
            "outline",
            "assign",
            "merge",
            "translate-en",
            "translate-yue",
            "assemble",
        )
    )
    need_inventory = any(
        s in stages
        for s in (
            "inventory",
            "outline",
            "assign",
            "merge",
            "translate-en",
            "translate-yue",
            "assemble",
        )
    )
    if need_extract:
        extract_from_manifest(
            manifest,
            args.artifacts_root,
            max_input_chars=args.max_input_chars,
        )
        last_done = "extract"
    if need_inventory:
        run_inventory(manifest, args.artifacts_root)
        last_done = "inventory"

    need_llm = any(s in stages for s in LLM_STAGES)
    client = llm_client
    if need_llm and client is None:
        try:
            client = build_llm_client(args)
        except LLMError as exc:
            print(str(exc), file=sys.stderr)
            return 3

    try:
        if "outline" in stages:
            run_outline(
                topic=args.topic,
                artifacts_root=args.artifacts_root,
                client=client,
                model=args.model,
                force=args.force,
            )
            last_done = "outline"
        if "assign" in stages:
            outline_path = args.artifacts_root / args.topic / "outline" / "outline.json"
            if not outline_path.is_file():
                print(
                    "outline.json not found; run --stage outline first.",
                    file=sys.stderr,
                )
                return 2
            run_assign(
                topic=args.topic,
                artifacts_root=args.artifacts_root,
                client=client,
                model=args.model,
                force=args.force,
            )
            last_done = "assign"
        if "merge" in stages:
            assignment_path = args.artifacts_root / args.topic / "outline" / "assignment.json"
            outline_path = args.artifacts_root / args.topic / "outline" / "outline.json"
            if not outline_path.is_file():
                print(
                    "outline.json not found; run --stage outline first.",
                    file=sys.stderr,
                )
                return 2
            if not assignment_path.is_file():
                print(
                    "assignment.json not found; run --stage assign first.",
                    file=sys.stderr,
                )
                return 2
            run_merge(
                topic=args.topic,
                artifacts_root=args.artifacts_root,
                client=client,
                model=args.model,
                force=args.force,
                max_input_chars=args.max_input_chars,
            )
            last_done = "merge"
        if "translate-en" in stages:
            if client is None:
                print("LLM client required for translate-en.", file=sys.stderr)
                return 3
            run_translate(
                topic=args.topic,
                artifacts_root=args.artifacts_root,
                client=client,
                stage="translate-en",
                model=args.model,
                force=args.force,
                max_input_chars=args.max_input_chars,
                strict_topic=args.strict_topic,
            )
            last_done = "translate-en"
        if "translate-yue" in stages:
            if client is None:
                print("LLM client required for translate-yue.", file=sys.stderr)
                return 3
            run_translate(
                topic=args.topic,
                artifacts_root=args.artifacts_root,
                client=client,
                stage="translate-yue",
                model=args.model,
                force=args.force,
                max_input_chars=args.max_input_chars,
                strict_topic=args.strict_topic,
            )
            last_done = "translate-yue"
        if "assemble" in stages:
            run_assemble(
                topic=args.topic,
                book_root=args.book_root,
                artifacts_root=args.artifacts_root,
                model=args.model,
                strict_topic=args.strict_topic,
            )
            last_done = "assemble"
    except MergeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except TranslateError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except AssembleError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    except LLMError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    later = [s for s in stages if s not in IMPLEMENTED_STAGES]
    if later:
        print(
            f"Remaining stages after {last_done} are not implemented until a later PR.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
