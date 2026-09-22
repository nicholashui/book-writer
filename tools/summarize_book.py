"""Uninstalled entry: python tools/summarize_book.py [topic]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from book_combiner.cli import _load_dotenv
from book_combiner.llm.xai import build_client
from book_combiner.outline import load_outline
from book_combiner.summarize import run_summarize_pair


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Summarize compiled EN/HK books to ~5% with aligned headings.")
    parser.add_argument("topic", nargs="?", default="facial-expression")
    parser.add_argument("--book-root", default="book")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--provider", default="xai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    model = args.model or __import__("os").environ.get("XAI_MODEL") or "grok-4"
    book_root = Path(args.book_root)
    en_path = book_root / f"{args.topic}.md"
    hk_path = book_root / f"{args.topic}.hk.md"
    if not en_path.is_file() or not hk_path.is_file():
        print(f"missing compiled books: {en_path} {hk_path}", file=sys.stderr)
        return 2
    outline = load_outline(Path(args.artifacts_root) / args.topic / "outline" / "outline.json")
    cache_dir = Path(args.artifacts_root) / args.topic / "cache"
    client = build_client(provider=args.provider, model=model, cache_dir=cache_dir, force=args.force)
    out_en = book_root / f"{args.topic}.summary.md"
    out_hk = book_root / f"{args.topic}.summary.hk.md"
    run_summarize_pair(
        en_path=en_path,
        hk_path=hk_path,
        outline=outline,
        client=client,
        out_en=out_en,
        out_hk=out_hk,
    )
    from book_combiner.summarize import tighten_markdown

    en_text = out_en.read_text(encoding="utf-8")
    if len(en_text) > 120_000:
        print("[summarize-en] second pass to ~50 pages")
        tight = tighten_markdown(
            text=en_text,
            client=client,
            language="English",
            stage="summarize-en-tight",
            ratio=0.35,
        )
        from book_combiner.merge import atomic_write_text

        atomic_write_text(out_en, tight)
        print(f"[summarize] {out_en.as_posix()} {len(tight)} chars (tightened)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
