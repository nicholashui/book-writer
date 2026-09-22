"""Turn a markdown book into a spoken HK-Cantonese YouTube script using ytscript.txt."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from book_combiner.cli import _load_dotenv
from book_combiner.llm.base import completed_text
from book_combiner.llm.xai import build_client
from book_combiner.merge import atomic_write_text
from book_combiner.outline import sha256_text

HEADING_SPLIT = re.compile(r"(?=^#{1,3} )", re.M)
YAML_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n?", re.S)
STAGE = "ytscript-yue"


def split_chunks(markdown: str) -> list[str]:
    body = YAML_RE.sub("", markdown, count=1).strip()
    raw = [c.strip() for c in HEADING_SPLIT.split(body) if c.strip()]
    merged: list[str] = []
    buf = ""
    for chunk in raw:
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        heading_only = len(lines) == 1 and lines[0].startswith("#")
        if heading_only:
            buf = (buf + "\n\n" + chunk).strip() if buf else chunk
            continue
        if buf:
            chunk = buf + "\n\n" + chunk
            buf = ""
        merged.append(chunk)
    if buf:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + buf
        else:
            merged.append(buf)
    return merged


def _max_tokens(source_chars: int) -> int:
    # Spoken expansion is longer than the source; stay under grok-4 8k cap.
    return max(1024, min(8192, int(source_chars * 1.8) + 256))


def convert_chunk(*, client, system: str, chunk: str, index: int, total: int) -> str:
    lead = ""
    if index == 0:
        lead = "呢段係片頭。用香港 YouTube 講者開場，然後先講目錄，再入正題。\n\n"
    user = (
        lead
        + f"呢段係全書第 {index + 1} / {total} 段。直接輸出講稿，唔好 Markdown。\n\n"
        + "【原文】\n"
        + chunk
    )
    record = client.complete(
        stage=STAGE,
        prompt_hash=sha256_text(system),
        input_hashes=[sha256_text(user)],
        user=user,
        system=system,
        temperature=0.35,
        max_tokens=_max_tokens(len(chunk)),
        max_input_chars=max(len(user) + 100, 8000),
        json_mode=False,
    )
    text = completed_text(record)
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="book/facial-expression.summary.hk.md")
    parser.add_argument("--prompt", default="ytscript.txt")
    parser.add_argument("--out", default="book/facial-expression.summary.hk.script.txt")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--topic", default="facial-expression")
    parser.add_argument("--provider", default="xai")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    source = Path(args.source)
    prompt_path = Path(args.prompt)
    if not source.is_file():
        print(f"missing source {source}", file=sys.stderr)
        return 2
    if not prompt_path.is_file():
        print(f"missing prompt {prompt_path}", file=sys.stderr)
        return 2
    system = prompt_path.read_text(encoding="utf-8")
    chunks = split_chunks(source.read_text(encoding="utf-8"))
    model = args.model or os.environ.get("XAI_MODEL") or "grok-4"
    client = build_client(
        provider=args.provider,
        model=model,
        cache_dir=Path(args.artifacts_root) / args.topic / "cache",
    )
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        print(f"[ytscript] {i + 1}/{len(chunks)} {len(chunk)} chars {chunk.splitlines()[0][:40]}")
        parts.append(convert_chunk(client=client, system=system, chunk=chunk, index=i, total=len(chunks)))
    out = Path(args.out)
    atomic_write_text(out, "\n\n".join(parts) + "\n")
    print(f"[ytscript] {out.as_posix()} {out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
