"""Discover top-level markdown sources for a topic and write manifest.json."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

TOPIC_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")


class DiscoverError(Exception):
    """Usage / empty-input failure (CLI exit 2)."""


@dataclass
class SourceEntry:
    path: str
    stem: str
    sha256: str
    bytes: int
    chars: int
    physical_lines: int


@dataclass
class Manifest:
    topic: str
    book_root: str
    sources: list[SourceEntry]
    skipped_epub: list[str]
    generated_at: str


def validate_topic(topic: str, book_root: Path) -> Path:
    """Return `<book-root>/<topic>` after rejecting traversal and bad names."""
    if not TOPIC_NAME_RE.fullmatch(topic):
        raise DiscoverError(f"Invalid topic name: {topic}")
    root_resolved = book_root.resolve()
    topic_dir_resolved = (root_resolved / topic).resolve()
    if not topic_dir_resolved.is_relative_to(root_resolved):
        raise DiscoverError(f"Invalid topic name: {topic}")
    return book_root / topic


def _is_excluded_markdown(name: str, topic: str) -> bool:
    if name.startswith("."):
        return True
    if name.endswith(".zh.md") or name.endswith(".en.md"):
        return True
    if name.startswith("compiled") and name.endswith(".md"):
        return True
    if name == f"{topic}.md" or name == f"{topic}.hk.md":
        return True
    return False


def _posix(path: Path) -> str:
    return path.as_posix()


def discover(
    topic: str,
    book_root: Path,
    artifacts_root: Path,
    topic_dir: Path | None = None,
) -> Manifest:
    topic_dir = topic_dir if topic_dir is not None else validate_topic(topic, book_root)
    display = f"{_posix(book_root).rstrip('/')}/{topic}/"
    if not topic_dir.is_dir():
        raise DiscoverError(f"No markdown sources in {display}")

    topic_resolved = topic_dir.resolve()
    artifacts_topic = (artifacts_root / topic).resolve()
    nested_artifacts = artifacts_topic != topic_resolved and artifacts_topic.is_relative_to(
        topic_resolved
    )

    skipped_epub: list[str] = []
    sources: list[SourceEntry] = []
    for entry in sorted(topic_dir.iterdir(), key=lambda p: p.name):
        if not entry.is_file():
            continue
        if nested_artifacts and entry.resolve().is_relative_to(artifacts_topic):
            continue
        name = entry.name
        if name.startswith("."):
            continue
        if name.endswith(".epub"):
            skipped_epub.append(name)
            print(f"[discover] skipped epub={name}")
            continue
        if not name.endswith(".md"):
            continue
        if _is_excluded_markdown(name, topic):
            continue
        path_display = _posix(book_root / topic / name)
        try:
            raw = entry.read_bytes()
            text = entry.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise DiscoverError(f"Cannot read {path_display} as UTF-8") from None
        except OSError as exc:
            raise DiscoverError(f"Cannot read {path_display}: {exc}") from exc
        sources.append(
            SourceEntry(
                path=path_display,
                stem=entry.stem,
                sha256=hashlib.sha256(raw).hexdigest(),
                bytes=entry.stat().st_size,
                chars=len(text),
                physical_lines=len(text.splitlines()),
            )
        )

    if not sources:
        raise DiscoverError(f"No markdown sources in {display}")

    manifest = Manifest(
        topic=topic,
        book_root=_posix(book_root),
        sources=sources,
        skipped_epub=skipped_epub,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    out_dir = artifacts_root / topic
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manifest.json"
    out_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    total_bytes = sum(s.bytes for s in sources)
    print(
        f"[discover] topic={topic} md={len(sources)} "
        f"epub_skipped={len(skipped_epub)} bytes={total_bytes}"
    )
    return manifest
