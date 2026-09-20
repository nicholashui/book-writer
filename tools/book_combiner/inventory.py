"""Heading inventory and overlap index from extract artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from book_combiner.discover import Manifest
from book_combiner.lexicon import keywords_for, load_lexicon
from book_combiner.models import Section
from book_combiner.overlap import compute_overlap

EXCERPT_CHARS = 40
_WS_RE = re.compile(r"\s+")


def make_excerpt(text: str, n: int = EXCERPT_CHARS) -> str:
    flat = _WS_RE.sub("", text.strip())
    if len(flat) > n:
        return flat[:n] + "……"
    return flat


def heading_row(section: Section, lexicon: dict[str, list[str]]) -> dict[str, object]:
    return {
        "section_id": section.id,
        "heading_path": list(section.heading_path),
        "keywords": keywords_for(section.heading_path, lexicon),
        "excerpt": make_excerpt(section.text),
        "char_count": section.char_count,
        "kind": section.kind,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_inventory(manifest: Manifest, artifacts_root: Path) -> dict:
    extract_dir = artifacts_root / manifest.topic / "extract"
    inv_dir = artifacts_root / manifest.topic / "inventory"
    inv_dir.mkdir(parents=True, exist_ok=True)

    files: list[tuple[Path, list[Section]]] = []
    all_sections: list[Section] = []
    for source in manifest.sources:
        path = extract_dir / f"{source.stem}.sections.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        secs = [Section(**row) for row in raw]
        files.append((path, secs))
        all_sections.extend(secs)

    lexicon = load_lexicon(manifest.topic)
    headings = [heading_row(sec, lexicon) for sec in all_sections]
    overlap = compute_overlap(all_sections)

    for sec in all_sections:
        sec.duplicate_of = overlap["duplicate_of"].get(sec.id)

    for path, secs in files:
        _write_json(path, [asdict(s) for s in secs])
    _write_json(inv_dir / "headings.json", headings)
    _write_json(inv_dir / "overlap.json", overlap)

    n_pairs = overlap["dominant_exactish_count"]
    stems = overlap["dominant_exactish_stems"]
    if stems:
        print(
            f"[inventory] headings={len(headings)} overlap pairs={n_pairs} "
            f"exactish {stems[0]} <-> {stems[1]}"
        )
    else:
        print(f"[inventory] headings={len(headings)} overlap pairs={n_pairs}")
    return overlap
