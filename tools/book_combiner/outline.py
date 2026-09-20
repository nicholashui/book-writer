"""Topic-driven outline from the heading inventory. Always bucket-split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from book_combiner.lexicon import LEXICON, keywords_for, load_lexicon
from book_combiner.llm.base import LLMClient
from book_combiner.models import (
    Outline,
    OutlineChapter,
    OutlinePart,
    OutlineSection,
    OutlineTitle,
)
from book_combiner.overlap import normalize_text

OUTLINE_TEMPERATURE = 0.3
OUTLINE_MAX_TOKENS = 4096
HEADING_TOKEN_CAP = 24_000
DEFAULT_CHARS_PER_TOKEN = 1.5
OTHER_BUCKET = "other"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_TOPICS_DIR = Path(__file__).resolve().parent / "topics"
OUTLINE_PROMPT_PATH = _PROMPTS_DIR / "outline.txt"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def humanize_topic(topic: str) -> str:
    return topic.replace("-", " ").replace("_", " ").title()


def fill_template(template: str, **kwargs: str) -> str:
    """Replace {placeholders} without interpreting braces inside JSON payloads."""
    out = template
    for key, value in kwargs.items():
        out = out.replace("{" + key + "}", value)
    return out


def split_prompt_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("---") and stripped.endswith("---") and len(stripped) > 6:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = stripped.strip("-").lower()
            buf = []
        else:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def load_outline_templates(path: Path | None = None) -> dict[str, str]:
    return split_prompt_sections((path or OUTLINE_PROMPT_PATH).read_text(encoding="utf-8"))


def fewshot_path(topic: str, topics_dir: Path | None = None) -> Path:
    return (topics_dir or _TOPICS_DIR) / topic / "outline_fewshot.json"


def load_fewshot(topic: str, topics_dir: Path | None = None) -> dict | None:
    path = fewshot_path(topic, topics_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fewshot_hash(topic: str, topics_dir: Path | None = None) -> str:
    path = fewshot_path(topic, topics_dir)
    if not path.is_file():
        return sha256_text("none")
    return sha256_file(path)


def format_fewshot_block(fewshot: dict | None) -> str:
    if not fewshot:
        return ""
    return (
        "Example of outline shape (ids and nesting only; do not copy these topics):\n"
        + json.dumps(fewshot, ensure_ascii=False, indent=2)
    )


def load_chars_per_token(artifacts_topic: Path, default: float = DEFAULT_CHARS_PER_TOKEN) -> float:
    path = artifacts_topic / "logs" / "token_stats.json"
    if not path.is_file():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    if not isinstance(data, dict):
        return default
    raw = data.get("chars_per_token")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def estimate_tokens(text: str, chars_per_token: float) -> int:
    if not text:
        return 0
    cpt = chars_per_token if chars_per_token > 0 else DEFAULT_CHARS_PER_TOKEN
    return max(1, int(len(text) / cpt))


def primary_bucket(heading_path: list[str], lexicon: dict[str, list[str]]) -> str:
    hits = keywords_for(heading_path, lexicon)
    return hits[0] if hits else OTHER_BUCKET


def partition_by_bucket(
    headings: list[dict],
    lexicon: dict[str, list[str]],
) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for row in headings:
        path = list(row.get("heading_path") or [])
        bucket = primary_bucket(path, lexicon)
        buckets.setdefault(bucket, []).append(row)
    return buckets


def bucket_order(lexicon: dict[str, list[str]], present: set[str]) -> list[str]:
    ordered = [key for key in lexicon if key in present]
    if OTHER_BUCKET in present and OTHER_BUCKET not in ordered:
        ordered.append(OTHER_BUCKET)
    for key in sorted(present):
        if key not in ordered:
            ordered.append(key)
    return ordered


def format_heading_line(row: dict) -> str:
    section_id = str(row.get("section_id") or "")
    stem = section_id.split(":")[0] if section_id else ""
    path = ">".join(row.get("heading_path") or [])
    chars = row.get("char_count", 0)
    excerpt = row.get("excerpt") or ""
    return f"{stem}|{path}|{chars}|{excerpt}"


def pack_heading_rows(
    rows: list[dict],
    chars_per_token: float,
    token_cap: int = HEADING_TOKEN_CAP,
) -> list[list[dict]]:
    """Pack heading dump lines until estimated tokens hit the cap."""
    packs: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0
    cap = max(1, token_cap)
    for row in rows:
        tok = estimate_tokens(format_heading_line(row), chars_per_token)
        if current and current_tokens + tok > cap:
            packs.append(current)
            current = []
            current_tokens = 0
        current.append(row)
        current_tokens += tok
    if current:
        packs.append(current)
    return packs


def collect_outline_ids(outline: Outline) -> list[str]:
    ids: list[str] = []
    for part in outline.parts:
        ids.append(part.id)
        for chapter in part.chapters:
            ids.append(chapter.id)
            for section in chapter.sections:
                ids.append(section.id)
    return ids


def parse_outline(text: str) -> Outline | None:
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        outline = Outline.model_validate(data)
    except ValidationError:
        return None
    if not outline.parts:
        return None
    ids = collect_outline_ids(outline)
    if not all(ids) or len(ids) != len(set(ids)):
        return None
    return outline


def _bucket_titles(bucket: str, lexicon: dict[str, list[str]]) -> tuple[str, str]:
    if bucket == OTHER_BUCKET:
        return "Other", "其他"
    title_en = bucket.replace("_", " ").title()
    pats = lexicon.get(bucket) or LEXICON.get(bucket) or [bucket]
    title_zh = pats[0]
    return title_en, title_zh


def fallback_outline_for_bucket(
    bucket: str,
    headings: list[dict],
    topic: str,
    lexicon: dict[str, list[str]] | None = None,
    pack_index: int = 0,
) -> Outline:
    """One part named after the bucket; chapters from heading_path[0]; sections from [1] or body."""
    table = lexicon if lexicon is not None else LEXICON
    title_en, title_zh = _bucket_titles(bucket, table)
    tag = bucket if pack_index == 0 else f"{bucket}-{pack_index}"
    chapters_map: dict[str, tuple[str, dict[str, str]]] = {}
    for row in headings:
        path = [str(p) for p in (row.get("heading_path") or [])]
        ch_title = path[0] if path else "body"
        sec_title = path[1] if len(path) > 1 else "body"
        ch_key = normalize_text(ch_title) or "body"
        sec_key = normalize_text(sec_title) or "body"
        if ch_key not in chapters_map:
            chapters_map[ch_key] = (ch_title, {})
        sec_map = chapters_map[ch_key][1]
        if sec_key not in sec_map:
            sec_map[sec_key] = sec_title

    chapters: list[OutlineChapter] = []
    for i, (_ch_key, (ch_title, secs)) in enumerate(chapters_map.items(), 1):
        sections = [
            OutlineSection(
                id=f"s-{tag}-{i}-{j}",
                title_en=st,
                title_zh=st,
            )
            for j, st in enumerate(secs.values(), 1)
        ]
        chapters.append(
            OutlineChapter(
                id=f"c-{tag}-{i}",
                title_en=ch_title,
                title_zh=ch_title,
                sections=sections,
            )
        )
    if not chapters:
        chapters.append(
            OutlineChapter(
                id=f"c-{tag}-1",
                title_en=title_en,
                title_zh=title_zh,
                sections=[
                    OutlineSection(id=f"s-{tag}-1-1", title_en="body", title_zh="body")
                ],
            )
        )
    human = humanize_topic(topic)
    return Outline(
        title=OutlineTitle(en=human, zh=human, yue_hint=human),
        parts=[
            OutlinePart(
                id=f"p-{tag}",
                title_en=title_en,
                title_zh=title_zh,
                chapters=chapters,
            )
        ],
    )


def _fresh_id(base: str, seen: set[str]) -> str:
    candidate = base if base else "id"
    n = 2
    while candidate in seen:
        candidate = f"{base}-{n}" if base else f"id-{n}"
        n += 1
    seen.add(candidate)
    return candidate


def uniquify_parts(outline: Outline, seen: set[str]) -> list[OutlinePart]:
    """Rewrite every part/chapter/section id until it is unique in `seen`."""
    parts: list[OutlinePart] = []
    for part in outline.parts:
        chapters: list[OutlineChapter] = []
        for chapter in part.chapters:
            sections = [
                section.model_copy(update={"id": _fresh_id(section.id, seen)})
                for section in chapter.sections
            ]
            chapters.append(
                chapter.model_copy(
                    update={"id": _fresh_id(chapter.id, seen), "sections": sections}
                )
            )
        parts.append(
            part.model_copy(update={"id": _fresh_id(part.id, seen), "chapters": chapters})
        )
    return parts


def stitch_outlines(topic: str, bucket_outlines: list[tuple[str, Outline]]) -> Outline:
    parts: list[OutlinePart] = []
    seen: set[str] = set()
    for _bucket, outline in bucket_outlines:
        parts.extend(uniquify_parts(outline, seen))
    if not parts:
        parts = fallback_outline_for_bucket(OTHER_BUCKET, [], topic).parts
    human = humanize_topic(topic)
    return Outline(
        title=OutlineTitle(en=human, zh=human, yue_hint=human),
        parts=parts,
    )


def outline_to_markdown(outline: Outline) -> str:
    lines = [f"# {outline.title.en} / {outline.title.zh}", ""]
    if outline.title.yue_hint:
        lines.append(f"yue_hint: {outline.title.yue_hint}")
        lines.append("")
    for part in outline.parts:
        lines.append(f"## {part.title_en} / {part.title_zh} (`{part.id}`)")
        lines.append("")
        for chapter in part.chapters:
            lines.append(f"### {chapter.title_en} / {chapter.title_zh} (`{chapter.id}`)")
            for section in chapter.sections:
                lines.append(
                    f"- {section.title_en} / {section.title_zh} (`{section.id}`)"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def outline_counts(outline: Outline) -> tuple[int, int, int]:
    n_parts = len(outline.parts)
    n_chapters = sum(len(part.chapters) for part in outline.parts)
    n_sections = sum(
        len(chapter.sections) for part in outline.parts for chapter in part.chapters
    )
    return n_parts, n_chapters, n_sections


def write_outline_artifacts(out_dir: Path, outline: Outline) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = outline.model_dump()
    (out_dir / "outline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "outline.md").write_text(outline_to_markdown(outline), encoding="utf-8")


def load_outline(path: Path) -> Outline:
    return Outline.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_headings(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return data


def run_outline(
    *,
    topic: str,
    artifacts_root: Path,
    client: LLMClient,
    model: str | None = None,
    force: bool = False,
    topics_dir: Path | None = None,
) -> Outline:
    artifacts_topic = artifacts_root / topic
    headings_path = artifacts_topic / "inventory" / "headings.json"
    headings = load_headings(headings_path)
    lexicon = load_lexicon(topic, topics_dir=topics_dir)
    fewshot = load_fewshot(topic, topics_dir=topics_dir)
    fewshot_block = format_fewshot_block(fewshot)
    templates = load_outline_templates()
    system = fill_template(
        templates.get("system", ""),
        humanized_topic=humanize_topic(topic),
        topic=topic,
    )
    user_tmpl = templates.get("user", "")
    merge_tmpl = templates.get("merge", "")
    p_hash = sha256_file(OUTLINE_PROMPT_PATH)
    stage_hashes = [sha256_file(headings_path), fewshot_hash(topic, topics_dir)]
    chars_per_token = load_chars_per_token(artifacts_topic)
    max_input_chars = int(HEADING_TOKEN_CAP * max(chars_per_token, DEFAULT_CHARS_PER_TOKEN))

    partitioned = partition_by_bucket(headings, lexicon)
    ordered = bucket_order(lexicon, set(partitioned))
    bucket_outlines: list[tuple[str, Outline]] = []
    fallback_parts = 0

    for bucket in ordered:
        rows = partitioned[bucket]
        packs = pack_heading_rows(rows, chars_per_token)
        pack_outlines: list[Outline] = []
        pack_fallback = False
        for pack_index, pack in enumerate(packs):
            dump = "\n".join(format_heading_line(row) for row in pack)
            user = fill_template(
                user_tmpl,
                humanized_topic=humanize_topic(topic),
                topic=topic,
                bucket=bucket,
                headings=dump,
                fewshot_block=fewshot_block,
            )
            record = client.complete(
                stage="outline",
                prompt_hash=p_hash,
                input_hashes=stage_hashes + [sha256_text(f"{bucket}\n{dump}")],
                user=user,
                system=system,
                temperature=OUTLINE_TEMPERATURE,
                max_tokens=OUTLINE_MAX_TOKENS,
                max_input_chars=max_input_chars,
                json_mode=True,
                model=model,
                force=force,
            )
            parsed = (
                parse_outline(record.response_text)
                if record.finish_reason != "length"
                else None
            )
            if parsed is None:
                pack_fallback = True
                pack_outlines.append(
                    fallback_outline_for_bucket(
                        bucket, pack, topic, lexicon, pack_index=pack_index
                    )
                )
            else:
                pack_outlines.append(parsed)
        if len(pack_outlines) == 1:
            combined = pack_outlines[0]
        else:
            combined = stitch_outlines(topic, [(bucket, ol) for ol in pack_outlines])
        if pack_fallback:
            fallback_parts += 1
        bucket_outlines.append((bucket, combined))

    trees_payload = [
        {"bucket": bucket, "outline": ol.model_dump()} for bucket, ol in bucket_outlines
    ]
    trees_json = json.dumps(trees_payload, ensure_ascii=False, indent=2)
    merge_user = fill_template(
        merge_tmpl,
        humanized_topic=humanize_topic(topic),
        topic=topic,
        bucket_trees=trees_json,
        fewshot_block=fewshot_block,
    )
    merge_record = client.complete(
        stage="outline",
        prompt_hash=p_hash,
        input_hashes=stage_hashes + [sha256_text(trees_json)],
        user=merge_user,
        system=system,
        temperature=OUTLINE_TEMPERATURE,
        max_tokens=OUTLINE_MAX_TOKENS,
        max_input_chars=max_input_chars,
        json_mode=True,
        model=model,
        force=force,
    )
    outline = (
        parse_outline(merge_record.response_text)
        if merge_record.finish_reason != "length"
        else None
    )
    if outline is None:
        outline = stitch_outlines(topic, bucket_outlines)
        fallback_parts += 1

    write_outline_artifacts(artifacts_topic / "outline", outline)
    n_parts, n_chapters, n_sections = outline_counts(outline)
    print(
        f"[outline] parts={n_parts} chapters={n_chapters} sections={n_sections} "
        f"buckets={len(ordered)} fallback_parts={fallback_parts}"
    )
    return outline
