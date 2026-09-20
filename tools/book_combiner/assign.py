"""Assign heading rows to outline nodes. Keyword pre-seed skips the LLM when unique."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from book_combiner.lexicon import keywords_for, load_lexicon
from book_combiner.llm.base import LLMClient
from book_combiner.models import AssignmentItem, AssignmentPayload, Outline
from book_combiner.outline import (
    fill_template,
    load_headings,
    load_outline,
    sha256_file,
    sha256_text,
    split_prompt_sections,
)

ASSIGN_TEMPERATURE = 0.2
ASSIGN_MAX_TOKENS = 2048
ASSIGN_BATCH_SIZE = 80  # upper bound; packing also caps by estimated output tokens
ASSIGN_OUTPUT_TOKENS_PER_ROW = 50
ASSIGN_MAX_INPUT_CHARS = 12_000

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
ASSIGN_PROMPT_PATH = _PROMPTS_DIR / "assign.txt"

OutlineNode = tuple[str, str, str]  # id, title_en, title_zh


def lexicon_hash(lexicon: dict[str, list[str]]) -> str:
    payload = json.dumps(lexicon, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(payload)


def assign_stage_input_hashes(
    outline_path: Path,
    headings_path: Path,
    lexicon: dict[str, list[str]],
) -> list[str]:
    """Stage-level hashes. Editing outline.json changes the first file hash."""
    return [
        sha256_file(outline_path),
        sha256_file(headings_path),
        lexicon_hash(lexicon),
    ]


def flatten_outline_nodes(outline: Outline) -> list[OutlineNode]:
    nodes: list[OutlineNode] = []
    for part in outline.parts:
        nodes.append((part.id, part.title_en, part.title_zh))
        for chapter in part.chapters:
            nodes.append((chapter.id, chapter.title_en, chapter.title_zh))
            for section in chapter.sections:
                nodes.append((section.id, section.title_en, section.title_zh))
    return nodes


def first_section_id(outline: Outline) -> str:
    for part in outline.parts:
        for chapter in part.chapters:
            for section in chapter.sections:
                return section.id
    for part in outline.parts:
        for chapter in part.chapters:
            return chapter.id
    if outline.parts:
        return outline.parts[0].id
    return "s-unknown"


def assign_batch_limit() -> int:
    by_tokens = max(1, ASSIGN_MAX_TOKENS // ASSIGN_OUTPUT_TOKENS_PER_ROW)
    return min(ASSIGN_BATCH_SIZE, by_tokens)


def node_matches_bucket(
    node: OutlineNode,
    bucket: str,
    lexicon: dict[str, list[str]],
) -> bool:
    node_id, title_en, title_zh = node
    bucket_l = bucket.lower()
    bucket_spaced = bucket_l.replace("_", " ")
    id_tokens = [tok for tok in re.split(r"[-_]", node_id.lower()) if tok]
    if bucket_l in id_tokens or bucket_l.replace("_", "") in id_tokens:
        return True
    if re.search(rf"\b{re.escape(bucket_spaced)}\b", title_en.lower()):
        return True
    for pat in lexicon.get(bucket, []):
        if not pat:
            continue
        if pat in title_zh or pat.lower() in title_en.lower():
            return True
    return False


def unique_node_for_bucket(
    bucket: str,
    nodes: list[OutlineNode],
    lexicon: dict[str, list[str]],
) -> str | None:
    hits = [node[0] for node in nodes if node_matches_bucket(node, bucket, lexicon)]
    unique = list(dict.fromkeys(hits))
    if len(unique) == 1:
        return unique[0]
    return None


def preseed_assignment(
    row: dict,
    nodes: list[OutlineNode],
    lexicon: dict[str, list[str]],
) -> AssignmentItem | None:
    """Skip the LLM when a lexicon bucket uniquely matches one outline node title."""
    path = list(row.get("heading_path") or [])
    keys = keywords_for(path, lexicon)
    if not keys:
        return None
    matched: list[str] = []
    for bucket in keys:
        node_id = unique_node_for_bucket(bucket, nodes, lexicon)
        if node_id is not None:
            matched.append(node_id)
    unique = list(dict.fromkeys(matched))
    if len(unique) != 1:
        return None
    return AssignmentItem(
        section_id=str(row["section_id"]),
        primary_node_id=unique[0],
        secondary_node_id=None,
    )


def parse_assignments(text: str) -> list[AssignmentItem] | None:
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    try:
        if isinstance(data, list):
            items = [AssignmentItem.model_validate(item) for item in data]
        elif isinstance(data, dict):
            items = AssignmentPayload.model_validate(data).assignments
        else:
            return None
    except (ValidationError, TypeError, ValueError):
        return None
    return items or None


def load_assign_templates(path: Path | None = None) -> dict[str, str]:
    text = (path or ASSIGN_PROMPT_PATH).read_text(encoding="utf-8")
    return split_prompt_sections(text)


def write_assignment(path: Path, items: list[AssignmentItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"assignments": [item.model_dump() for item in items]}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_assign(
    *,
    topic: str,
    artifacts_root: Path,
    client: LLMClient,
    model: str | None = None,
    force: bool = False,
    topics_dir: Path | None = None,
) -> list[AssignmentItem]:
    artifacts_topic = artifacts_root / topic
    outline_path = artifacts_topic / "outline" / "outline.json"
    headings_path = artifacts_topic / "inventory" / "headings.json"
    outline = load_outline(outline_path)
    headings = load_headings(headings_path)
    lexicon = load_lexicon(topic, topics_dir=topics_dir)
    nodes = flatten_outline_nodes(outline)
    known_ids = {node[0] for node in nodes}
    fallback_id = first_section_id(outline)
    stage_hashes = assign_stage_input_hashes(outline_path, headings_path, lexicon)

    assigned: dict[str, AssignmentItem] = {}
    pending: list[dict] = []
    for row in headings:
        section_id = str(row.get("section_id") or "")
        seeded = preseed_assignment(row, nodes, lexicon)
        if seeded is not None:
            assigned[section_id] = seeded
        else:
            pending.append(row)

    templates = load_assign_templates()
    system = templates.get("system", "")
    user_tmpl = templates.get("user", "")
    p_hash = sha256_file(ASSIGN_PROMPT_PATH)
    nodes_payload = json.dumps(
        [{"id": n[0], "title_en": n[1], "title_zh": n[2]} for n in nodes],
        ensure_ascii=False,
    )

    def fallback_item(row: dict) -> AssignmentItem:
        return AssignmentItem(
            section_id=str(row["section_id"]),
            primary_node_id=fallback_id,
            secondary_node_id=None,
        )

    def from_llm(row: dict, item: AssignmentItem | None) -> AssignmentItem:
        section_id = str(row["section_id"])
        if item is None or item.primary_node_id not in known_ids:
            return fallback_item(row)
        secondary = item.secondary_node_id
        if secondary is not None and secondary not in known_ids:
            secondary = None
        return AssignmentItem(
            section_id=section_id,
            primary_node_id=item.primary_node_id,
            secondary_node_id=secondary,
        )

    def complete_batch(batch: list[dict]) -> list[AssignmentItem] | None:
        records_json = json.dumps(batch, ensure_ascii=False)
        user = fill_template(
            user_tmpl,
            outline_nodes=nodes_payload,
            records=records_json,
        )
        record = client.complete(
            stage="assign",
            prompt_hash=p_hash,
            input_hashes=stage_hashes + [sha256_text(records_json)],
            user=user,
            system=system,
            temperature=ASSIGN_TEMPERATURE,
            max_tokens=ASSIGN_MAX_TOKENS,
            max_input_chars=ASSIGN_MAX_INPUT_CHARS,
            json_mode=True,
            model=model,
            force=force,
        )
        if record.finish_reason == "length":
            return None
        return parse_assignments(record.response_text)

    def apply_parsed(batch: list[dict], parsed: list[AssignmentItem]) -> None:
        by_id = {item.section_id: item for item in parsed}
        for row in batch:
            assigned[str(row["section_id"])] = from_llm(row, by_id.get(str(row["section_id"])))

    def apply_fallback(batch: list[dict]) -> None:
        for row in batch:
            assigned[str(row["section_id"])] = fallback_item(row)

    def run_batch(batch: list[dict], *, retried: bool = False) -> None:
        parsed = complete_batch(batch)
        if parsed is not None:
            apply_parsed(batch, parsed)
            return
        if retried or len(batch) <= 1:
            apply_fallback(batch)
            return
        mid = max(1, len(batch) // 2)
        run_batch(batch[:mid], retried=True)
        if batch[mid:]:
            run_batch(batch[mid:], retried=True)

    limit = assign_batch_limit()
    for start in range(0, len(pending), limit):
        run_batch(pending[start : start + limit])

    items = [
        assigned[str(row["section_id"])]
        for row in headings
        if str(row.get("section_id") or "") in assigned
    ]
    write_assignment(artifacts_topic / "outline" / "assignment.json", items)
    n_secondary = sum(1 for item in items if item.secondary_node_id)
    print(f"[assign] primary={len(items)} secondary={n_secondary}")
    return items
