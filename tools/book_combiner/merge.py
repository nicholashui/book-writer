"""Union-of-facts merge for same-heading groups; concatenate complementary subgroups."""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from pydantic import ValidationError

from book_combiner.assign import flatten_outline_nodes
from book_combiner.coverage import missing_tokens, required_tokens
from book_combiner.extract import content_hash
from book_combiner.llm.base import LLMClient
from book_combiner.models import AssignmentItem, CacheRecord, Conflict, Outline, Section
from book_combiner.outline import (
    fill_template,
    load_outline,
    sha256_file,
    sha256_text,
    split_prompt_sections,
)
from book_combiner.overlap import (
    EXACTISH_JACCARD,
    char_shingles,
    estimated_jaccard,
    minhash_signature,
    normalize_text,
)

MERGE_TEMPERATURE = 0.2
_CONFLICTS_LOCK = threading.Lock()
MODEL_OUTPUT_CAP = 8192
DEFAULT_CHARS_PER_TOKEN = 1.0
SIZE_FLOOR_RATIO = 0.7
MAX_TOURNAMENT_DEPTH = 3
TOKEN_STATS_MIN_SAMPLES = 5
MERGEABLE_KINDS = frozenset({"body", "preface", "skill_test"})
CONFLICTS_TERMINATOR = "---CONFLICTS-JSON---"
PARSE_RETRY_SUFFIX = (
    "Return the same markdown, then a line exactly ---CONFLICTS-JSON---, then a JSON array only."
)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MERGE_PROMPT_PATH = _PROMPTS_DIR / "merge.txt"

FORBIDDEN_RE = re.compile(
    r"总之，本[章节]|综上所述|In summary|this chapter covers|the sources generally say",
    re.I,
)


class MergeError(Exception):
    """One or more outline nodes failed coverage, parse, or size checks."""


@dataclass
class UnionMetrics:
    unique_leaf_chars: int
    remaining_overlap: float
    union_chars: float
    size_floor: float


@dataclass
class GroupMergeResult:
    markdown: str
    conflicts: list[Conflict] = field(default_factory=list)
    nonconverged: bool = False
    failed: bool = False
    fail_reason: str = ""
    missing_tokens: list[str] = field(default_factory=list)
    member_ids: list[str] = field(default_factory=list)
    heading_path: list[str] = field(default_factory=list)
    metrics: UnionMetrics | None = None
    out_chars: int = 0
    llm_calls: int = 0


def normalize_heading_path(path: list[str]) -> str:
    return ">".join(normalize_text(p) for p in path)


def canonical_sort_key(section: Section) -> tuple[int, str]:
    return (-section.char_count, section.id)


def unique_members(members: list[Section]) -> list[Section]:
    unique = [s for s in members if s.duplicate_of is None]
    if not unique:
        unique = list(members)
    unique.sort(key=canonical_sort_key)
    return unique


def pairwise_jaccard(left: Section, right: Section) -> float:
    if left.content_hash == right.content_hash:
        return 1.0
    sa, sb = minhash_signature(left.text), minhash_signature(right.text)
    if sa is not None and sb is not None:
        return estimated_jaccard(sa, sb)
    na, nb = normalize_text(left.text), normalize_text(right.text)
    if na and na == nb:
        return 1.0
    sha, shb = char_shingles(na), char_shingles(nb)
    if not sha or not shb:
        return 0.0
    return len(sha & shb) / len(sha | shb)


def remaining_overlap(members: list[Section]) -> float:
    if len(members) < 2:
        return 0.0
    scores = [pairwise_jaccard(a, b) for a, b in combinations(members, 2)]
    return sum(scores) / len(scores)


def union_metrics(members: list[Section]) -> UnionMetrics:
    unique = unique_members(members)
    unique_leaf_chars = sum(s.char_count for s in unique)
    overlap = remaining_overlap(unique)
    union_chars = unique_leaf_chars * (1.0 - overlap)
    return UnionMetrics(
        unique_leaf_chars=unique_leaf_chars,
        remaining_overlap=overlap,
        union_chars=union_chars,
        size_floor=SIZE_FLOOR_RATIO * union_chars,
    )


def expected_out_chars(*, union_chars: float, graft_payload_chars: int, graft_applied: bool) -> int:
    if graft_applied:
        return int(graft_payload_chars)
    return int(union_chars)


def needed_tokens(expected: float, chars_per_token: float) -> int:
    cpt = chars_per_token if chars_per_token > 0 else DEFAULT_CHARS_PER_TOKEN
    return math.ceil(expected / cpt) + 256


def max_tokens_for(expected: float, chars_per_token: float, n_samples: int) -> int:
    needed = needed_tokens(expected, chars_per_token)
    if needed > MODEL_OUTPUT_CAP:
        return MODEL_OUTPUT_CAP
    if n_samples < TOKEN_STATS_MIN_SAMPLES:
        return MODEL_OUTPUT_CAP
    return min(MODEL_OUTPUT_CAP, needed)


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip("\n"))
    return [p.strip("\n") for p in parts if p.strip()]


def paragraph_jaccard(left: str, right: str) -> float:
    sa, sb = minhash_signature(left), minhash_signature(right)
    if sa is not None and sb is not None:
        return estimated_jaccard(sa, sb)
    na, nb = normalize_text(left), normalize_text(right)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    sha, shb = char_shingles(na), char_shingles(nb)
    if not sha or not shb:
        return 1.0 if na == nb else 0.0
    return len(sha & shb) / len(sha | shb)


def unique_paragraphs(text: str, canonical_paras: list[str]) -> list[str]:
    out: list[str] = []
    for para in split_paragraphs(text):
        if all(paragraph_jaccard(para, canon) < EXACTISH_JACCARD for canon in canonical_paras):
            out.append(para)
    return out


def format_source_block(section: Section) -> str:
    return f"---SOURCE {section.source_stem}---\n{section.text}\n---END SOURCE---"


def pack_graft_payload(unique: list[Section], max_input_chars: int) -> tuple[str, bool, int]:
    """Canonical + UNIQUE FROM blocks when raw concat exceeds the input cap."""
    if not unique:
        return "", False, 0
    raw = "\n\n".join(format_source_block(s) for s in unique)
    if len(raw) <= max_input_chars:
        return raw, False, len(raw)
    canonical, rest = unique[0], unique[1:]
    canon_paras = split_paragraphs(canonical.text) or [canonical.text]
    blocks = [format_source_block(canonical)]
    for src in rest:
        extras = unique_paragraphs(src.text, canon_paras)
        if extras:
            blocks.append(f"UNIQUE FROM {src.source_stem}:\n" + "\n\n".join(extras))
    grafted = "\n\n".join(blocks)
    return grafted, True, len(grafted)


def payload_over_budget(
    payload_chars: int,
    expected: float,
    max_input_chars: int,
    chars_per_token: float,
) -> bool:
    if payload_chars > max_input_chars:
        return True
    return needed_tokens(expected, chars_per_token) > MODEL_OUTPUT_CAP


def load_merge_templates(path: Path | None = None) -> dict[str, str]:
    return split_prompt_sections((path or MERGE_PROMPT_PATH).read_text(encoding="utf-8"))


def merge_input_hashes(
    members: list[Section],
    node_id: str,
    title_en: str,
    title_zh: str,
    *,
    payload: str = "",
    heading_key: str = "",
    round_id: str = "",
) -> list[str]:
    """Per-call cache key pieces: this group's unique members, packed payload, round."""
    hashes = sorted(s.content_hash for s in members)
    hashes.append(node_id)
    hashes.append(sha256_text(title_en + title_zh))
    hashes.append(sha256_text(heading_key))
    hashes.append(sha256_text(payload))
    hashes.append(sha256_text(round_id))
    return hashes


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def parse_merge_output(text: str) -> tuple[str, list[Conflict]] | None:
    lines = text.splitlines()
    idx = next((i for i, line in enumerate(lines) if line.strip() == CONFLICTS_TERMINATOR), None)
    if idx is None:
        return None
    markdown = "\n".join(lines[:idx]).strip()
    blob = "\n".join(lines[idx + 1 :]).strip()
    if blob.startswith("```"):
        return None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    try:
        conflicts = [Conflict.model_validate(item) for item in data]
    except (ValidationError, TypeError, ValueError):
        return None
    return markdown, conflicts


def load_token_stats(artifacts_topic: Path) -> tuple[float, int]:
    path = artifacts_topic / "logs" / "token_stats.json"
    if not path.is_file():
        return DEFAULT_CHARS_PER_TOKEN, 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_CHARS_PER_TOKEN, 0
    if not isinstance(data, dict):
        return DEFAULT_CHARS_PER_TOKEN, 0
    samples = data.get("merge_samples")
    if not isinstance(samples, list):
        return DEFAULT_CHARS_PER_TOKEN, 0
    ratios: list[float] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        try:
            chars_out = float(sample["chars_out"])
            output_tokens = float(sample["output_tokens"])
        except (KeyError, TypeError, ValueError):
            continue
        if output_tokens > 0 and chars_out > 0:
            ratios.append(chars_out / output_tokens)
    if len(ratios) < TOKEN_STATS_MIN_SAMPLES:
        return DEFAULT_CHARS_PER_TOKEN, len(ratios)
    return sum(ratios) / len(ratios), len(ratios)


def record_merge_sample(artifacts_topic: Path, chars_out: int, output_tokens: int) -> None:
    if output_tokens <= 0 or chars_out <= 0:
        return
    path = artifacts_topic / "logs" / "token_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}
    samples = data.get("merge_samples")
    if not isinstance(samples, list):
        samples = []
    samples.append({"chars_out": chars_out, "output_tokens": output_tokens})
    data["merge_samples"] = samples
    ratios = [
        s["chars_out"] / s["output_tokens"]
        for s in samples
        if isinstance(s, dict)
        and s.get("output_tokens")
        and s.get("chars_out")
    ]
    if len(ratios) >= TOKEN_STATS_MIN_SAMPLES:
        data["chars_per_token"] = sum(ratios) / len(ratios)
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def load_sections(artifacts_topic: Path) -> list[Section]:
    extract_dir = artifacts_topic / "extract"
    sections: list[Section] = []
    if not extract_dir.is_dir():
        return sections
    for path in sorted(extract_dir.glob("*.sections.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        sections.extend(Section(**row) for row in raw)
    return sections


def load_assignments(path: Path) -> list[AssignmentItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("assignments", data) if isinstance(data, dict) else data
    return [AssignmentItem.model_validate(row) for row in items]


def section_from_markdown(text: str, *, node_id: str, heading_path: list[str]) -> Section:
    return Section(
        id=f"{node_id}:merged:{sha256_text(text)[:12]}",
        source_stem="merged",
        source_book="merged",
        heading_path=list(heading_path),
        level=3,
        start_line=0,
        end_line=0,
        text=text,
        kind="body",
        char_count=len(text),
        content_hash=content_hash(text),
        duplicate_of=None,
    )


def concat_fragments(unique: list[Section], *, heading_path: list[str]) -> GroupMergeResult:
    chunks: list[str] = []
    for section in unique:
        title = section.heading_path[-1] if section.heading_path else section.source_stem
        label = f"{title}（{section.source_stem}）" if section.source_stem else title
        chunks.append(f"#### {label}\n\n{section.text.strip()}")
    markdown = "\n\n".join(chunks).strip()
    metrics = union_metrics(unique)
    print("[merge] merge_nonconverged=true")
    return GroupMergeResult(
        markdown=markdown,
        nonconverged=True,
        member_ids=[s.id for s in unique],
        heading_path=list(heading_path),
        metrics=metrics,
        out_chars=len(markdown),
    )


def _failed_result(
    members: list[Section],
    heading_path: list[str],
    reason: str,
    missing: list[str] | None = None,
    llm_calls: int = 0,
) -> GroupMergeResult:
    return GroupMergeResult(
        markdown="",
        failed=True,
        fail_reason=reason,
        missing_tokens=list(missing or []),
        member_ids=[s.id for s in members],
        heading_path=list(heading_path),
        metrics=union_metrics(members),
        llm_calls=llm_calls,
    )


def _complete_merge(
    *,
    client: LLMClient,
    system: str,
    user: str,
    prompt_hash: str,
    input_hashes: list[str],
    max_tokens: int,
    max_input_chars: int,
    model: str | None,
    force: bool,
) -> tuple[CacheRecord, int]:
    record = client.complete(
        stage="merge",
        prompt_hash=prompt_hash,
        input_hashes=input_hashes,
        user=user,
        system=system,
        temperature=MERGE_TEMPERATURE,
        max_tokens=max_tokens,
        max_input_chars=max_input_chars,
        json_mode=False,
        model=model,
        force=force,
    )
    return record, 1


def merge_heading_group(
    members: list[Section],
    *,
    node_id: str,
    title_en: str,
    title_zh: str,
    client: LLMClient,
    templates: dict[str, str],
    prompt_hash: str,
    max_input_chars: int,
    chars_per_token: float,
    n_samples: int,
    artifacts_topic: Path,
    model: str | None,
    force: bool,
    depth: int = 0,
) -> GroupMergeResult:
    heading_path = list(members[0].heading_path) if members else []
    unique = unique_members(members)
    metrics = union_metrics(unique)
    required = required_tokens(unique)
    member_ids = [s.id for s in unique]
    heading_key = normalize_heading_path(heading_path)
    round_id = f"depth:{depth}"

    if not unique:
        return _failed_result(members, heading_path, "no unique members")

    if len(unique) == 1:
        text = unique[0].text.strip()
        return GroupMergeResult(
            markdown=text,
            member_ids=member_ids,
            heading_path=heading_path,
            metrics=metrics,
            out_chars=len(text),
            llm_calls=0,
        )

    payload, graft_applied, payload_chars = pack_graft_payload(unique, max_input_chars)
    input_hashes = merge_input_hashes(
        unique,
        node_id,
        title_en,
        title_zh,
        payload=payload,
        heading_key=heading_key,
        round_id=round_id,
    )
    expected = expected_out_chars(
        union_chars=metrics.union_chars,
        graft_payload_chars=payload_chars,
        graft_applied=graft_applied,
    )
    over = payload_over_budget(payload_chars, expected, max_input_chars, chars_per_token)

    if over:
        if len(unique) >= 3 and depth < MAX_TOURNAMENT_DEPTH:
            left = merge_heading_group(
                unique[:2],
                node_id=node_id,
                title_en=title_en,
                title_zh=title_zh,
                client=client,
                templates=templates,
                prompt_hash=prompt_hash,
                max_input_chars=max_input_chars,
                chars_per_token=chars_per_token,
                n_samples=n_samples,
                artifacts_topic=artifacts_topic,
                model=model,
                force=force,
                depth=depth + 1,
            )
            if left.failed:
                return left
            synth = section_from_markdown(
                left.markdown, node_id=node_id, heading_path=heading_path
            )
            folded = merge_heading_group(
                [synth, *unique[2:]],
                node_id=node_id,
                title_en=title_en,
                title_zh=title_zh,
                client=client,
                templates=templates,
                prompt_hash=prompt_hash,
                max_input_chars=max_input_chars,
                chars_per_token=chars_per_token,
                n_samples=n_samples,
                artifacts_topic=artifacts_topic,
                model=model,
                force=force,
                depth=depth + 1,
            )
            folded.conflicts = left.conflicts + folded.conflicts
            folded.nonconverged = left.nonconverged or folded.nonconverged
            folded.llm_calls = left.llm_calls + folded.llm_calls
            folded.member_ids = member_ids
            folded.metrics = metrics
            return folded
        return concat_fragments(unique, heading_path=heading_path)

    max_tokens = max_tokens_for(expected, chars_per_token, n_samples)
    user = fill_template(
        templates.get("user", ""),
        node_id=node_id,
        title_en=title_en,
        title_zh=title_zh,
        payload=payload,
    )
    system = templates.get("system", "")

    record, calls = _complete_merge(
        client=client,
        system=system,
        user=user,
        prompt_hash=prompt_hash,
        input_hashes=input_hashes,
        max_tokens=max_tokens,
        max_input_chars=max_input_chars,
        model=model,
        force=force,
    )
    if record.finish_reason == "length":
        if len(unique) >= 3 and depth < MAX_TOURNAMENT_DEPTH:
            return merge_heading_group(
                unique,
                node_id=node_id,
                title_en=title_en,
                title_zh=title_zh,
                client=client,
                templates=templates,
                prompt_hash=prompt_hash,
                max_input_chars=max(1, max_input_chars // 2),
                chars_per_token=chars_per_token,
                n_samples=n_samples,
                artifacts_topic=artifacts_topic,
                model=model,
                force=force,
                depth=depth + 1,
            )
        result = concat_fragments(unique, heading_path=heading_path)
        result.llm_calls = calls
        return result

    parsed = parse_merge_output(record.response_text)
    if parsed is None:
        retry_user = user + "\n\n" + PARSE_RETRY_SUFFIX
        record, extra = _complete_merge(
            client=client,
            system=system,
            user=retry_user,
            prompt_hash=prompt_hash,
            input_hashes=input_hashes + [sha256_text("conflicts-retry")],
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            model=model,
            force=True,
        )
        calls += extra
        if record.finish_reason == "length":
            result = concat_fragments(unique, heading_path=heading_path)
            result.llm_calls = calls
            return result
        parsed = parse_merge_output(record.response_text)
        if parsed is None:
            result = concat_fragments(unique, heading_path=heading_path)
            result.llm_calls = calls
            return result

    markdown, conflicts = parsed
    missing = missing_tokens(markdown, required)
    if missing:
        suffix = "MISSING: " + "、".join(missing)
        record, extra = _complete_merge(
            client=client,
            system=system,
            user=user + "\n\n" + suffix,
            prompt_hash=prompt_hash,
            input_hashes=input_hashes + [sha256_text(suffix)],
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            model=model,
            force=True,
        )
        calls += extra
        if record.finish_reason == "length":
            result = concat_fragments(unique, heading_path=heading_path)
            result.llm_calls = calls
            return result
        parsed = parse_merge_output(record.response_text)
        if parsed is None:
            result = concat_fragments(unique, heading_path=heading_path)
            result.llm_calls = calls
            return result
        markdown, conflicts = parsed
        missing = missing_tokens(markdown, required)

    record_merge_sample(artifacts_topic, len(markdown), record.output_tokens)

    if missing:
        result = concat_fragments(unique, heading_path=heading_path)
        result.llm_calls = calls
        return result
    if len(markdown) < metrics.size_floor:
        # Keep every unique leaf rather than fail the node on a short LLM merge.
        result = concat_fragments(unique, heading_path=heading_path)
        result.llm_calls = calls
        return result
    if FORBIDDEN_RE.search(markdown):
        return _failed_result(
            unique, heading_path, "forbidden summary phrasing", llm_calls=calls
        )

    return GroupMergeResult(
        markdown=markdown.strip(),
        conflicts=conflicts,
        member_ids=member_ids,
        heading_path=heading_path,
        metrics=metrics,
        out_chars=len(markdown),
        llm_calls=calls,
    )


def partition_by_heading(members: list[Section]) -> list[tuple[str, list[Section]]]:
    groups: dict[str, list[Section]] = {}
    for section in sorted(members, key=canonical_sort_key):
        key = normalize_heading_path(section.heading_path)
        groups.setdefault(key, []).append(section)
    return list(groups.items())


def wrap_node_markdown(
    title_zh: str,
    groups: list[tuple[list[str], GroupMergeResult]],
) -> str:
    lines = [f"### {title_zh}", ""]
    if len(groups) == 1:
        result = groups[0][1]
        body = result.markdown.strip()
        if body:
            lines.append(body)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
    for path, result in groups:
        if result.nonconverged:
            lines.append(result.markdown.strip())
            lines.append("")
            continue
        sub = path[-1] if path else "body"
        lines.append(f"#### {sub}")
        lines.append("")
        if result.markdown.strip():
            lines.append(result.markdown.strip())
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_failed_node(
    nodes_dir: Path,
    node_id: str,
    members: list[Section],
    reason: str,
    missing: list[str],
) -> None:
    zh_path = nodes_dir / f"{node_id}.zh.md"
    if zh_path.is_file():
        zh_path.unlink()
    lines = [
        f"# FAILED {node_id}",
        "",
        f"reason: {reason}",
        "members: " + ", ".join(s.id for s in members),
        "missing_tokens: " + "、".join(missing),
        "",
    ]
    atomic_write_text(nodes_dir / f"{node_id}.FAILED.md", "\n".join(lines))


def reset_conflicts_jsonl(path: Path) -> None:
    """Start each merge run with a fresh conflicts log so reruns do not duplicate."""
    atomic_write_text(path, "")


def write_conflicts_jsonl(path: Path, conflicts: list[Conflict]) -> None:
    if not conflicts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = "".join(
        json.dumps(c.model_dump(), ensure_ascii=False) + "\n" for c in conflicts
    )
    with _CONFLICTS_LOCK:
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        atomic_write_text(path, existing + extra)


def merge_outline_node(
    *,
    node_id: str,
    title_en: str,
    title_zh: str,
    members: list[Section],
    client: LLMClient,
    artifacts_topic: Path,
    max_input_chars: int,
    model: str | None,
    force: bool,
) -> GroupMergeResult:
    nodes_dir = artifacts_topic / "merge" / "nodes"
    templates = load_merge_templates()
    prompt_hash = sha256_file(MERGE_PROMPT_PATH)
    chars_per_token, n_samples = load_token_stats(artifacts_topic)
    groups = partition_by_heading(members)
    merged: list[tuple[list[str], GroupMergeResult]] = []
    all_conflicts: list[Conflict] = []
    nonconverged = False
    llm_calls = 0
    missing: list[str] = []
    fail_reason = ""

    for _key, group in groups:
        result = merge_heading_group(
            group,
            node_id=node_id,
            title_en=title_en,
            title_zh=title_zh,
            client=client,
            templates=templates,
            prompt_hash=prompt_hash,
            max_input_chars=max_input_chars,
            chars_per_token=chars_per_token,
            n_samples=n_samples,
            artifacts_topic=artifacts_topic,
            model=model,
            force=force,
        )
        llm_calls += result.llm_calls
        if result.failed:
            fail_reason = result.fail_reason
            missing = result.missing_tokens
            write_failed_node(nodes_dir, node_id, members, fail_reason, missing)
            metrics = union_metrics(unique_members(members))
            print(
                f"[merge] {node_id} FAILED reason={fail_reason} "
                f"missing={len(missing)}"
            )
            return GroupMergeResult(
                markdown="",
                failed=True,
                fail_reason=fail_reason,
                missing_tokens=missing,
                member_ids=[s.id for s in members],
                metrics=metrics,
                llm_calls=llm_calls,
            )
        nonconverged = nonconverged or result.nonconverged
        all_conflicts.extend(result.conflicts)
        merged.append((list(group[0].heading_path), result))
        if result.metrics and not result.nonconverged:
            print(
                f"[merge] {node_id} {group[0].heading_path[-1] if group[0].heading_path else 'body'} "
                f"union {len(unique_members(group))} sources "
                f"{result.metrics.unique_leaf_chars}→{result.out_chars} chars "
                f"conflicts={len(result.conflicts)} coverage=ok"
            )

    if len(merged) > 1:
        labels = [path[-1] if path else "body" for path, _ in merged]
        print(f"[merge] {node_id} complementary concat ####={len(merged)} ({', '.join(labels)})")

    markdown = wrap_node_markdown(title_zh, merged)
    failed_path = nodes_dir / f"{node_id}.FAILED.md"
    if failed_path.is_file():
        failed_path.unlink()
    atomic_write_text(nodes_dir / f"{node_id}.zh.md", markdown)
    metrics = union_metrics(unique_members(members))
    meta = {
        "node_id": node_id,
        "coverage": "ok",
        "fragments": [s.id for s in members],
        "conflict_ids": [c.id for c in all_conflicts],
        "merge_nonconverged": nonconverged,
        "size_floor": metrics.size_floor,
        "union_chars": metrics.union_chars,
        "unique_leaf_chars": metrics.unique_leaf_chars,
        "remaining_overlap": metrics.remaining_overlap,
        "llm_calls": llm_calls,
    }
    atomic_write_text(
        nodes_dir / f"{node_id}.meta.json",
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
    )
    write_conflicts_jsonl(artifacts_topic / "merge" / "conflicts.jsonl", all_conflicts)
    return GroupMergeResult(
        markdown=markdown,
        conflicts=all_conflicts,
        nonconverged=nonconverged,
        member_ids=[s.id for s in members],
        metrics=metrics,
        out_chars=len(markdown),
        llm_calls=llm_calls,
    )


def gather_members(
    sections: list[Section],
    assignments: list[AssignmentItem],
) -> dict[str, list[Section]]:
    by_id = {s.id: s for s in sections}
    grouped: dict[str, list[Section]] = {}
    for item in assignments:
        section = by_id.get(item.section_id)
        if section is None or section.kind not in MERGEABLE_KINDS:
            continue
        grouped.setdefault(item.primary_node_id, []).append(section)
    return grouped


def node_titles(outline: Outline) -> dict[str, tuple[str, str]]:
    return {node_id: (en, zh) for node_id, en, zh in flatten_outline_nodes(outline)}


def run_merge(
    *,
    topic: str,
    artifacts_root: Path,
    client: LLMClient,
    model: str | None = None,
    force: bool = False,
    max_input_chars: int = 6000,
    concurrency: int = 1,
) -> list[GroupMergeResult]:
    artifacts_topic = artifacts_root / topic
    outline = load_outline(artifacts_topic / "outline" / "outline.json")
    assignments = load_assignments(artifacts_topic / "outline" / "assignment.json")
    sections = load_sections(artifacts_topic)
    titles = node_titles(outline)
    by_node = gather_members(sections, assignments)
    reset_conflicts_jsonl(artifacts_topic / "merge" / "conflicts.jsonl")
    results: list[GroupMergeResult] = []
    failures = 0
    node_ids = sorted(by_node)

    def _merge_one(node_id: str) -> GroupMergeResult:
        members = by_node[node_id]
        title_en, title_zh = titles.get(node_id, (node_id, node_id))
        return merge_outline_node(
            node_id=node_id,
            title_en=title_en,
            title_zh=title_zh,
            members=members,
            client=client,
            artifacts_topic=artifacts_topic,
            max_input_chars=max_input_chars,
            model=model,
            force=force,
        )

    workers = max(1, int(concurrency))
    if workers == 1:
        merged_nodes = [_merge_one(nid) for nid in node_ids]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            merged_nodes = list(pool.map(_merge_one, node_ids))
    for result in merged_nodes:
        results.append(result)
        if result.failed:
            failures += 1
    if failures:
        raise MergeError(f"{failures} merge node(s) failed")
    return results
