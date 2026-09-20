"""Assemble ZH / EN / HK books from merge and translate node artifacts."""

from __future__ import annotations

import json
import os
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from book_combiner import __version__ as COMBINER_VERSION
from book_combiner.assign import flatten_outline_nodes
from book_combiner.merge import atomic_write_text, load_assignments, load_sections
from book_combiner.models import AssignmentItem, Conflict, Outline
from book_combiner.outline import humanize_topic, load_outline
from book_combiner.translate import (
    CJK_RE,
    HAN_DENSITY_CAP,
    BOOK_TITLE_RE,
    HEADING_RE,
    en_han_density,
    evaluate_cantonese,
    first_heading_text,
    heading_lines,
    traditionalize,
)

TOC_TITLES = {
    "zh": "目录",
    "en": "Contents",
    "yue": "目錄",
}
APPENDIX_A = {
    "zh": "附录 A · 来源",
    "en": "Appendix A — Sources",
    "yue": "附錄 A · 來源",
}
APPENDIX_B = {
    "zh": "附录 B · 心理测试",
    "en": "Appendix B — Quizzes",
    "yue": "附錄 B · 心理測試",
}
APPENDIX_C = {
    "zh": "附录 C · 题外材料",
    "en": "Appendix C — Off-topic material",
    "yue": "附錄 C · 題外材料",
}
APPENDIX_D = {
    "zh": "附录 D · 参考书目",
    "en": "Appendix D — Bibliography",
    "yue": "附錄 D · 參考書目",
}
APPENDIX_E = {
    "zh": "附录 E · 冲突",
    "en": "Appendix E — Conflicts",
    "yue": "附錄 E · 衝突",
}

BLOB_B = "appendix-b-quizzes"
BLOB_C = "appendix-c-offtopic"
BLOB_D = "appendix-d-bibliography"
BLOB_E = "appendix-e-conflicts"
BLOB_A = "appendix-a-sources"

SUMMARY_CANARY_RATIO = 0.5
SUMMARY_CANARY_OVERLAP = 0.3


class AssembleError(Exception):
    """Validation failed; compiled outputs must not be replaced."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_manifest(artifacts_topic: Path) -> dict:
    path = artifacts_topic / "manifest.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def outline_leaves(outline: Outline) -> list[tuple[str, str, str]]:
    leaves: list[tuple[str, str, str]] = []
    for part in outline.parts:
        if not part.chapters:
            leaves.append((part.id, part.title_en, part.title_zh))
            continue
        for chapter in part.chapters:
            if not chapter.sections:
                leaves.append((chapter.id, chapter.title_en, chapter.title_zh))
            else:
                for section in chapter.sections:
                    leaves.append((section.id, section.title_en, section.title_zh))
    return leaves


def node_titles(outline: Outline) -> dict[str, tuple[str, str]]:
    return {node_id: (en, zh) for node_id, en, zh in flatten_outline_nodes(outline)}


def read_text_if_exists(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def failed_node_ids(artifacts_topic: Path) -> list[str]:
    nodes_dir = artifacts_topic / "merge" / "nodes"
    if not nodes_dir.is_dir():
        return []
    return sorted(path.name[: -len(".FAILED.md")] for path in nodes_dir.glob("*.FAILED.md"))


def appendix_zh_path(artifacts_topic: Path, blob_id: str) -> Path:
    return artifacts_topic / "merge" / "appendices" / f"{blob_id}.zh.md"


def translate_node_path(artifacts_topic: Path, lang: str, node_id: str) -> Path:
    return artifacts_topic / "translate" / lang / f"{node_id}.md"


def translate_appendix_path(artifacts_topic: Path, lang: str, blob_id: str) -> Path:
    return artifacts_topic / "translate" / lang / "appendices" / f"{blob_id}.md"


def format_appendix_a_table(manifest: dict) -> str:
    sources = list(manifest.get("sources") or [])
    sources = sorted(sources, key=lambda row: str(row.get("stem") or ""))
    lines = [
        "| stem | bytes | sha256 |",
        "|---|---:|---|",
    ]
    for row in sources:
        stem = str(row.get("stem") or "")
        raw_bytes = row.get("bytes", "")
        digest = str(row.get("sha256") or "")
        lines.append(f"| {stem} | {raw_bytes} | {digest} |")
    return "\n".join(lines) + "\n"


def format_appendix_a(lang: str, manifest: dict) -> str:
    title = APPENDIX_A[lang]
    return f"## {title}\n\n{format_appendix_a_table(manifest)}"


def concat_quiz_markdown(sections) -> str:
    blocks: list[str] = []
    quizzes = [s for s in sections if s.kind == "quiz"]
    quizzes.sort(key=lambda s: (s.source_stem, s.start_line, s.id))
    for section in quizzes:
        heading = section.heading_path[-1] if section.heading_path else section.id
        body = section.text.strip()
        blocks.append(f"### {heading}\n\n{body}\n")
    return "\n".join(blocks).strip() + ("\n" if blocks else "")


def harvest_bibliography(sections, extract_dir: Path) -> tuple[str, list[str]]:
    blocks: list[str] = []
    scraped: list[str] = []
    seen: set[str] = set()
    biblio = [s for s in sections if s.kind == "bibliography"]
    biblio.sort(key=lambda s: (s.source_stem, s.start_line, s.id))
    for section in biblio:
        heading = section.heading_path[-1] if section.heading_path else "参考书目"
        body = section.text.strip()
        blocks.append(f"### {heading}\n\n{body}\n")
        for title in BOOK_TITLE_RE.findall(section.text):
            if title not in seen:
                seen.add(title)
                scraped.append(title)
    if extract_dir.is_dir():
        for path in sorted(extract_dir.glob("*.dropped.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "")
                for title in BOOK_TITLE_RE.findall(text):
                    if title not in seen:
                        seen.add(title)
                        scraped.append(title)
    if scraped:
        scrape_lines = "\n".join(f"- {title}" for title in scraped)
        blocks.append(f"### 书名摘录\n\n{scrape_lines}\n")
    return "\n".join(blocks).strip() + ("\n" if blocks else ""), scraped


def format_conflicts_zh(conflicts: list[Conflict]) -> str:
    if not conflicts:
        return ""
    blocks: list[str] = []
    for conflict in conflicts:
        blocks.append(
            f"### {conflict.id}\n\n"
            f"- {conflict.claim_a.source}: {conflict.claim_a.text}\n"
            f"- {conflict.claim_b.source}: {conflict.claim_b.text}\n"
            f"- inline_tag: `{conflict.inline_tag}`\n"
        )
    return "\n".join(blocks).strip() + "\n"


def load_conflicts(path: Path) -> list[Conflict]:
    if not path.is_file():
        return []
    out: list[Conflict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(Conflict.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError):
            continue
    return out


def prepare_appendices(
    *,
    topic: str,
    artifacts_root: Path,
    strict_topic: bool = False,
) -> list[str]:
    """Write deterministic ZH appendix blobs. Empty blobs are not written."""
    artifacts_topic = artifacts_root / topic
    app_dir = artifacts_topic / "merge" / "appendices"
    app_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    manifest = load_manifest(artifacts_topic)
    if manifest.get("sources"):
        body = format_appendix_a("zh", manifest)
        atomic_write_text(appendix_zh_path(artifacts_topic, BLOB_A), body)
        written.append(BLOB_A)

    sections = load_sections(artifacts_topic)
    quizzes = concat_quiz_markdown(sections)
    if quizzes.strip():
        atomic_write_text(appendix_zh_path(artifacts_topic, BLOB_B), quizzes)
        written.append(BLOB_B)

    biblio, scraped = harvest_bibliography(sections, artifacts_topic / "extract")
    if scraped:
        atomic_write_text(
            app_dir / "biblio_scrape.json",
            json.dumps({"biblio_scrape": scraped}, ensure_ascii=False, indent=2) + "\n",
        )
    if biblio.strip():
        atomic_write_text(appendix_zh_path(artifacts_topic, BLOB_D), biblio)
        written.append(BLOB_D)

    conflicts = load_conflicts(artifacts_topic / "merge" / "conflicts.jsonl")
    conflict_md = format_conflicts_zh(conflicts)
    if conflict_md.strip():
        atomic_write_text(appendix_zh_path(artifacts_topic, BLOB_E), conflict_md)
        written.append(BLOB_E)

    off_path = appendix_zh_path(artifacts_topic, BLOB_C)
    if strict_topic and off_path.is_file() and off_path.read_text(encoding="utf-8").strip():
        written.append(BLOB_C)
    for path in app_dir.glob("*.zh.md"):
        blob_id = path.name[: -len(".zh.md")] if path.name.endswith(".zh.md") else path.stem
        if blob_id not in written and blob_id != BLOB_C:
            path.unlink()
    return written


def yue_heading_for(
    artifacts_topic: Path,
    node_id: str,
    title_zh: str,
) -> str:
    path = translate_node_path(artifacts_topic, "yue", node_id)
    text = read_text_if_exists(path)
    if text:
        heading = first_heading_text(text)
        if heading:
            return heading
    return traditionalize(title_zh)


def cross_ref_line(
    lang: str,
    *,
    node_id: str,
    title_en: str,
    title_zh: str,
    yue_heading: str,
) -> str:
    if lang == "en":
        return f"See {title_en} ({node_id})."
    if lang == "yue":
        return f"見{yue_heading}（{node_id}）。"
    return f"见{title_zh}（{node_id}）。"


def secondary_map(assignments: list[AssignmentItem]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in assignments:
        if not item.secondary_node_id:
            continue
        grouped.setdefault(item.secondary_node_id, []).append(item.primary_node_id)
    return grouped


def strip_leading_heading(markdown: str) -> str:
    lines = markdown.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and heading_lines(lines[i]):
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        return "\n".join(lines[i:]).strip()
    return markdown.strip()


def node_markdown(
    artifacts_topic: Path,
    lang: str,
    node_id: str,
) -> str | None:
    if lang == "zh":
        path = artifacts_topic / "merge" / "nodes" / f"{node_id}.zh.md"
    else:
        folder = "en" if lang == "en" else "yue"
        path = translate_node_path(artifacts_topic, folder, node_id)
    return read_text_if_exists(path)


def appendix_markdown(
    artifacts_topic: Path,
    lang: str,
    blob_id: str,
) -> str | None:
    if lang == "zh" or blob_id == BLOB_A:
        path = appendix_zh_path(artifacts_topic, blob_id)
        text = read_text_if_exists(path)
        if text is None:
            return None
        if blob_id == BLOB_A:
            return text if lang == "zh" else None
        return text
    folder = "en" if lang == "en" else "yue"
    return read_text_if_exists(translate_appendix_path(artifacts_topic, folder, blob_id))


def wrap_appendix(lang: str, blob_id: str, body: str) -> str:
    titles = {
        BLOB_B: APPENDIX_B,
        BLOB_C: APPENDIX_C,
        BLOB_D: APPENDIX_D,
        BLOB_E: APPENDIX_E,
    }
    title = titles[blob_id][lang]
    stripped = body.strip()
    first = first_heading_text(stripped)
    if first and (first == title or first.lstrip("# ").strip() == title):
        content = strip_leading_heading(stripped)
        return f"## {title}\n\n{content}\n" if content else f"## {title}\n"
    if stripped.startswith("## "):
        return stripped + ("\n" if not stripped.endswith("\n") else "")
    return f"## {title}\n\n{stripped}\n"


def toc_from_headings(headings: list[tuple[int, str]]) -> str:
    lines: list[str] = []
    for level, title in headings:
        if level < 2:
            continue
        indent = "  " * (level - 2)
        lines.append(f"{indent}- {title}")
    return "\n".join(lines)


def collect_body_headings(body: str) -> list[tuple[int, str]]:
    return [(level, title) for level, title in heading_lines(body) if level >= 2]


def yaml_quote(value: str) -> str:
    if value == "" or any(ch in value for ch in ":#{}[]&*!|>'\"%@`"):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def format_front_matter(
    *,
    topic: str,
    title: str,
    language: str,
    register: str | None,
    source_files: list[str],
    generated_at: str,
    model: str,
) -> str:
    files = "\n".join(f"  - {path}" for path in source_files) or "  []"
    register_value = "null" if register is None else register
    return (
        "---\n"
        f"topic: {topic}\n"
        f"title: {yaml_quote(title)}\n"
        f"language: {language}\n"
        f"register: {register_value}\n"
        "source_files:\n"
        f"{files}\n"
        f"generated_at: {generated_at}\n"
        f"model: {model}\n"
        f"combiner_version: {COMBINER_VERSION}\n"
        "---\n"
    )


def language_title(outline: Outline, lang: str) -> str:
    if lang == "en":
        return outline.title.en or humanize_topic(outline.title.zh)
    if lang == "yue":
        return traditionalize(outline.title.yue_hint or outline.title.zh)
    return outline.title.zh or outline.title.en


def retarget_headings(markdown: str, target_level: int) -> str:
    """Shift ATX headings so the first heading sits at target_level (cap 1–6)."""
    first = heading_lines(markdown)
    if not first:
        return markdown
    delta = target_level - first[0][0]
    if delta == 0:
        return markdown
    lines: list[str] = []
    for line in markdown.splitlines():
        match = HEADING_RE.match(line)
        if not match:
            lines.append(line)
            continue
        level = max(1, min(6, len(match.group(1)) + delta))
        lines.append("#" * level + " " + match.group(2).strip())
    return "\n".join(lines)


def container_title(
    lang: str,
    artifacts_topic: Path,
    node_id: str,
    title_en: str,
    title_zh: str,
) -> str:
    if lang == "en":
        return title_en
    if lang == "yue":
        return yue_heading_for(artifacts_topic, node_id, title_zh)
    return title_zh


def build_body(
    *,
    outline: Outline,
    artifacts_topic: Path,
    lang: str,
    assignments: list[AssignmentItem],
    appendix_ids: list[str],
    manifest: dict,
) -> str:
    titles = node_titles(outline)
    secondaries = secondary_map(assignments)
    chunks: list[str] = []

    def emit_cross_refs(node_id: str) -> None:
        for primary_id in secondaries.get(node_id, []):
            ten, tzh = titles.get(primary_id, (primary_id, primary_id))
            yue = yue_heading_for(artifacts_topic, primary_id, tzh)
            chunks.append(
                cross_ref_line(
                    lang,
                    node_id=primary_id,
                    title_en=ten,
                    title_zh=tzh,
                    yue_heading=yue,
                )
            )

    for part in outline.parts:
        part_md = node_markdown(artifacts_topic, lang, part.id)
        if part_md and part_md.strip():
            chunks.append(retarget_headings(part_md.strip(), 2))
        else:
            title = container_title(lang, artifacts_topic, part.id, part.title_en, part.title_zh)
            chunks.append(f"## {title}")
        emit_cross_refs(part.id)
        for chapter in part.chapters:
            ch_md = node_markdown(artifacts_topic, lang, chapter.id)
            if ch_md and ch_md.strip():
                chunks.append(retarget_headings(ch_md.strip(), 3))
            else:
                title = container_title(
                    lang, artifacts_topic, chapter.id, chapter.title_en, chapter.title_zh
                )
                chunks.append(f"### {title}")
            emit_cross_refs(chapter.id)
            for section in chapter.sections:
                sec_md = node_markdown(artifacts_topic, lang, section.id)
                if sec_md is None:
                    continue
                chunks.append(retarget_headings(sec_md.strip(), 4))
                emit_cross_refs(section.id)
    for blob_id in appendix_ids:
        if blob_id == BLOB_A:
            chunks.append(format_appendix_a(lang, manifest).strip())
            continue
        raw = appendix_markdown(artifacts_topic, lang, blob_id)
        if not raw or not raw.strip():
            continue
        chunks.append(wrap_appendix(lang, blob_id, raw).strip())
    return "\n\n".join(chunk for chunk in chunks if chunk).strip() + "\n"


def present_appendix_ids(artifacts_topic: Path, strict_topic: bool) -> list[str]:
    ids: list[str] = []
    for blob_id in (BLOB_A, BLOB_B, BLOB_C, BLOB_D, BLOB_E):
        path = appendix_zh_path(artifacts_topic, blob_id)
        if blob_id == BLOB_C and not strict_topic:
            continue
        if path.is_file() and path.read_text(encoding="utf-8").strip():
            ids.append(blob_id)
    return ids


def assemble_document(
    *,
    outline: Outline,
    artifacts_topic: Path,
    lang: str,
    assignments: list[AssignmentItem],
    appendix_ids: list[str],
    manifest: dict,
    topic: str,
    model: str,
    generated_at: str,
    with_front_matter: bool,
) -> str:
    title = language_title(outline, lang)
    body = build_body(
        outline=outline,
        artifacts_topic=artifacts_topic,
        lang=lang,
        assignments=assignments,
        appendix_ids=appendix_ids,
        manifest=manifest,
    )
    headings = collect_body_headings(body)
    toc_title = TOC_TITLES[lang]
    toc = toc_from_headings(headings)
    core = f"# {title}\n\n## {toc_title}\n\n{toc}\n\n{body}".rstrip() + "\n"
    if not with_front_matter:
        return core
    language = "en" if lang == "en" else "yue-Hant-HK"
    register = None if lang == "en" else "hk-written-cantonese"
    sources = [str(row.get("path") or "") for row in (manifest.get("sources") or [])]
    fm = format_front_matter(
        topic=topic,
        title=title,
        language=language,
        register=register,
        source_files=sources,
        generated_at=generated_at,
        model=model,
    )
    return fm + "\n" + core


def heading_signature(text: str) -> list[tuple[int, str]]:
    """Level sequence used to compare ZH/EN/HK trees (titles may differ)."""
    return [(level, title) for level, title in heading_lines(text)]


def heading_levels(text: str) -> list[int]:
    return [level for level, _title in heading_signature(text) if level >= 1]


def outline_heading_levels(text: str) -> list[int]:
    """H1–H3 only: merge concat injects extra #### that translations may drop."""
    return [level for level in heading_levels(text) if level <= 3]


def strip_cjk_from_headings(text: str) -> str:
    """Drop leftover CJK from English heading lines; keep the line if nothing remains."""
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        core = line.rstrip("\n")
        nl = line[len(core) :]
        match = HEADING_RE.match(core)
        if not match:
            out.append(line)
            continue
        hashes, title = match.group(1), match.group(2)
        cleaned = CJK_RE.sub("", title)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -—·")
        if not cleaned:
            out.append(line)
            continue
        out.append(f"{hashes} {cleaned}{nl}")
    return "".join(out)


def count_h1(text: str) -> int:
    return sum(1 for level, _title in heading_signature(text) if level == 1)


def toc_follows_title(text: str, toc_title: str) -> bool:
    headings = heading_signature(text)
    if len(headings) < 2:
        return False
    if headings[0][0] != 1:
        return False
    return headings[1][0] == 2 and headings[1][1] == toc_title


def en_headings_have_cjk(text: str) -> bool:
    for _level, title in heading_signature(text):
        if CJK_RE.search(title):
            return True
    return False


def source_files_exist(manifest: dict) -> list[str]:
    missing: list[str] = []
    for row in manifest.get("sources") or []:
        path = Path(str(row.get("path") or ""))
        if not path.is_file():
            missing.append(str(path))
    return missing


def merge_canary_fail(artifacts_topic: Path) -> str | None:
    nodes_dir = artifacts_topic / "merge" / "nodes"
    if not nodes_dir.is_dir():
        return None
    ratios: list[float] = []
    overlaps: list[float] = []
    for path in sorted(nodes_dir.glob("*.meta.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        unique = float(meta.get("unique_leaf_chars") or 0)
        overlap = float(meta.get("remaining_overlap") or 0)
        zh_path = path.with_name(path.name.replace(".meta.json", ".zh.md"))
        if unique <= 0 or not zh_path.is_file():
            continue
        out_chars = len(zh_path.read_text(encoding="utf-8"))
        ratios.append(out_chars / unique)
        overlaps.append(overlap)
    if not ratios:
        return None
    if statistics.median(ratios) < SUMMARY_CANARY_RATIO and statistics.median(overlaps) < SUMMARY_CANARY_OVERLAP:
        return (
            f"merge summarization canary: median output/unique="
            f"{statistics.median(ratios):.3f} overlap={statistics.median(overlaps):.3f}"
        )
    return None


def validate_documents(
    *,
    zh: str,
    en: str,
    yue: str,
    outline: Outline,
    artifacts_topic: Path,
    manifest: dict,
) -> list[str]:
    errors: list[str] = []
    failed = failed_node_ids(artifacts_topic)
    if failed:
        errors.append("FAILED nodes: " + ", ".join(failed))
    for node_id, _en, _zh in outline_leaves(outline):
        if (artifacts_topic / "merge" / "nodes" / f"{node_id}.FAILED.md").is_file():
            errors.append(f"FAILED node {node_id}")
    if count_h1(zh) != 1:
        errors.append(f"ZH H1 count={count_h1(zh)}")
    if count_h1(en) != 1:
        errors.append(f"EN H1 count={count_h1(en)}")
    if count_h1(yue) != 1:
        errors.append(f"HK H1 count={count_h1(yue)}")
    if not toc_follows_title(zh, TOC_TITLES["zh"]):
        errors.append("ZH TOC not immediately after title")
    if not toc_follows_title(en, TOC_TITLES["en"]):
        errors.append("EN TOC not immediately after title")
    if not toc_follows_title(yue, TOC_TITLES["yue"]):
        errors.append("HK TOC not immediately after title")
    if en_headings_have_cjk(en):
        errors.append("EN headings contain CJK")
    density = en_han_density(en)
    if density > HAN_DENSITY_CAP:
        errors.append(f"EN Han density {density:.4%} > {HAN_DENSITY_CAP:.2%}")
    yue_ok, yue_reasons = evaluate_cantonese(yue)
    if not yue_ok:
        errors.append("HK Cantonese: " + "; ".join(yue_reasons))
    missing_sources = source_files_exist(manifest)
    if missing_sources:
        errors.append("missing source_files: " + ", ".join(missing_sources))
    canary = merge_canary_fail(artifacts_topic)
    if canary:
        errors.append(canary)
    return errors


def _write_partials(dest_to_text: dict[Path, str]) -> None:
    for dest, text in dest_to_text.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".partial")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())


def _replace_partials(dests: list[Path]) -> None:
    for dest in dests:
        tmp = dest.with_name(dest.name + ".partial")
        os.replace(tmp, dest)


def word_count(text: str) -> int:
    body = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)
    return len(re.findall(r"\S+", body))


def run_assemble(
    *,
    topic: str,
    book_root: Path,
    artifacts_root: Path,
    model: str = "grok-4",
    strict_topic: bool = False,
    generated_at: str | None = None,
) -> dict[str, Path]:
    artifacts_topic = artifacts_root / topic
    outline_path = artifacts_topic / "outline" / "outline.json"
    if not outline_path.is_file():
        raise AssembleError("outline.json not found; run --stage outline first.")
    assignment_path = artifacts_topic / "outline" / "assignment.json"
    assignments = load_assignments(assignment_path) if assignment_path.is_file() else []
    outline = load_outline(outline_path)
    prepare_appendices(
        topic=topic,
        artifacts_root=artifacts_root,
        strict_topic=strict_topic,
    )
    appendix_ids = present_appendix_ids(artifacts_topic, strict_topic)
    manifest = load_manifest(artifacts_topic)
    stamp = generated_at or _utc_now()

    zh = assemble_document(
        outline=outline,
        artifacts_topic=artifacts_topic,
        lang="zh",
        assignments=assignments,
        appendix_ids=appendix_ids,
        manifest=manifest,
        topic=topic,
        model=model,
        generated_at=stamp,
        with_front_matter=False,
    )
    en = assemble_document(
        outline=outline,
        artifacts_topic=artifacts_topic,
        lang="en",
        assignments=assignments,
        appendix_ids=appendix_ids,
        manifest=manifest,
        topic=topic,
        model=model,
        generated_at=stamp,
        with_front_matter=True,
    )
    en = strip_cjk_from_headings(en)
    yue = assemble_document(
        outline=outline,
        artifacts_topic=artifacts_topic,
        lang="yue",
        assignments=assignments,
        appendix_ids=appendix_ids,
        manifest=manifest,
        topic=topic,
        model=model,
        generated_at=stamp,
        with_front_matter=True,
    )
    yue = traditionalize(yue)

    zh_path = artifacts_topic / "compiled.zh.md"
    en_path = Path(book_root) / f"{topic}.md"
    hk_path = Path(book_root) / f"{topic}.hk.md"
    dests = {zh_path: zh, en_path: en, hk_path: yue}
    _write_partials(dests)

    errors = validate_documents(
        zh=zh,
        en=en,
        yue=yue,
        outline=outline,
        artifacts_topic=artifacts_topic,
        manifest=manifest,
    )
    if errors:
        raise AssembleError("; ".join(errors))

    _replace_partials([zh_path, en_path, hk_path])
    words = word_count(en)
    print(f"[assemble] {en_path.as_posix()} {words:,} words")
    return {"zh": zh_path, "en": en_path, "hk": hk_path}
