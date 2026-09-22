"""Extract, de-noise, and section markdown sources."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from book_combiner.discover import Manifest
from book_combiner.hierarchy import (
    HEADING_RE,
    LIST_ITEM_RE,
    TOC_KEYS,
    compact_key,
    display_title,
    looks_like_chapter,
    looks_like_section,
    normalize_title,
    parse_toc_entries,
    repair_heading_paths,
)
from book_combiner.models import Section

EXTRACT_CODE_VERSION = "1"
CHUNK_CHAR_LIMIT = 4000

PREFACE_KEYS = frozenset({"前言", "序言", "导读", "引言", "preface"})
COVER_KEYS = frozenset({"封面"})
COPYRIGHT_HEADING_KEYS = frozenset({"版权信息", "cip"})

HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
ISBN_RE = re.compile(r"ISBN", re.I)
CHOICE_A_RE = re.compile(r"(?:^|[\s　])A[.．、]")
CHOICE_B_RE = re.compile(r"(?:^|[\s　])B[.．、]")
CHOICE_C_RE = re.compile(r"(?:^|[\s　])C[.．、]")
FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?:\*\*|__)?\s*(图\s*\d+(?:[.\-－—]\d+)?)(?:[　\s]+(.+?))?\s*(?:\*\*|__)?\s*$"
)
HASH_KEEP_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)
INSTRUCTIONAL_73855_RE = re.compile(
    r"7\s*[%％].{0,40}38.{0,40}55|7.{0,20}38.{0,20}55.{0,20}[%％]"
)
NAME_DOT_RE = re.compile(r"[\u4e00-\u9fff]·[\u4e00-\u9fff]")
TEACHING_CUE_RE = re.compile(r"名言|说过|指出|提出|理论|定律|规则|研究表明")
SKILL_TEST_RE = re.compile(
    r"表情辨识能力测试|普通表情的辨识|常见表情的辨识|复杂表情的辨识|表情的辨识"
)


@dataclass
class DroppedItem:
    reason: str
    start_line: int
    end_line: int
    heading: str | None
    text: str


@dataclass
class ExtractFileResult:
    stem: str
    sections: list[Section] = field(default_factory=list)
    dropped: list[DroppedItem] = field(default_factory=list)
    source_books: list[str] = field(default_factory=list)
    forthcoming_books: list[str] = field(default_factory=list)
    return_toc_count: int = 0


@dataclass
class _Segment:
    heading_line: int | None
    heading_raw_level: int
    heading_title: str
    start_line: int
    end_line: int
    body_lines: list[str]
    heading_line_text: str


def content_hash(text: str) -> str:
    """Whitespace+punctuation stripped sha256 of section text."""
    normalized = unicodedata.normalize("NFKC", text)
    stripped = HASH_KEEP_RE.sub("", normalized)
    return hashlib.sha256(stripped.encode("utf-8")).hexdigest()


def _visible_line(line: str) -> str:
    text = line.strip()
    text = re.sub(r"^#+\s*", "", text)
    item = LIST_ITEM_RE.match(text)
    if item:
        text = item.group(2)
    text = re.sub(r"^\*\*(.*)\*\*$", r"\1", text)
    text = re.sub(r"^__(.*)__$", r"\1", text)
    text = re.sub(r"^\*(.*)\*$", r"\1", text)
    text = re.sub(r"^_(.*)_$", r"\1", text)
    return normalize_title(text)


def is_return_to_toc_line(line: str) -> bool:
    return _visible_line(line) == "返回总目录"


def is_standalone_return_to_toc(line: str) -> bool:
    """Bare 返回总目录 (markdown-wrapped ok); TOC bullets do not count as cuts."""
    if LIST_ITEM_RE.match(line):
        return False
    return is_return_to_toc_line(line)


def is_hr(line: str) -> bool:
    return bool(HR_RE.match(line))


def is_toc_heading(title: str) -> bool:
    return compact_key(title) in TOC_KEYS


def is_preface_heading(title: str) -> bool:
    key = compact_key(title)
    if key in PREFACE_KEYS:
        return True
    n = normalize_title(title).replace(" ", "")
    return any(n.startswith(p) or p in n[:6] for p in ("前言", "序言", "导读", "引言"))


def is_bibliography_heading(title: str) -> bool:
    n = compact_key(title)
    return n.startswith("参考书目") or n.startswith("索引")


def is_publisher_note(title: str) -> bool:
    return compact_key(title).startswith("出版说明")


def is_copyright_heading(title: str) -> bool:
    key = compact_key(title)
    if key in COPYRIGHT_HEADING_KEYS:
        return True
    n = normalize_title(title)
    return "图书在版编目" in n or n.upper().startswith("CIP")


def is_cover_title(title: str) -> bool:
    return compact_key(title) in COVER_KEYS


def is_afterword_heading(title: str) -> bool:
    return "后记" in compact_key(title)


def _body_text(lines: list[str]) -> str:
    kept = [ln for ln in lines if not is_hr(ln)]
    return "\n".join(kept).strip("\n")


def _is_metadata_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if ISBN_RE.search(s) or "版权所有" in s or "图书在版编目" in s:
        return True
    if re.search(r"责任编辑|责任印制|出版发行|中国版本图书馆|\bCIP\b", s):
        return True
    if re.search(r"\*\*[^*]{1,12}：\*\*", s):
        return True
    return False


def _is_prose_body(lines: list[str]) -> bool:
    chunks: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or is_hr(ln) or LIST_ITEM_RE.match(ln) or _is_metadata_line(ln):
            continue
        chunks.append(s)
    if not chunks:
        return False
    return len("".join(chunks)) >= 40


def is_cip_or_copyright_text(text: str) -> bool:
    has_isbn = bool(ISBN_RE.search(text))
    has_rights = "版权所有" in text
    has_cip = "图书在版编目" in text or "CIP数据" in text or re.search(r"\bCIP\b", text)
    has_meta = "责任印制" in text or "责任编辑" in text
    if has_cip:
        return True
    if has_isbn and (has_rights or has_meta):
        return True
    return False


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip("\n"))
    return [p.strip("\n") for p in parts if p.strip()]


def _is_colophon_paragraph(para: str) -> bool:
    if ISBN_RE.search(para):
        return True
    if "版权所有" in para:
        return True
    if "责任印制" in para or "责任编辑" in para:
        return True
    if "限于时间和水平" in para:
        return True
    return False


def _is_instructional_paragraph(para: str) -> bool:
    if INSTRUCTIONAL_73855_RE.search(para):
        return True
    if re.search(r"[“\"「].+[”\"」]", para) and TEACHING_CUE_RE.search(para):
        return True
    if re.search(r"\b(?:METT|FACS)\b", para):
        return True
    if NAME_DOT_RE.search(para) and TEACHING_CUE_RE.search(para):
        return True
    return False


def _is_sales_copy(para: str) -> bool:
    return bool(
        re.search(
            r"权威之作|职场达人|谈判高手|管理大师|恋爱专家|无论你是销售员还是管理者",
            para,
        )
    )


def _filter_afterword(text: str) -> tuple[str, list[str]]:
    """Keep instructional 后记 paragraphs; drop colophon/closing marketing."""
    paras = _split_paragraphs(text)
    if not paras:
        return "", []
    instr_idx = [i for i, p in enumerate(paras) if _is_instructional_paragraph(p)]
    if not instr_idx:
        return "", paras
    last_instr = instr_idx[-1]
    kept: list[str] = []
    dropped: list[str] = []
    for i, para in enumerate(paras):
        if _is_colophon_paragraph(para) or (i > last_instr and _is_sales_copy(para)):
            dropped.append(para)
        else:
            kept.append(para)
    return "\n\n".join(kept).strip(), dropped


def replace_figure_captions(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        m = FIGURE_CAPTION_RE.match(line)
        if m and "如图" not in line:
            original = line.strip()
            original = re.sub(r"^\*\*|\*\*$", "", original).strip()
            original = re.sub(r"^__|__$", "", original).strip()
            out.append(f"*[Figure omitted from source conversion: {original}]*")
        else:
            out.append(line)
    return "\n".join(out)


def _has_quiz_choices(text: str) -> bool:
    return bool(CHOICE_A_RE.search(text) and CHOICE_B_RE.search(text) and CHOICE_C_RE.search(text))


def classify_kind(title: str, body: str) -> str:
    if is_bibliography_heading(title):
        return "bibliography"
    if is_preface_heading(title) or is_afterword_heading(title):
        return "preface"
    if SKILL_TEST_RE.search(normalize_title(title)):
        return "skill_test"
    if "心理测试" in title:
        if _has_quiz_choices(body) or "结果分析" in body:
            return "quiz"
        return "body"
    return "body"


def _parse_segments(lines: list[str]) -> list[_Segment]:
    heading_at: list[int] = []
    for i, line in enumerate(lines):
        if HEADING_RE.match(line):
            heading_at.append(i)
    segments: list[_Segment] = []
    if heading_at and heading_at[0] > 0:
        segments.append(
            _Segment(
                heading_line=None,
                heading_raw_level=0,
                heading_title="",
                start_line=1,
                end_line=heading_at[0],
                body_lines=lines[0 : heading_at[0]],
                heading_line_text="",
            )
        )
    if not heading_at:
        if any(ln.strip() for ln in lines):
            segments.append(
                _Segment(
                    heading_line=None,
                    heading_raw_level=0,
                    heading_title="",
                    start_line=1,
                    end_line=len(lines),
                    body_lines=list(lines),
                    heading_line_text="",
                )
            )
        return segments
    for idx, start in enumerate(heading_at):
        end = heading_at[idx + 1] if idx + 1 < len(heading_at) else len(lines)
        m = HEADING_RE.match(lines[start])
        assert m is not None
        title = m.group(2).strip()
        segments.append(
            _Segment(
                heading_line=start + 1,
                heading_raw_level=len(m.group(1)),
                heading_title=title,
                start_line=start + 1,
                end_line=end,
                body_lines=lines[start + 1 : end],
                heading_line_text=lines[start],
            )
        )
    return segments


def _harvest_forthcoming(body_lines: list[str]) -> list[str]:
    titles: list[str] = []
    for ln in body_lines:
        item = LIST_ITEM_RE.match(ln)
        if not item:
            continue
        title = display_title(item.group(2))
        if title and title != "返回总目录" and compact_key(title) not in COVER_KEYS:
            titles.append(title)
    return titles


def _label_source_book(
    stem: str,
    book_number: int,
    h1: str | None,
    forthcoming: list[str],
) -> str:
    if h1:
        n = compact_key(h1)
        matches = [
            f
            for f in forthcoming
            if n and (compact_key(f).startswith(n) or n in compact_key(f) or compact_key(f) in n)
        ]
        unique: list[str] = []
        for m in matches:
            if m not in unique:
                unique.append(m)
        if len(unique) == 1:
            return display_title(h1)
        return f"{stem}#{book_number}"
    if 1 <= book_number <= len(forthcoming):
        return forthcoming[book_number - 1]
    return f"{stem}#{book_number}"


def _looks_like_book_title(title: str) -> bool:
    if not title:
        return False
    if is_toc_heading(title) or is_copyright_heading(title) or is_publisher_note(title):
        return False
    if is_preface_heading(title) or is_bibliography_heading(title) or is_afterword_heading(title):
        return False
    if looks_like_chapter(title) or looks_like_section(title):
        return False
    return True


def _cip_follows_segment(segments: list[_Segment], index: int) -> bool:
    """True when this H1 is a book title with a CIP/copyright block after it."""
    seg = segments[index]
    if seg.heading_raw_level != 1 or not _looks_like_book_title(seg.heading_title):
        return False
    if is_cip_or_copyright_text(_body_text(seg.body_lines)):
        return True
    # A later CIP heading is not this title's colophon if the title already has body.
    if any(
        ln.strip() and not is_hr(ln) and not _is_metadata_line(ln) for ln in seg.body_lines
    ):
        return False
    j = index + 1
    while j < len(segments):
        nxt = segments[j]
        if is_toc_heading(nxt.heading_title):
            j += 1
            continue
        if is_copyright_heading(nxt.heading_title) or is_cip_or_copyright_text(
            _body_text(nxt.body_lines)
        ):
            return True
        break
    return False


def _drop_reason_for_segment(seg: _Segment) -> str | None:
    title = seg.heading_title
    if not title:
        text = _body_text(seg.body_lines)
        if is_cip_or_copyright_text(text):
            return "copyright"
        return None
    if is_toc_heading(title):
        return "toc"
    if is_publisher_note(title):
        return "publisher_note"
    if is_copyright_heading(title):
        return "copyright"
    if is_cover_title(title):
        return "cover"
    if compact_key(title) == "返回总目录" or is_return_to_toc_line(seg.heading_line_text):
        return "omnibus_marker"
    if is_cip_or_copyright_text(_body_text(seg.body_lines)) and not _is_prose_body(seg.body_lines):
        if seg.heading_raw_level == 1 and _looks_like_book_title(title):
            return "copyright_body"
        return "copyright"
    return None


def _in_toc_continuation(seg: _Segment) -> bool:
    if is_toc_heading(seg.heading_title):
        return True
    if seg.heading_raw_level >= 2 and not _is_prose_body(seg.body_lines):
        return True
    return False


def _rewrite_figures_and_drop_hrs(text: str) -> str:
    text = replace_figure_captions(text)
    lines = [ln for ln in text.splitlines() if not is_hr(ln)]
    cleaned: list[str] = []
    blank = False
    for ln in lines:
        if not ln.strip():
            if not blank and cleaned:
                cleaned.append("")
            blank = True
        else:
            cleaned.append(ln)
            blank = False
    return "\n".join(cleaned).strip("\n")


def _chunk_text(text: str, max_input_chars: int) -> list[str]:
    """Split oversized leaves on blank-line groups into chunks of ≤ cap."""
    if len(text) <= max_input_chars:
        return [text]
    cap = min(CHUNK_CHAR_LIMIT, max_input_chars) if max_input_chars > 0 else CHUNK_CHAR_LIMIT
    paras = _split_paragraphs(text)
    if not paras:
        return [text]

    chunks: list[list[int]] = []
    buf: list[int] = []

    def buf_len(idxs: list[int]) -> int:
        return len("\n\n".join(paras[i] for i in idxs))

    for i, para in enumerate(paras):
        trial = buf + [i]
        if buf and buf_len(trial) > cap:
            chunks.append(buf)
            overlap = buf[-1]
            if buf_len([overlap, i]) <= cap:
                buf = [overlap, i]
            else:
                buf = [i]
        else:
            buf.append(i)
    if buf:
        chunks.append(buf)
    return ["\n\n".join(paras[i] for i in idxs) for idxs in chunks]


def _make_section(
    *,
    stem: str,
    source_book: str,
    heading_path: list[str],
    level: int,
    start_line: int,
    end_line: int,
    text: str,
    kind: str,
    suffix: str = "",
) -> Section:
    sid = f"{stem}:{start_line}:{end_line}{suffix}"
    return Section(
        id=sid,
        source_stem=stem,
        source_book=source_book,
        heading_path=heading_path,
        level=level,
        start_line=start_line,
        end_line=end_line,
        text=text,
        kind=kind,
        char_count=len(text),
        content_hash=content_hash(text),
        duplicate_of=None,
    )


def _next_keepable(
    segments: list[_Segment],
    drop_flags: list[str | None],
    start: int,
    *,
    h1_only: bool = False,
) -> _Segment | None:
    for j in range(start, len(segments)):
        if drop_flags[j]:
            continue
        nxt = segments[j]
        if nxt.heading_line is None or not nxt.heading_title:
            continue
        if is_toc_heading(nxt.heading_title):
            continue
        if h1_only and nxt.heading_raw_level != 1:
            continue
        return nxt
    return None


def _assign_drop_flags(segments: list[_Segment]) -> list[str | None]:
    drop_flags: list[str | None] = [None] * len(segments)
    in_toc = False
    for i, seg in enumerate(segments):
        if in_toc:
            if _in_toc_continuation(seg):
                drop_flags[i] = "toc"
                continue
            in_toc = False
        reason = _drop_reason_for_segment(seg)
        if reason == "toc":
            in_toc = True
            drop_flags[i] = "toc"
            continue
        if reason == "copyright_body":
            drop_flags[i] = "copyright"
            continue
        if reason:
            drop_flags[i] = reason
    return drop_flags


def _collect_book_cuts(
    segments: list[_Segment],
    drop_flags: list[str | None],
    *,
    is_omnibus: bool,
) -> list[int]:
    """Record cut line numbers only; labels are applied later in document order."""
    cuts: set[int] = set()

    first_keep: int | None = None
    for i, seg in enumerate(segments):
        if drop_flags[i] is None:
            first_keep = seg.start_line
            break
    if first_keep is not None:
        cuts.add(first_keep)

    if not is_omnibus:
        return sorted(cuts)

    seen_keepable = False
    for i, seg in enumerate(segments):
        if drop_flags[i] is None and (seg.heading_title or _is_prose_body(seg.body_lines)):
            seen_keepable = True
        if not seen_keepable:
            continue
        if seg.heading_raw_level == 1 and _cip_follows_segment(segments, i):
            cuts.add(seg.start_line)

    for i, seg in enumerate(segments):
        if drop_flags[i] != "copyright" or seg.heading_raw_level != 1:
            continue
        if not is_copyright_heading(seg.heading_title):
            continue
        if not any(drop_flags[k] is None for k in range(i)):
            continue
        nxt = _next_keepable(segments, drop_flags, i + 1, h1_only=True)
        if nxt is not None:
            cuts.add(nxt.start_line)

    for i, seg in enumerate(segments):
        heading_marker = bool(seg.heading_line_text) and is_standalone_return_to_toc(
            seg.heading_line_text
        )
        body_marker = any(is_standalone_return_to_toc(ln) for ln in seg.body_lines)
        if not heading_marker and not body_marker:
            continue
        nxt = _next_keepable(segments, drop_flags, i + 1)
        if nxt is None:
            continue
        # Opening TOC of a book that already started (CIP/title/前言 cut) is not a new book.
        prev_keep: _Segment | None = None
        for k in range(i - 1, -1, -1):
            if drop_flags[k] is None and segments[k].heading_title:
                prev_keep = segments[k]
                break
        if prev_keep is not None and prev_keep.start_line in cuts:
            continue
        cuts.add(nxt.start_line)

    return sorted(cuts)


def _path_is_under(path: list[str], parent: list[str]) -> bool:
    return len(path) > len(parent) and path[: len(parent)] == parent


def extract_markdown(
    text: str,
    stem: str,
    max_input_chars: int = 6000,
) -> ExtractFileResult:
    lines = text.splitlines()
    result = ExtractFileResult(stem=stem)
    result.return_toc_count = sum(1 for ln in lines if is_return_to_toc_line(ln))

    segments = _parse_segments(lines)
    if not segments:
        return result

    forthcoming: list[str] = []
    for seg in segments:
        if is_toc_heading(seg.heading_title) and compact_key(seg.heading_title) == "总目录":
            forthcoming = _harvest_forthcoming(seg.body_lines)
            break
    result.forthcoming_books = list(forthcoming)

    is_omnibus = bool(forthcoming) or result.return_toc_count > 0
    drop_flags = _assign_drop_flags(segments)
    ordered_cuts = _collect_book_cuts(segments, drop_flags, is_omnibus=is_omnibus)

    book_start_line: dict[int, str] = {}
    used_labels: set[str] = set()
    for n, line in enumerate(ordered_cuts, start=1):
        h1 = None
        for seg in segments:
            if seg.start_line == line and seg.heading_raw_level == 1:
                h1 = seg.heading_title
                break
        label = _label_source_book(stem, n, h1 if n > 1 else None, forthcoming)
        if n == 1 and forthcoming:
            label = forthcoming[0]
        if label in used_labels:
            label = f"{stem}#{n}"
        used_labels.add(label)
        book_start_line[line] = label

    def source_book_for(line: int) -> str:
        current = stem
        for cut in ordered_cuts:
            if cut <= line:
                current = book_start_line[cut]
            else:
                break
        return current

    result.source_books = [book_start_line[k] for k in ordered_cuts] or [stem]

    toc_by_book: dict[str, list[list[str]]] = {}
    current_book_for_toc = result.source_books[0] if result.source_books else stem
    pending_toc_lines: list[str] = []
    for i, seg in enumerate(segments):
        sb = source_book_for(seg.start_line)
        if sb != current_book_for_toc:
            if pending_toc_lines:
                toc_by_book[current_book_for_toc] = parse_toc_entries(pending_toc_lines)
            pending_toc_lines = []
            current_book_for_toc = sb
        if drop_flags[i] == "toc":
            if seg.heading_line_text:
                pending_toc_lines.append(seg.heading_line_text)
            pending_toc_lines.extend(seg.body_lines)
        elif drop_flags[i] is None and pending_toc_lines:
            toc_by_book[sb] = parse_toc_entries(pending_toc_lines)
            pending_toc_lines = []
    if pending_toc_lines:
        toc_by_book[current_book_for_toc] = parse_toc_entries(pending_toc_lines)

    for i, seg in enumerate(segments):
        reason = drop_flags[i]
        if not reason:
            continue
        raw_lines: list[str] = []
        if seg.heading_line_text:
            raw_lines.append(seg.heading_line_text)
        raw_lines.extend(seg.body_lines)
        result.dropped.append(
            DroppedItem(
                reason=reason,
                start_line=seg.start_line,
                end_line=seg.end_line,
                heading=seg.heading_title or None,
                text=_body_text(raw_lines),
            )
        )

    keep_indices = [i for i, flag in enumerate(drop_flags) if flag is None]
    headings_meta: list[tuple[int, int, str]] = []
    for i in keep_indices:
        seg = segments[i]
        if not seg.heading_title:
            continue
        headings_meta.append((i, seg.heading_raw_level, seg.heading_title))

    paths_by_index: dict[int, list[str]] = {}
    by_book: dict[str, list[tuple[int, int, str]]] = {}
    for i, raw_level, title in headings_meta:
        sb = source_book_for(segments[i].start_line)
        by_book.setdefault(sb, []).append((i, raw_level, title))
    for sb, items in by_book.items():
        repaired = repair_heading_paths(
            [(raw, title) for _, raw, title in items],
            toc_by_book.get(sb, []),
        )
        for (i, _raw, _title), path in zip(items, repaired):
            paths_by_index[i] = path

    pending: list[tuple[int, list[str], str, str, str]] = []
    # (seg_index, path, title, body, source_book)
    for i in keep_indices:
        seg = segments[i]
        title = seg.heading_title
        path = paths_by_index.get(i) or ([display_title(title)] if title else [])
        body = _body_text(seg.body_lines)
        body = "\n".join(ln for ln in body.splitlines() if not is_return_to_toc_line(ln)).strip("\n")
        body = _rewrite_figures_and_drop_hrs(body)

        if title and is_afterword_heading(title):
            body, dropped_paras = _filter_afterword(body)
            for para in dropped_paras:
                result.dropped.append(
                    DroppedItem(
                        reason="colophon",
                        start_line=seg.start_line,
                        end_line=seg.end_line,
                        heading=title,
                        text=para,
                    )
                )
            if not body.strip():
                result.dropped.append(
                    DroppedItem(
                        reason="colophon",
                        start_line=seg.start_line,
                        end_line=seg.end_line,
                        heading=title,
                        text="",
                    )
                )
                continue

        if not body.strip():
            continue
        pending.append((i, path, title, body.strip(), source_book_for(seg.start_line)))

    kinds: list[str] = []
    for i, path, title, body, _sb in pending:
        subtree = body
        for j, other_path, other_title, other_body, _s in pending:
            if j != i and _path_is_under(other_path, path):
                subtree = subtree + "\n" + other_title + "\n" + other_body
        kind = classify_kind(title, subtree) if title else "body"
        if any(is_preface_heading(p) or is_afterword_heading(p) for p in path):
            if kind != "bibliography":
                kind = "preface"
        kinds.append(kind)

    quiz_paths = [pending[n][1] for n, kind in enumerate(kinds) if kind == "quiz"]
    for n, kind in enumerate(kinds):
        if kind == "quiz":
            continue
        path = pending[n][1]
        if any(_path_is_under(path, qp) for qp in quiz_paths):
            kinds[n] = "quiz"

    built: list[Section] = []
    for (i, path, _title, body, sb), kind in zip(pending, kinds):
        seg = segments[i]
        level = len(path) if path else 1
        pieces = _chunk_text(body, max_input_chars)
        multi = len(pieces) > 1
        for p_i, chunk in enumerate(pieces):
            suffix = f"#p{p_i}" if multi else ""
            built.append(
                _make_section(
                    stem=stem,
                    source_book=sb,
                    heading_path=path,
                    level=level,
                    start_line=seg.start_line,
                    end_line=seg.end_line,
                    text=chunk,
                    kind=kind,
                    suffix=suffix,
                )
            )

    merged: list[Section] = []
    for sec in built:
        if (
            merged
            and merged[-1].heading_path == sec.heading_path
            and merged[-1].source_book == sec.source_book
            and merged[-1].kind == sec.kind
            and "#p" not in merged[-1].id
            and "#p" not in sec.id
        ):
            prev = merged[-1]
            text = (prev.text.rstrip() + "\n\n" + sec.text.lstrip()).strip()
            merged[-1] = _make_section(
                stem=prev.source_stem,
                source_book=prev.source_book,
                heading_path=prev.heading_path,
                level=prev.level,
                start_line=prev.start_line,
                end_line=sec.end_line,
                text=text,
                kind=prev.kind,
            )
        else:
            merged.append(sec)
    result.sections = merged
    return result


def write_extract_artifacts(result: ExtractFileResult, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    sections_path = extract_dir / f"{result.stem}.sections.json"
    dropped_path = extract_dir / f"{result.stem}.dropped.json"
    sections_path.write_text(
        json.dumps([asdict(s) for s in result.sections], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dropped_path.write_text(
        json.dumps(
            {
                "stem": result.stem,
                "source_books": result.source_books,
                "forthcoming_books": result.forthcoming_books,
                "return_toc_count": result.return_toc_count,
                "extract_code_version": EXTRACT_CODE_VERSION,
                "items": [asdict(d) for d in result.dropped],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def extract_file_is_fresh(src_path: Path, extract_dir: Path, stem: str) -> bool:
    """True when extract JSON exists and is at least as new as the source file."""
    sections_path = extract_dir / f"{stem}.sections.json"
    if not src_path.is_file() or not sections_path.is_file():
        return False
    return sections_path.stat().st_mtime >= src_path.stat().st_mtime


def extract_from_manifest(
    manifest: Manifest,
    artifacts_root: Path,
    max_input_chars: int = 6000,
    force: bool = False,
) -> list[ExtractFileResult]:
    extract_dir = artifacts_root / manifest.topic / "extract"
    results: list[ExtractFileResult] = []
    for source in manifest.sources:
        src_path = Path(source.path)
        if not force and extract_file_is_fresh(src_path, extract_dir, source.stem):
            print(f"[extract] {src_path.name} skip (fresh)")
            continue
        text = src_path.read_text(encoding="utf-8")
        result = extract_markdown(text, source.stem, max_input_chars=max_input_chars)
        write_extract_artifacts(result, extract_dir)
        results.append(result)
        dropped_reasons = sorted({d.reason for d in result.dropped})
        reason_note = f" ({','.join(dropped_reasons)})" if dropped_reasons else ""
        omnibus = ""
        if len(result.source_books) > 1:
            omnibus = f" omnibus_books={len(result.source_books)}"
        print(
            f"[extract] {src_path.name}{omnibus} "
            f"sections={len(result.sections)} dropped={len(result.dropped)}{reason_note}"
        )
    return results
