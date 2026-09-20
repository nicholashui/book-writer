"""TOC-guided heading repair. Markdown # depth is not trusted."""

from __future__ import annotations

import re
import unicodedata

CHAPTER_RE = re.compile(
    r"^【?第[0-9零○〇一二三四五六七八九十百千两]+章"
)
SECTION_RE = re.compile(
    r"^【?第[0-9零○〇一二三四五六七八九十百千两]+节"
)
LIST_ITEM_RE = re.compile(r"^(\s*)(?:[-*+]|[0-9]+[.)、]|[一二三四五六七八九十]+、)\s+(.*\S)\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

TOC_KEYS = frozenset({"目录", "总目录", "contents", "微表情相关章节目录"})

_PUNCT_STRIP = re.compile(r"[\s\u3000：:、，。！？!?,;；\-—–_·•\.．]+")


def display_title(title: str) -> str:
    """Strip emphasis and collapse spaces without NFKC (keeps ： and ，)."""
    text = re.sub(r"[*_`]+", "", title)
    text = text.replace("\u3000", " ")
    text = re.sub(r" +", " ", text).strip()
    return text


def normalize_title(title: str) -> str:
    """NFKC, strip emphasis, collapse whitespace (including ideographic)."""
    text = unicodedata.normalize("NFKC", title)
    text = re.sub(r"[*_`]+", "", text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_key(title: str) -> str:
    """Punctuation-and-space-free key for TOC/body matching."""
    text = normalize_title(title)
    return _PUNCT_STRIP.sub("", text).lower()


def looks_like_chapter(title: str) -> bool:
    return bool(CHAPTER_RE.match(normalize_title(title)))


def looks_like_section(title: str) -> bool:
    return bool(SECTION_RE.match(normalize_title(title)))


def numbered_depth(title: str) -> int | None:
    """1.1 → 2, 1.1.1 → 3. Bare '1. title' is not treated as depth."""
    n = normalize_title(title)
    m = re.match(r"^(\d+(?:[.\-．]\d+)+)\b", n)
    if not m:
        return None
    return m.group(1).count(".") + m.group(1).count("-") + m.group(1).count("．") + 1


def looks_like_front_or_back_matter(title: str) -> bool:
    key = compact_key(title)
    if key in {"前言", "序言", "导读", "引言", "preface", "参考书目", "索引"}:
        return True
    if "后记" in key:
        return True
    n = display_title(title).replace(" ", "")
    return any(n.startswith(p) or p in n[:8] for p in ("前言", "序言", "导读", "引言"))


def structural_level(title: str, raw_level: int) -> int:
    """Best-effort outline depth; chapters beat the raw hash count."""
    if looks_like_chapter(title) or looks_like_front_or_back_matter(title):
        return 1
    if looks_like_section(title):
        return 2
    depth = numbered_depth(title)
    if depth is not None:
        return depth
    return max(1, raw_level)


def is_magazine_quiz_heading(title: str) -> bool:
    """True for 心理测试　你的情绪稳定吗 / 心理测试：…, not bare glossary 心理测试."""
    n = normalize_title(title)
    return bool(re.match(r"^心理测试[\s:：]+.+$", n))


def is_short_topical(title: str, raw_level: int) -> bool:
    """Promoted H1 that is not a chapter/preface-style title."""
    if raw_level != 1:
        return False
    n = normalize_title(title)
    if looks_like_chapter(n) or looks_like_section(n) or looks_like_front_or_back_matter(n):
        return False
    if numbered_depth(n) is not None:
        return False
    if len(n) > 40:
        return False
    return True


def parse_toc_entries(lines: list[str]) -> list[list[str]]:
    """Return heading paths described by a TOC block (headings + indented lists)."""
    paths: list[list[str]] = []
    stack: list[tuple[int, str]] = []
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        heading = HEADING_RE.match(line)
        if heading:
            hashes, title = heading.group(1), heading.group(2).strip()
            title = display_title(title)
            if compact_key(title) in TOC_KEYS:
                continue
            level = len(hashes)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            paths.append([t for _, t in stack])
            continue
        item = LIST_ITEM_RE.match(line)
        if not item:
            continue
        indent = len(item.group(1).replace("\t", "  "))
        title = display_title(item.group(2))
        if not title:
            continue
        # Nested lists sit below the most recent heading; indent 0 is a sibling of that heading's children.
        level = 100 + indent
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        paths.append([t for _, t in stack])
    return paths


def _path_from_toc(
    title: str,
    toc_paths: list[list[str]],
    used: set[int],
) -> list[str] | None:
    key = compact_key(title)
    if not key:
        return None
    for i, path in enumerate(toc_paths):
        if i in used:
            continue
        if compact_key(path[-1]) == key:
            used.add(i)
            return list(path)
    return None


def repair_heading_paths(
    headings: list[tuple[int, str]],
    toc_paths: list[list[str]] | None = None,
) -> list[list[str]]:
    """Map (raw_level, title) in document order to repaired heading_path values.

    Order: TOC-guided match, then numbering heuristics, then short topical H1s
    under the nearest 第N章. Duplicate titles reuse the most recent path.
    """
    toc_paths = toc_paths or []
    used_toc: set[int] = set()
    stack: list[tuple[int, str]] = []
    recent_by_key: dict[str, list[str]] = {}
    out: list[list[str]] = []
    nearest_chapter_path: list[str] = []

    for raw_level, title in headings:
        title_n = display_title(title)
        key = compact_key(title_n)

        toc_path = _path_from_toc(title_n, toc_paths, used_toc)
        if toc_path is not None:
            path = toc_path
            level = len(path)
            stack = [(i + 1, t) for i, t in enumerate(path)]
            if looks_like_chapter(title_n):
                nearest_chapter_path = path
            recent_by_key[key] = path
            out.append(path)
            continue

        if key in recent_by_key and is_short_topical(title_n, raw_level):
            path = recent_by_key[key]
            out.append(path)
            continue

        if is_short_topical(title_n, raw_level) and nearest_chapter_path:
            parent_level = len(nearest_chapter_path)
            path = nearest_chapter_path + [title_n]
            stack = [(i + 1, t) for i, t in enumerate(nearest_chapter_path)]
            stack.append((parent_level + 1, title_n))
            recent_by_key[key] = path
            out.append(path)
            continue

        # Flattened magazine-quiz internals (## 问卷项目 after ## 心理测试　…) nest
        # under that quiz parent. Bare glossary ### 心理测试 does not swallow siblings.
        quiz_idx = next((i for i, (_, t) in enumerate(stack) if is_magazine_quiz_heading(t)), None)
        if (
            quiz_idx is not None
            and "心理测试" not in title_n
            and not looks_like_chapter(title_n)
            and not looks_like_front_or_back_matter(title_n)
        ):
            stack = stack[: quiz_idx + 1]
            path = [t for _, t in stack] + [title_n]
            stack.append((stack[-1][0] + 1, title_n))
            recent_by_key[key] = path
            out.append(path)
            continue

        level = structural_level(title_n, raw_level)
        while stack and stack[-1][0] >= level:
            stack.pop()
        path = [t for _, t in stack] + [title_n]
        stack.append((level, title_n))
        if looks_like_chapter(title_n):
            nearest_chapter_path = path
        recent_by_key[key] = path
        out.append(path)

    return out
