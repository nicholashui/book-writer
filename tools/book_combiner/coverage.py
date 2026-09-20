"""Frozen coverage-token extractors for union-of-facts merge checks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from book_combiner.models import Section

NUMBERED_LINE = re.compile(
    r"^\s*(?:\d+[\.、]|[一二三四五六七八九十]+、)\s*(.+)$",
    re.M,
)
# Running-prose 出汗 catalogues. Do NOT findall `[\u4e00-\u9fff]{2,8}出汗` on the
# raw span: 和 is CJK, so L1196 「味觉性出汗和运动性出汗」 would yield 和运动性出汗.
CATALOGUE_SPLIT = re.compile(r"[、，;；和与及]")  # also strip a leading 以及 after split
SWEAT_COMPOUND = re.compile(r"^[\u4e00-\u9fff]{2,8}出汗$")
LEADING_CONJ = re.compile(r"^(?:和|与|及|以及|、)+")
_TRAIL_PUNCT = re.compile(r"[\s。．.；;，,、：:！!？?]+$")

BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
# Extract-stage figure captions; "Figure" is not a source fact.
FIGURE_PLACEHOLDER = re.compile(
    r"\*\[Figure omitted from source conversion:.*?\]\*",
    re.S,
)
# TOC-style 10.2 / 10.2.1, not durations like 0.04.
OUTLINE_NUMBER = re.compile(r"^[1-9]\d{0,2}(?:\.\d{1,3})+$")

NUMBER_PATTERNS = [
    re.compile(r"\d+%"),
    re.compile(r"\d+\s*/\s*\d+"),
    re.compile(r"(?<![.\d])0\.\d+"),
    re.compile(r"(?<![.\d])\d{2,}(?![.\d])"),
    re.compile(r"\d+(?:秒|分钟|小时|毫秒|次|种|个|条|项|章|节)"),
]

NAME_PATTERNS = [
    re.compile(r"\b(?:METT|FACS|AU\d+)\b"),
    re.compile(r"[A-Z][A-Za-z]+(?:[ -][A-Z][A-Za-z]+)*"),
    re.compile(r"[\u4e00-\u9fff]{1,4}·[\u4e00-\u9fff]{1,4}"),
]

MUSCLE_OR_AU = [
    re.compile(r"[\u4e00-\u9fff]{1,4}肌"),
    re.compile(r"AU\d+"),
]


@dataclass
class CoverageTokens:
    numbered_items: list[str] = field(default_factory=list)
    bold_terms: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    muscle_or_au: list[str] = field(default_factory=list)


def sweat_compounds(text: str) -> list[str]:
    """Pieces of a 顿号/conjunction catalogue, then `{2,8}出汗` on each piece."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in CATALOGUE_SPLIT.split(text):
        piece = LEADING_CONJ.sub("", raw.strip())
        piece = _TRAIL_PUNCT.sub("", piece)
        if SWEAT_COMPOUND.fullmatch(piece) and piece not in seen:
            seen.add(piece)
            out.append(piece)
    return out


_NAME_TRAIL = re.compile(r"[提认说的在是了与和及]+$")
_MUSCLE_LEAD = frozenset("与和及的将把被块组造成两令这发只由容")
_MUSCLE_JUNK = re.compile(r"[块组造成两令这发只由的主导]")
_MUSCLE_TOKEN = re.compile(r"[\u4e00-\u9fff]{1,4}肌")
_DROP_TOKENS = frozenset({"参考书目", "索引", "小结", "Contents"})
_NUMBERED_MAX = 12


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        token = item.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _clean_cjk_name(token: str) -> str:
    if "·" not in token:
        return token
    return _NAME_TRAIL.sub("", token) or token


def _clean_muscle(token: str) -> str:
    if token.startswith("AU"):
        return token
    for i in range(len(token)):
        cand = token[i:]
        if (
            _MUSCLE_TOKEN.fullmatch(cand)
            and cand[0] not in _MUSCLE_LEAD
            and not _MUSCLE_JUNK.search(cand)
        ):
            return cand
    return ""


def _sweat_spans(text: str) -> list[str]:
    """Catalogue split on the full span, each paragraph, and each 。 sentence."""
    spans = [text]
    spans.extend(re.split(r"\n\s*\n", text))
    spans.extend(re.split(r"[。．]", text))
    found: list[str] = []
    for span in spans:
        found.extend(sweat_compounds(span))
    return found


def extract_tokens(text: str) -> CoverageTokens:
    text = FIGURE_PLACEHOLDER.sub("", text)
    numbered = [
        m.strip()
        for m in NUMBERED_LINE.findall(text)
        if 0 < len(m.strip()) <= _NUMBERED_MAX
    ]
    for payload in list(numbered):
        numbered.extend(_sweat_spans(payload))
    numbered.extend(_sweat_spans(text))
    bold = [m.strip() for m in BOLD_RE.findall(text) if len(m.strip()) >= 2]
    names: list[str] = []
    for pat in NAME_PATTERNS:
        names.extend(_clean_cjk_name(m) if isinstance(m, str) else m for m in pat.findall(text))
    numbers: list[str] = []
    for pat in NUMBER_PATTERNS:
        numbers.extend(
            n for n in pat.findall(text) if not OUTLINE_NUMBER.fullmatch(n)
        )
    muscle: list[str] = []
    for pat in MUSCLE_OR_AU:
        muscle.extend(_clean_muscle(m) for m in pat.findall(text))
    return CoverageTokens(
        numbered_items=_dedupe(numbered),
        bold_terms=_dedupe(bold),
        names=_dedupe(names),
        numbers=_dedupe(numbers),
        muscle_or_au=_dedupe(muscle),
    )


def flatten_tokens(tokens: CoverageTokens) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in (
        tokens.numbered_items,
        tokens.bold_terms,
        tokens.names,
        tokens.numbers,
        tokens.muscle_or_au,
    ):
        for raw in group:
            token = unicodedata.normalize("NFKC", raw.strip()).strip("：:;；。、,.")
            if not token or token in seen:
                continue
            if OUTLINE_NUMBER.fullmatch(token) or token in _DROP_TOKENS:
                continue
            seen.add(token)
            out.append(token)
    return out


def tokens_from_text(text: str) -> list[str]:
    return flatten_tokens(extract_tokens(text))


def required_tokens(sections: list[Section]) -> list[str]:
    """Tokens from unique leaves (`duplicate_of is None`) only."""
    out: list[str] = []
    seen: set[str] = set()
    for section in sections:
        if section.duplicate_of is not None:
            continue
        for token in tokens_from_text(section.text):
            if token not in seen:
                seen.add(token)
                out.append(token)
    return out


def missing_tokens(output: str, required: list[str]) -> list[str]:
    from book_combiner.translate import traditionalize

    haystack = unicodedata.normalize("NFKC", output)
    hay_t = traditionalize(haystack)
    missing: list[str] = []
    for token in required:
        needle = unicodedata.normalize("NFKC", token)
        if needle in haystack or traditionalize(needle) in hay_t:
            continue
        missing.append(token)
    return missing
