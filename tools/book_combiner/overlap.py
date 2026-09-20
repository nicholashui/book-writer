"""In-process MinHash overlap index. No datasketch; hashes via hashlib, not the builtin."""

from __future__ import annotations

import hashlib
import re
import struct
import unicodedata
from collections import Counter
from itertools import combinations

from book_combiner.models import Section

NUM_PERM = 128
SEED = 0xC0FFEE
SHINGLE_N = 5
MIN_LEAF_CHARS = 40
EXACTISH_JACCARD = 0.90
OVERLAP_JACCARD = 0.55
MAX_HASH = 0xFFFFFFFF

_EMPHASIS_RE = re.compile(r"[*_`]+")
_KEEP_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """NFKC, strip markdown emphasis, then keep \\w and CJK only."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _EMPHASIS_RE.sub("", normalized)
    return _KEEP_RE.sub("", normalized)


def char_shingles(normalized: str, n: int = SHINGLE_N) -> set[str]:
    if len(normalized) < n:
        return set()
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def _u32(digest: bytes) -> int:
    return struct.unpack(">I", digest[:4])[0]


def _perm_coeffs() -> list[tuple[int, int]]:
    coeffs: list[tuple[int, int]] = []
    for i in range(NUM_PERM):
        digest = hashlib.sha256(struct.pack(">II", SEED, i)).digest()
        a = _u32(digest[0:4]) | 1
        b = _u32(digest[4:8])
        coeffs.append((a, b))
    return coeffs


_COEFFS = _perm_coeffs()


def _shingle_hash(shingle: str) -> int:
    blob = struct.pack(">I", SEED) + shingle.encode("utf-8")
    return _u32(hashlib.sha256(blob).digest())


def minhash_signature(text: str) -> tuple[int, ...] | None:
    """128×32-bit MinHash of 5-gram shingles, or None if the leaf is too short."""
    if len(text) < MIN_LEAF_CHARS:
        return None
    shingles = char_shingles(normalize_text(text))
    if not shingles:
        return None
    sig = [MAX_HASH] * NUM_PERM
    for shingle in shingles:
        x = _shingle_hash(shingle)
        for i, (a, b) in enumerate(_COEFFS):
            h = (a * x + b) & MAX_HASH
            if h < sig[i]:
                sig[i] = h
    return tuple(sig)


def estimated_jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    matches = sum(1 for x, y in zip(sig_a, sig_b, strict=True) if x == y)
    return matches / NUM_PERM


def compute_overlap(sections: list[Section]) -> dict:
    """Pairwise MinHash; JSON-ready pairs/components plus duplicate_of."""
    parent = {s.id: s.id for s in sections}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    signatures = {s.id: minhash_signature(s.text) for s in sections}
    pairs: list[dict] = []
    for left, right in combinations(sections, 2):
        hash_equal = left.content_hash == right.content_hash
        sa, sb = signatures[left.id], signatures[right.id]
        jac = estimated_jaccard(sa, sb) if sa is not None and sb is not None else 0.0
        if not hash_equal and jac < OVERLAP_JACCARD:
            continue
        a_id, b_id = tuple(sorted((left.id, right.id)))
        exactish = hash_equal or jac >= EXACTISH_JACCARD
        pairs.append(
            {
                "a": a_id,
                "b": b_id,
                "jaccard": jac,
                "content_hash_equal": hash_equal,
                "relation": "exactish" if exactish else "overlap",
            }
        )
        if exactish:
            union(left.id, right.id)

    by_root: dict[str, list[Section]] = {}
    for sec in sections:
        by_root.setdefault(find(sec.id), []).append(sec)

    duplicate_of: dict[str, str] = {}
    components: list[dict] = []
    for members in by_root.values():
        if len(members) < 2:
            continue
        canonical = min(members, key=lambda s: (-s.char_count, s.id))
        components.append({"canonical": canonical.id, "members": sorted(s.id for s in members)})
        for sec in members:
            if sec.id != canonical.id:
                duplicate_of[sec.id] = canonical.id

    stem_counts: Counter[tuple[str, str]] = Counter()
    by_id = {s.id: s for s in sections}
    for pair in pairs:
        if pair["relation"] != "exactish":
            continue
        sa = by_id[pair["a"]].source_stem
        sb = by_id[pair["b"]].source_stem
        if sa != sb:
            stem_counts[tuple(sorted((sa, sb)))] += 1
    if stem_counts:
        stems, n_pairs = max(stem_counts.items(), key=lambda kv: (kv[1], kv[0]))
    else:
        stems, n_pairs = None, sum(1 for p in pairs if p["relation"] == "exactish")

    return {
        "num_perm": NUM_PERM,
        "seed": SEED,
        "shingle_size": SHINGLE_N,
        "min_leaf_chars": MIN_LEAF_CHARS,
        "thresholds": {"exactish": EXACTISH_JACCARD, "overlap": OVERLAP_JACCARD},
        "pairs": pairs,
        "components": components,
        "duplicate_of": duplicate_of,
        "dominant_exactish_stems": list(stems) if stems else None,
        "dominant_exactish_count": n_pairs,
    }
