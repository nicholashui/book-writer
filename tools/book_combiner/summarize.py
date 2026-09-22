"""Condense compiled EN/HK books to ~5% length with a shared outline spine."""

from __future__ import annotations

import re
from pathlib import Path

from book_combiner.llm.base import LLMClient, completed_text
from book_combiner.merge import atomic_write_text
from book_combiner.models import Outline, OutlinePart
from book_combiner.outline import sha256_text
from book_combiner.translate import HEADING_RE, strip_yaml_front_matter, traditionalize

RATIO = 0.05
CHUNK_CHARS = 20_000
SUMMARIZE_TEMPERATURE = 0.2
STAGE_EN = "summarize-en"
STAGE_YUE = "summarize-yue"
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SUMMARIZE_PROMPT_PATH = _PROMPTS_DIR / "summarize.txt"

H2_SPLIT = re.compile(r"(?=^## )", re.M)
YAML_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n?", re.S)


def split_h2_sections(text: str) -> list[tuple[str, str]]:
    body = strip_yaml_front_matter(text)
    chunks = [c for c in H2_SPLIT.split(body) if c.strip()]
    out: list[tuple[str, str]] = []
    for chunk in chunks:
        lines = chunk.splitlines()
        first = lines[0] if lines else ""
        match = re.match(r"^##\s+(.+?)\s*$", first)
        if not match:
            continue
        title = match.group(1).strip()
        rest = "\n".join(lines[1:]).strip()
        out.append((title, rest))
    return out


def drop_toc(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    skip = {"contents", "目錄", "目录"}
    return [(t, b) for t, b in sections if t.strip().casefold() not in skip]


def pack_chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    paras = re.split(r"\n\s*\n", text)
    packs: list[str] = []
    buf: list[str] = []
    n = 0
    for para in paras:
        extra = len(para) + 2
        if buf and n + extra > size:
            packs.append("\n\n".join(buf))
            buf = [para]
            n = len(para)
        else:
            buf.append(para)
            n += extra
    if buf:
        packs.append("\n\n".join(buf))
    return packs


def _max_tokens_for(source_chars: int) -> int:
    target_chars = max(400, int(source_chars * RATIO))
    return max(256, min(900, int(target_chars / 3.0) + 80))


def summarize_chunk(
    *,
    client: LLMClient,
    text: str,
    language: str,
    stage: str,
) -> str:
    template = SUMMARIZE_PROMPT_PATH.read_text(encoding="utf-8")
    system = template.format(ratio_pct=int(RATIO * 100), language=language)
    user = "SOURCE:\n" + text
    record = client.complete(
        stage=stage,
        prompt_hash=sha256_text(system),
        input_hashes=[sha256_text(user)],
        user=user,
        system=system,
        temperature=SUMMARIZE_TEMPERATURE,
        max_tokens=_max_tokens_for(len(text)),
        max_input_chars=max(CHUNK_CHARS, len(user)),
        json_mode=False,
    )
    return completed_text(record)


def summarize_body(
    *,
    client: LLMClient,
    text: str,
    language: str,
    stage: str,
) -> str:
    if len(text) < 800:
        return text.strip()
    pieces = [summarize_chunk(client=client, text=p, language=language, stage=stage) for p in pack_chunks(text)]
    return "\n\n".join(p for p in pieces if p.strip())


def strip_inner_headings(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if HEADING_RE.match(line.strip()):
            continue
        lines.append(line)
    body = "\n".join(lines)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


def distribute_under_chapters(body: str, chapter_titles: list[str]) -> str:
    if not chapter_titles:
        return body.strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n", strip_inner_headings(body)) if p.strip()]
    n = len(chapter_titles)
    if not paras:
        return "\n\n".join(f"### {title}\n" for title in chapter_titles)
    blocks: list[str] = []
    for i, title in enumerate(chapter_titles):
        start = i * len(paras) // n
        end = (i + 1) * len(paras) // n
        chunk = "\n\n".join(paras[start:end])
        if chunk:
            blocks.append(f"### {title}\n\n{chunk}")
        else:
            blocks.append(f"### {title}\n")
    return "\n\n".join(blocks)


def part_chapter_titles(part: OutlinePart, lang: str) -> list[str]:
    titles: list[str] = []
    for chapter in part.chapters:
        titles.append(chapter.title_en if lang == "en" else traditionalize(chapter.title_zh))
    return titles


def appendix_titles(lang: str) -> list[tuple[str, str]]:
    if lang == "en":
        return [
            ("Appendix A — Sources", "a"),
            ("Appendix B — Quizzes", "b"),
            ("Appendix D — Bibliography", "d"),
        ]
    return [
        ("附錄 A · 來源", "a"),
        ("附錄 B · 心理測試", "b"),
        ("附錄 D · 參考書目", "d"),
    ]


def rebuild_toc(part_titles: list[str], lang: str) -> str:
    toc_title = "Contents" if lang == "en" else "目錄"
    lines = [f"- {t}" for t in part_titles]
    return f"## {toc_title}\n\n" + "\n".join(lines)


def front_matter(*, lang: str, source_name: str) -> str:
    language = "en" if lang == "en" else "yue-Hant-HK"
    register = "null" if lang == "en" else "hk-written-cantonese"
    return (
        "---\n"
        f"topic: facial-expression\n"
        f"kind: summary\n"
        f"language: {language}\n"
        f"register: {register}\n"
        f"source: {source_name}\n"
        f"ratio: {RATIO}\n"
        "---\n\n"
    )


def run_summarize_pair(
    *,
    en_path: Path,
    hk_path: Path,
    outline: Outline,
    client: LLMClient,
    out_en: Path,
    out_hk: Path,
) -> None:
    en_sections = drop_toc(split_h2_sections(en_path.read_text(encoding="utf-8")))
    hk_sections = drop_toc(split_h2_sections(hk_path.read_text(encoding="utf-8")))
    if len(en_sections) != len(hk_sections):
        raise ValueError(
            f"H2 count mismatch EN={len(en_sections)} HK={len(hk_sections)}"
        )
    n_parts = len(outline.parts)
    if len(en_sections) < n_parts:
        raise ValueError(f"compiled book has {len(en_sections)} H2 parts, outline has {n_parts}")

    en_bodies: list[str] = []
    hk_bodies: list[str] = []
    part_titles_en: list[str] = []
    part_titles_hk: list[str] = []

    for i, part in enumerate(outline.parts):
        _en_title, en_body = en_sections[i]
        _hk_title, hk_body = hk_sections[i]
        title_en = part.title_en
        title_hk = traditionalize(part.title_zh)
        part_titles_en.append(title_en)
        part_titles_hk.append(title_hk)
        print(f"[summarize-en] {part.id} {len(en_body)} chars")
        en_sum = summarize_body(
            client=client, text=en_body, language="English", stage=STAGE_EN
        )
        print(f"[summarize-yue] {part.id} {len(hk_body)} chars")
        hk_sum = summarize_body(
            client=client,
            text=hk_body,
            language="Hong Kong written Cantonese (書面粵語)",
            stage=STAGE_YUE,
        )
        ch_en = part_chapter_titles(part, "en")
        ch_hk = part_chapter_titles(part, "yue")
        en_block = f"## {title_en}\n\n" + distribute_under_chapters(en_sum, ch_en)
        hk_block = f"## {title_hk}\n\n" + distribute_under_chapters(hk_sum, ch_hk)
        en_bodies.append(en_block)
        hk_bodies.append(hk_block)

    extra_en = en_sections[n_parts:]
    extra_hk = hk_sections[n_parts:]
    for (src_title, src_body), (hk_title, hk_body), (canon_en, _k), (canon_hk, _k2) in zip(
        extra_en, extra_hk, appendix_titles("en"), appendix_titles("yue"), strict=False
    ):
        part_titles_en.append(canon_en)
        part_titles_hk.append(canon_hk)
        if len(src_body) < 2500:
            en_sum, hk_sum = src_body, hk_body
        else:
            print(f"[summarize-en] appendix {canon_en} {len(src_body)} chars")
            en_sum = summarize_body(
                client=client, text=src_body, language="English", stage=STAGE_EN
            )
            print(f"[summarize-yue] appendix {canon_hk} {len(hk_body)} chars")
            hk_sum = summarize_body(
                client=client,
                text=hk_body,
                language="Hong Kong written Cantonese (書面粵語)",
                stage=STAGE_YUE,
            )
        en_bodies.append(f"## {canon_en}\n\n{strip_inner_headings(en_sum)}")
        hk_bodies.append(f"## {canon_hk}\n\n{strip_inner_headings(hk_sum)}")

    book_title_en = outline.title.en or "Facial Expression"
    book_title_hk = traditionalize(outline.title.zh or "面部表情")
    en_md = (
        front_matter(lang="en", source_name=en_path.as_posix())
        + f"# {book_title_en}\n\n"
        + rebuild_toc(part_titles_en, "en")
        + "\n\n"
        + "\n\n".join(en_bodies)
        + "\n"
    )
    hk_md = (
        front_matter(lang="yue", source_name=hk_path.as_posix())
        + f"# {book_title_hk}\n\n"
        + rebuild_toc(part_titles_hk, "yue")
        + "\n\n"
        + "\n\n".join(hk_bodies)
        + "\n"
    )
    atomic_write_text(out_en, en_md)
    atomic_write_text(out_hk, hk_md)
    print(f"[summarize] {out_en.as_posix()} {len(en_md)} chars")
    print(f"[summarize] {out_hk.as_posix()} {len(hk_md)} chars")


def _split_h3(body: str) -> list[tuple[str | None, str]]:
    chunks = re.split(r"(?=^### )", body, flags=re.M)
    out: list[tuple[str | None, str]] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        match = re.match(r"^###\s+(.+?)\s*$", lines[0])
        if match:
            out.append((match.group(1).strip(), "\n".join(lines[1:]).strip()))
        else:
            out.append((None, chunk))
    return out


def tighten_markdown(
    *,
    text: str,
    client: LLMClient,
    language: str,
    stage: str,
    ratio: float,
) -> str:
    """Second-pass compress, keeping the existing heading skeleton."""
    global RATIO
    old = RATIO
    RATIO = ratio
    try:
        yaml_m = YAML_RE.match(text)
        yaml = yaml_m.group(0) if yaml_m else ""
        body = text[len(yaml) :]
        h1 = ""
        rest = body.lstrip()
        if rest.startswith("# ") and not rest.startswith("##"):
            first, _, rest = rest.partition("\n")
            h1 = first.strip()
        if not rest.lstrip().startswith("##"):
            rest = "## Body\n\n" + rest
        sections = split_h2_sections(rest)
        rebuilt: list[str] = []
        if h1:
            rebuilt.append(h1)
        for title, sec_body in sections:
            if title.casefold() in {"contents", "目錄", "目录"}:
                rebuilt.append(f"## {title}\n\n{sec_body.strip()}")
                continue
            pieces = _split_h3(sec_body)
            blocks = [f"## {title}"]
            for ch_title, ch_body in pieces:
                if not ch_body.strip():
                    if ch_title:
                        blocks.append(f"### {ch_title}")
                    continue
                print(f"[{stage}] tighten {ch_title or title} {len(ch_body)} chars")
                compressed = summarize_body(
                    client=client, text=ch_body, language=language, stage=stage
                )
                if ch_title:
                    blocks.append(f"### {ch_title}\n\n{strip_inner_headings(compressed)}")
                else:
                    blocks.append(strip_inner_headings(compressed))
            rebuilt.append("\n\n".join(blocks))
        return yaml + "\n".join(rebuilt).strip() + "\n"
    finally:
        RATIO = old
