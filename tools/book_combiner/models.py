"""Dataclasses for extract records and LLM cache keys; Pydantic for LLM JSON."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, field_validator


@dataclass
class Section:
    id: str
    source_stem: str
    source_book: str
    heading_path: list[str]
    level: int
    start_line: int
    end_line: int
    text: str
    kind: str
    char_count: int
    content_hash: str
    duplicate_of: str | None


@dataclass
class CacheKey:
    stage: str
    model: str
    prompt_hash: str
    params: dict
    input_hashes: list[str]


@dataclass
class CacheRecord:
    key: CacheKey
    response_text: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    created_at: str


class OutlineTitle(BaseModel):
    """Book title. yue_hint is the only Cantonese field; sections have none."""

    model_config = ConfigDict(extra="ignore")

    en: str
    zh: str
    yue_hint: str


class OutlineSection(BaseModel):
    """Leaf node. No title_yue — Yue headings come from translate artifacts later."""

    model_config = ConfigDict(extra="ignore")

    id: str
    title_en: str
    title_zh: str


class OutlineChapter(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title_en: str
    title_zh: str
    sections: list[OutlineSection]


class OutlinePart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title_en: str
    title_zh: str
    chapters: list[OutlineChapter]


class Outline(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: OutlineTitle
    parts: list[OutlinePart]


class AssignmentItem(BaseModel):
    """One heading row → primary outline node, optional secondary for a later cross-ref."""

    model_config = ConfigDict(extra="ignore")

    section_id: str
    primary_node_id: str
    secondary_node_id: str | None = None

    @field_validator("secondary_node_id", mode="before")
    @classmethod
    def _empty_secondary_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value


class AssignmentPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assignments: list[AssignmentItem]


class ConflictClaim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    text: str


class Conflict(BaseModel):
    """Inline dual-claim plus Appendix E row. id must appear in inline_tag."""

    model_config = ConfigDict(extra="ignore")

    id: str
    node_id: str
    claim_a: ConflictClaim
    claim_b: ConflictClaim
    equivalent_hint: bool
    inline_tag: str
