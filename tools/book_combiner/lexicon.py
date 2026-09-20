"""Generic heading-keyword lexicon. Topic extras are additive JSON files."""

from __future__ import annotations

import json
from pathlib import Path

LEXICON: dict[str, list[str]] = {
    "eyes": ["眼", "瞳孔", "眨眼", "目光", "视线"],
    "brows": ["眉"],
    "nose": ["鼻"],
    "mouth": ["嘴", "唇", "舌", "牙"],
    "chin": ["下巴", "颏"],
    "smile": ["笑", "微笑"],
    "micro": ["微表情", "微反应"],
    "hands": ["手", "掌", "指"],
    "arms": ["臂", "肩"],
    "legs": ["腿", "脚", "足", "坐姿", "站姿", "走"],
    "voice": ["声", "语速", "音调", "说话"],
    "lie": ["谎", "测谎", "欺骗"],
    "emotion_basic": ["惊讶", "厌恶", "愤怒", "恐惧", "悲伤", "愉悦", "轻蔑"],
    "workplace": ["职场", "面试", "上司", "客户"],
    "quiz": ["心理测试"],
}

_TOPICS_DIR = Path(__file__).resolve().parent / "topics"


def load_lexicon(topic: str, topics_dir: Path | None = None) -> dict[str, list[str]]:
    """Generic LEXICON plus optional `topics/<topic>/lexicon.json`. Missing file → skip."""
    merged: dict[str, list[str]] = {key: list(pats) for key, pats in LEXICON.items()}
    extra_path = (topics_dir or _TOPICS_DIR) / topic / "lexicon.json"
    if not extra_path.is_file():
        return merged
    extra = json.loads(extra_path.read_text(encoding="utf-8"))
    if not extra:
        return merged
    for key, patterns in extra.items():
        bucket = merged.setdefault(key, [])
        for pat in patterns:
            if pat not in bucket:
                bucket.append(pat)
    return merged


def keywords_for(heading_path: list[str], lexicon: dict[str, list[str]] | None = None) -> list[str]:
    """Bucket keys whose patterns hit any heading in the path (LEXICON order)."""
    table = lexicon if lexicon is not None else LEXICON
    haystack = "".join(heading_path)
    return [key for key, pats in table.items() if any(pat in haystack for pat in pats)]
