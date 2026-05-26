"""
skill_evidence.py
Raw-text evidence checks for ambiguous / concept JD skills.
Prevents Tier 2 embedding false positives (e.g. Figma -> Design Patterns).
"""

from __future__ import annotations

import re

# Skills whose names are vague — embedding-only match is not enough.
_SOFT_SKILL_HINTS = (
    "pattern",
    "concept",
    "principle",
    "methodology",
    "paradigm",
    "architecture",  # protects solution architecture vs marketing layouts
)


def normalize_compact(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[\s\.\-\/]+", "", text.strip().lower())


def requires_raw_text_evidence(skill: str) -> bool:
    """Concept-style JD skills must appear in candidate raw_text, not embeddings alone."""
    lowered = skill.lower().strip()
    if not lowered:
        return False
    return any(hint in lowered for hint in _SOFT_SKILL_HINTS)


def has_raw_text_evidence(skill: str, raw_text: str) -> bool:
    """
    Return True only when resume raw text clearly supports the JD skill.
    Uses phrase / word-boundary checks — not loose embedding similarity.
    """
    if not raw_text or not raw_text.strip():
        return False

    compact = normalize_compact(raw_text)
    norm_skill = normalize_compact(skill)
    lowered = raw_text.lower()

    # 1. Direct compact match optimization
    if norm_skill and norm_skill in compact:
        return True

    # 2. Design Patterns — require explicit pattern wording, not UI/graphic "design" alone
    if "design" in norm_skill and "pattern" in norm_skill:
        if re.search(r"design\s+patterns?", lowered):
            return True
        if "designpattern" in compact:
            return True
        return False

    # 3. OOP / object-oriented — word boundaries; avoid matching "loop" or random substrings
    if "oop" in norm_skill or ("object" in norm_skill and "concept" in norm_skill):
        if re.search(r"\boop\b", lowered):
            return True
        if re.search(r"object[-\s]?oriented", lowered):
            return True
        if re.search(r"\bood\b", lowered):
            return True
        if "objectoriented" in compact or "objectorientedprogramming" in compact:
            return True
        return False

    # 4. Other soft skills/concepts: require normalized skill phrase in compact text
    if requires_raw_text_evidence(skill):
        return norm_skill in compact

    # 5. SAFE FALLBACK: If it doesn't require explicit conceptual evidence (e.g., AWS, Python, Docker), 
    # we return True to allow it to pass through to normal Tier 2 embedding similarity evaluation.
    return True