"""
refinement.py
Tier 3 — LLM implied-skill detection from candidate raw_text.
Only promotes skills still missing after keyword + embedding tiers.
"""

from __future__ import annotations

import json
import logging
import re

from openai import AsyncOpenAI

from config import settings
from engine.models import RefinementAdjustment

logger = logging.getLogger("match_score_api.refinement")


IMPLIED_SKILLS_SYSTEM_PROMPT = """
You are a technical recruiter AI. Given a candidate's resume raw text and lists of JD skills
that are NOT yet matched, identify which missing skills are clearly evidenced in the resume text
through synonyms, abbreviations, or project descriptions (implied skills).

Rules:
1. Return ONLY skills that appear in the provided missing_primary or missing_good_to_have lists
   (use the exact string from those lists).
2. Evidence must be explicit in candidate_raw_text (projects, roles, tools). No guessing.
3. Never match "Java" to "JavaScript" — they are distinct.
4. Do not infer "React" unless frontend/UI framework use is clearly described.
5. "Design Patterns" requires explicit "design pattern(s)" or software pattern terminology —
   NOT graphic design, Figma, UI layout, or marketing design alone.
6. "OOP concepts" requires explicit OOP, object-oriented, or OOD wording — NOT JavaScript alone.
7. If evidence is weak or absent, omit the skill.
8. Return ONLY valid JSON. No markdown fences.

Required JSON schema:
{
  "implied_primary_matches": [],
  "implied_good_to_have_matches": [],
  "reasoning_first_thought": "One short sentence on what implied evidence was found."
}
""".strip()


class RefinementEngine:
    """Detects implied skill matches in candidate raw_text via structured LLM output."""

    def __init__(
        self,
        openai_api_key: str,
        model: str = settings.OPENAI_LLM_MODEL,
    ) -> None:
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model

    async def detect_implied_skills(
        self,
        candidate_raw_text: str,
        missing_primary: list[str],
        missing_good_to_have: list[str],
    ) -> RefinementAdjustment:
        """
        Scan candidate raw_text for implied evidence of still-missing JD skills.
        """
        if not missing_primary and not missing_good_to_have:
            return RefinementAdjustment()

        cleaned_text = (candidate_raw_text or "").strip()
        if not cleaned_text:
            logger.info("Tier 3 skipped: candidate raw_text is empty.")
            return RefinementAdjustment(
                reasoning="No candidate raw text available for implied skill detection.",
            )

        max_chars = settings.llm_resume_raw_text_max_chars
        if len(cleaned_text) > max_chars:
            cleaned_text = cleaned_text[:max_chars]
            logger.debug("Truncated candidate raw_text to %d chars for Tier 3.", max_chars)

        payload = {
            "candidate_raw_text": cleaned_text,
            "missing_primary": missing_primary,
            "missing_good_to_have": missing_good_to_have,
        }

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": IMPLIED_SKILLS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
            )

            raw_content = response.choices[0].message.content or "{}"
            return self._parse_response(
                raw_json=raw_content,
                missing_primary=missing_primary,
                missing_good_to_have=missing_good_to_have,
            )

        except Exception as exc:
            logger.exception("Tier 3 implied skill detection failed: %s", str(exc))
            return RefinementAdjustment(
                reasoning="Implied skill detection failed; using Tier 1 and Tier 2 matches only.",
            )

    @staticmethod
    def _normalize_skill_key(skill: str) -> str:
        return re.sub(r"[\s\.\-_/]+", "", skill.lower().strip())

    @classmethod
    def _filter_to_allowed(
        cls,
        llm_skills: list,
        allowed_skills: list[str],
    ) -> list[str]:
        """Map LLM output back to canonical JD skill strings."""
        allowed_map = {cls._normalize_skill_key(s): s for s in allowed_skills}
        matched: list[str] = []
        seen: set[str] = set()

        for item in llm_skills or []:
            if not isinstance(item, str):
                continue
            canonical = allowed_map.get(cls._normalize_skill_key(item))
            if canonical and canonical not in seen:
                matched.append(canonical)
                seen.add(canonical)

        return matched

    @classmethod
    def _parse_response(
        cls,
        raw_json: str,
        missing_primary: list[str],
        missing_good_to_have: list[str],
    ) -> RefinementAdjustment:
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning("Tier 3 returned invalid JSON.")
            return RefinementAdjustment(
                reasoning="Invalid JSON from implied skill model.",
            )

        implied_primary = cls._filter_to_allowed(
            data.get("implied_primary_matches", []),
            missing_primary,
        )
        implied_good = cls._filter_to_allowed(
            data.get("implied_good_to_have_matches", []),
            missing_good_to_have,
        )

        reasoning = str(data.get("reasoning_first_thought", ""))

        pairs: list[dict[str, str]] = []
        for skill in implied_primary:
            pairs.append({"jd_skill": skill, "source": "implied_from_raw_text", "tier": "3"})
        for skill in implied_good:
            pairs.append({"jd_skill": skill, "source": "implied_from_raw_text", "tier": "3"})

        if implied_primary or implied_good:
            logger.info(
                "Tier 3 implied matches | primary=%s | good_to_have=%s",
                implied_primary,
                implied_good,
            )

        return RefinementAdjustment(
            implied_primary_matches=implied_primary,
            implied_good_to_have_matches=implied_good,
            matched_pairs=pairs,
            reasoning=reasoning,
        )
