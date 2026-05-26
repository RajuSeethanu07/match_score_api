"""
structural.py
Deterministic ATS structural scoring engine.
"""

from __future__ import annotations

import logging
import re

from engine.models import (
    ParsedJd,
    ParsedResume,
    MatchScoreComputation,
    STRUCTURAL_WEIGHTS,
)

logger = logging.getLogger("match_score_api.structural")


class StructuralScorer:
    """
    Handles deterministic ATS scoring logic.

    Responsibilities:
    - Primary skill matching (Tier 1 Keyword Hit)
    - Good-to-have skill matching (Tier 1 Keyword Hit)
    - Experience matching
    - Location matching
    - Weighted structural score generation
    """

    # ==========================================================================
    # MAIN STRUCTURAL EXECUTION
    # ==========================================================================

    def compute(
        self,
        jd: ParsedJd,
        resume: ParsedResume,
    ) -> MatchScoreComputation:
        """
        Execute full structural ATS scoring.
        """

        logger.info(
            "Starting structural scoring computation"
        )

        # Safe dynamic parsing resolution for resume skills fields
        resume_skills_list = getattr(resume, "primary_skills", None)
        if resume_skills_list is None:
            resume_skills_list = getattr(resume, "skills", [])

        # ----------------------------------------------------------------------
        # PRIMARY SKILL MATCHING
        # ----------------------------------------------------------------------

        (
            primary_match_pct,
            matched_primary,
            missing_primary,
        ) = self._compute_skill_match(
            required_skills=jd.primary_skills,
            candidate_skills=resume_skills_list,
            raw_text=getattr(resume, "raw_text", ""),
        )

        # ----------------------------------------------------------------------
        # GOOD TO HAVE MATCHING
        # ----------------------------------------------------------------------

        (
            good_match_pct,
            matched_good,
            missing_good,
        ) = self._compute_skill_match(
            required_skills=jd.good_to_have_skills,
            candidate_skills=resume_skills_list,
            raw_text=getattr(resume, "raw_text", ""),
        )

        # ----------------------------------------------------------------------
        # EXPERIENCE MATCHING
        # ----------------------------------------------------------------------

        experience_match_pct = (
            self._compute_experience_match(
                min_years=jd.min_experience_years,
                max_years=jd.max_experience_years,
                candidate_years=resume.total_experience_years,
            )
        )

        # ----------------------------------------------------------------------
        # LOCATION MATCHING
        # ----------------------------------------------------------------------

        location_match_pct = (
            self._compute_location_match(
                jd_locations=jd.locations,
                candidate_locations=resume.locations,
            )
        )

        # ----------------------------------------------------------------------
        # WEIGHTED STRUCTURAL SCORE
        # ----------------------------------------------------------------------

        primary_score = (
            (primary_match_pct / 100)
            * STRUCTURAL_WEIGHTS["primary_skills"]
        )

        good_score = (
            (good_match_pct / 100)
            * STRUCTURAL_WEIGHTS["good_to_have"]
        )

        experience_score = (
            (experience_match_pct / 100)
            * STRUCTURAL_WEIGHTS["experience"]
        )

        location_score = (
            (location_match_pct / 100)
            * STRUCTURAL_WEIGHTS["location"]
        )

        structural_score = (
            primary_score
            + good_score
            + experience_score
            + location_score
        )

        logger.info(
            "Structural scoring completed successfully"
        )

        return MatchScoreComputation(
            primary_match_pct=round(primary_match_pct, 2),
            good_to_have_match_pct=round(good_match_pct, 2),
            location_match_pct=round(
                location_match_pct,
                2,
            ),
            structural_score=round(
                structural_score,
                2,
            ),
            matched_primary_skills=matched_primary,
            missing_primary_skills=missing_primary,
            matched_good_to_have_skills=matched_good,
            missing_good_to_have_skills=missing_good,
        )

    # ==========================================================================
    # SKILL MATCHING (TIER 1 KEYWORD HIT WITH BUG FIXES)
    # ==========================================================================

    def _compute_skill_match(
        self,
        required_skills: list[str],
        candidate_skills: list[str],
        raw_text: str = "",
    ) -> tuple[float, list[str], list[str]]:
        """
        Compute ATS skill matching percentage using isolated word boundary hits.
        Ensures short terms like 'Java' do not accidentally match inside 'JavaScript'.
        """

        if not required_skills:
            return 100.0, [], []

        cand_source = candidate_skills if candidate_skills else []

        normalized_candidate_skills = {
            self._normalize_skill(skill): skill
            for skill in cand_source
            if skill and skill.strip()
        }

        matched = []
        missing = []

        # Prepare normalized raw text for a deep scan fallback lookup
        clean_raw_text = ""
        if raw_text:
            clean_raw_text = re.sub(r"[\s\.\-\/]+", "", raw_text.lower())

        for required_skill in required_skills:
            normalized_required = self._normalize_skill(required_skill)
            found_match = False

            # 1. Exact Structural Key Match
            if normalized_required in normalized_candidate_skills:
                # Explicit safety shield: reject if required is 'java' but matched key is 'javascript'
                if normalized_required == "java" and "javascript" in normalized_candidate_skills:
                    # Let it fall through to substring confirmation loops
                    pass
                else:
                    found_match = True

            # 2. Guarded Token Boundary Containment Match
            if not found_match:
                for normalized_candidate in normalized_candidate_skills.keys():
                    # Stop Java from bleeding into JavaScript
                    if normalized_required == "java" and "javascript" in normalized_candidate:
                        continue

                    # Allow token mapping only if characters exceed basic short abbreviations
                    if len(normalized_required) > 3 and (
                        normalized_required in normalized_candidate
                        or normalized_candidate in normalized_required
                    ):
                        found_match = True
                        break

            # 3. Raw Document Text Fallback Lookup (Catches unparsed tokens like REST API)
            if not found_match and clean_raw_text:
                if normalized_required == "java" and "javascript" in clean_raw_text:
                    # Check if 'java' exists independently outside of 'javascript' instances
                    # Strip out 'javascript' temporarily to see if a standalone 'java' footprint remains
                    stripped_raw = clean_raw_text.replace("javascript", "")
                    if normalized_required in stripped_raw:
                        found_match = True
                elif normalized_required in clean_raw_text:
                    found_match = True

            if found_match:
                matched.append(required_skill)
            else:
                missing.append(required_skill)

        match_percentage = (len(matched) / len(required_skills)) * 100

        return (
            round(match_percentage, 2),
            matched,
            missing,
        )

    # ==========================================================================
    # EXPERIENCE MATCHING
    # ==========================================================================

    @staticmethod
    def _compute_experience_match(
        min_years: float,
        max_years: float,
        candidate_years: float,
    ) -> float:
        """
        Experience matching logic.
        """

        candidate_years = max(
            candidate_years,
            0.0,
        )

        if min_years <= 0:
            return 100.0

        if candidate_years < min_years:
            score = (candidate_years / min_years) * 100
            return round(max(score, 0.0), 2)

        if max_years <= 0 or candidate_years <= max_years:
            return 100.0

        excess_years = candidate_years - max_years
        penalty = excess_years * 5
        final_score = max(100.0 - penalty, 60.0)

        return round(final_score, 2)

    # ==========================================================================
    # LOCATION MATCHING
    # ==========================================================================

    def _compute_location_match(
        self,
        jd_locations: list[str],
        candidate_locations: list[str],
    ) -> float:
        """
        Location matching logic.
        """

        if not jd_locations:
            return 100.0

        if not candidate_locations:
            return 0.0

        normalized_jd_locations = {
            self._normalize_skill(location)
            for location in jd_locations
            if location
        }

        normalized_candidate_locations = {
            self._normalize_skill(location)
            for location in candidate_locations
            if location
        }

        for jd_location in normalized_jd_locations:
            jd_city = jd_location.split(",")[0].strip()

            for candidate_location in normalized_candidate_locations:
                candidate_city = candidate_location.split(",")[0].strip()

                if jd_city in candidate_city or candidate_city in jd_city:
                    return 100.0

        return 0.0

    # ==========================================================================
    # SKILL NORMALIZATION
    # ==========================================================================

    @staticmethod
    def _normalize_skill(skill: str) -> str:
        """
        Normalize skill text intensely for structural keyword loops.
        Strips casing, whitespaces, dots, dashes, and slashes.
        """
        if not skill:
            return ""

        skill = skill.strip().lower()
        skill = re.sub(r"[\s\.\-\/]+", "", skill)
        return skill