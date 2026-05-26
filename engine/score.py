from __future__ import annotations
import asyncio
import logging
import re
import unicodedata
from functools import lru_cache

from config import settings
from engine.models import ParsedJd, ParsedResume, STRUCTURAL_WEIGHTS
from engine.structural import StructuralScorer
from engine.semantic import SemanticEngine
from engine.refinement import RefinementEngine
from engine.contextual_semantic import contextual_semantic_matcher
from engine.formatter import build_final_response
from engine.skill_evidence import has_raw_text_evidence, requires_raw_text_evidence

logger = logging.getLogger("match_score_api.scorer")


class MatchScoreEngine:

    def __init__(self) -> None:
        self.structural_engine = StructuralScorer()
        self.semantic_engine = SemanticEngine(openai_api_key=settings.OPENAI_API_KEY)
        self.refinement_engine = RefinementEngine(openai_api_key=settings.OPENAI_API_KEY)
        self.contextual_matcher = contextual_semantic_matcher

    # ---------------- SAFE NORMALIZATION ----------------
    def _normalize_raw_text(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text).lower()
        text = re.sub(r"[\u200b\u200c\u200d]", " ", text)
        return " " + " ".join(text.split()) + " "

    @staticmethod
    @lru_cache(maxsize=5000)
    def _compile_flexible_pattern(skill_lower: str):
        escaped = re.escape(skill_lower).replace(r"\ ", r"[\s\-\.\/_]*")
        if skill_lower and skill_lower[0].isalnum():
            escaped = rf"\b{escaped}"
        if skill_lower and skill_lower[-1].isalnum():
            escaped = rf"{escaped}\b"
        return re.compile(escaped)

    def _verify_raw_text_fallback(self, skill: str, text: str) -> bool:
        skill_lower = skill.strip().lower()
        if not skill_lower:
            return False

        try:
            if self._compile_flexible_pattern(skill_lower).search(text):
                return True
        except Exception as e:
            logger.warning("Regex failed for %s: %s", skill_lower, e)

        comp_skill = re.sub(r"[^a-z0-9]", "", skill_lower)
        comp_text = re.sub(r"[^a-z0-9]", "", text)

        if len(comp_skill) > 3 and comp_skill in comp_text:
            return True

        return False

    # ---------------- SAFE SKILL CHECK ----------------
    def _safe_has_skill(self, lst, skill):
        return lst and skill in lst

    # ---------------- MAIN PIPELINE ----------------
    async def score(
        self,
        jd: ParsedJd,
        resume: ParsedResume,
        existing_jd_embedding=None,
        existing_resume_embedding=None
    ):
        structural = self.structural_engine.compute(jd=jd, resume=resume)

        matched_primary = list(set(structural.matched_primary_skills))
        matched_good = list(set(structural.matched_good_to_have_skills))

        missing_primary = list(structural.missing_primary_skills)
        missing_good = list(structural.missing_good_to_have_skills)

        norm_text = self._normalize_raw_text(getattr(resume, "raw_text", "") or "")

        # ---------------- RAW TEXT FALLBACK ----------------
        for skill in list(missing_primary):
            if self._verify_raw_text_fallback(skill, norm_text):
                if skill not in matched_primary:
                    matched_primary.append(skill)
                missing_primary.remove(skill)

        for skill in list(missing_good):
            if self._verify_raw_text_fallback(skill, norm_text):
                if skill not in matched_good:
                    matched_good.append(skill)
                missing_good.remove(skill)

        # ---------------- TIER 2: SEMANTIC ----------------
        resume_skills = getattr(resume, "primary_skills", []) or []
        resume_skills_list = list(set(str(s).strip() for s in resume_skills if s))

        if resume_skills_list:

            cand_results = await asyncio.gather(
                *[self.semantic_engine.generate_embedding(s) for s in resume_skills_list],
                return_exceptions=True
            )

            cand_vectors = []
            for i, r in enumerate(cand_results):
                if isinstance(r, Exception):
                    logger.warning("Embedding failed for %s: %s", resume_skills_list[i], r)
                elif isinstance(r, list):
                    cand_vectors.append((resume_skills_list[i], r))

            missing_all = list(set(missing_primary + missing_good))

            missing_results = await asyncio.gather(
                *[self.semantic_engine.generate_embedding(s) for s in missing_all],
                return_exceptions=True
            )

            missing_map = {}
            for i, r in enumerate(missing_results):
                if isinstance(r, list):
                    missing_map[missing_all[i]] = r

            for skill in missing_primary[:]:
                vec = missing_map.get(skill)
                if not vec:
                    continue

                sims = [
                    self.semantic_engine.compute_similarity_score(vec, cv) / 100.0
                    for _, cv in cand_vectors
                ]

                if sims and max(sims) > 0:
                    if self._verify_raw_text_fallback(skill, norm_text):
                        matched_primary.append(skill)

            for skill in missing_good[:]:
                vec = missing_map.get(skill)
                if not vec:
                    continue

                sims = [
                    self.semantic_engine.compute_similarity_score(vec, cv) / 100.0
                    for _, cv in cand_vectors
                ]

                if sims and max(sims) > 0:
                    if self._verify_raw_text_fallback(skill, norm_text):
                        matched_good.append(skill)

        # ---------------- TIER 3: LLM ----------------
        still_miss_p = [s for s in (jd.primary_skills or []) if s not in matched_primary]
        still_miss_g = [s for s in (jd.good_to_have_skills or []) if s not in matched_good]

        ctx_score, ctx_matches = await self.contextual_matcher.compute_contextual_match(
            jd=jd, resume=resume
        )

        if settings.llm_implied_skills_enabled and (still_miss_p or still_miss_g):
            ref = await self.refinement_engine.detect_implied_skills(
                norm_text, still_miss_p, still_miss_g
            )

            for s in ref.implied_primary_matches:
                if s not in matched_primary and self._verify_raw_text_fallback(s, norm_text):
                    matched_primary.append(s)

            for s in ref.implied_good_to_have_matches:
                if s not in matched_good and self._verify_raw_text_fallback(s, norm_text):
                    matched_good.append(s)

        # ---------------- SAFE SCORING ----------------
        total_p = len(jd.primary_skills or []) or 1
        total_g = len(jd.good_to_have_skills or []) or 1

        p_score = (len(matched_primary) / total_p) * STRUCTURAL_WEIGHTS["primary_skills"]
        g_score = (len(matched_good) / total_g) * STRUCTURAL_WEIGHTS["good_to_have"]

        req_exp = float(jd.min_experience_years or 0)
        cand_exp = float(resume.total_experience_years or 0)

        exp_pct = 100.0 if cand_exp >= req_exp else (cand_exp / req_exp * 100 if req_exp else 0)
        exp_score = (exp_pct / 100.0) * STRUCTURAL_WEIGHTS["experience"]

        loc_score = (structural.location_match_pct / 100.0) * STRUCTURAL_WEIGHTS["location"]

        overall = min(round(p_score + g_score + exp_score + loc_score, 1), 100.0)

        res = build_final_response(
            overall_score=overall,
            jd=jd,
            resume=resume,
            matched_primary=matched_primary,
            missing_primary=[s for s in (jd.primary_skills or []) if s not in matched_primary],
            matched_good=matched_good,
            missing_good=[s for s in (jd.good_to_have_skills or []) if s not in matched_good],
            primary_score=p_score,
            good_score=g_score,
            experience_score=exp_score,
            location_score=loc_score,
            experience_match_pct=exp_pct,
            location_match_pct=structural.location_match_pct
        )

        if hasattr(res, "experienceMatch") and isinstance(res.experienceMatch, dict):
            res.experienceMatch["candidateExperience"] = f"{cand_exp} years" if cand_exp else "Not specified"

        res._internal_contextual_score = ctx_score
        res._internal_contextual_matches = ctx_matches

        jd_emb = existing_jd_embedding or await self.semantic_engine.generate_embedding(jd.raw_text)
        res_emb = existing_resume_embedding or await self.semantic_engine.generate_embedding(resume.raw_text)

        return res, jd_emb, res_emb