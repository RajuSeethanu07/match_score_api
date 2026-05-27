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

logger = logging.getLogger("match_score_api.scorer")


class MatchScoreEngine:

    def __init__(self, db_client=None) -> None:
        self.structural_engine = StructuralScorer()
        # Explicitly passing through the MongoDB client instance to enable persistent lookup vectors
        self.semantic_engine = SemanticEngine(openai_api_key=settings.OPENAI_API_KEY, db_client=db_client)
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
        """
        Compiles a robust regex pattern handling word boundaries safely,
        even for skills containing special characters (e.g., .net, c++, c#).
        """
        # Escape characters, but treat spaces as flexible separators
        escaped = re.escape(skill_lower).replace(r"\ ", r"[\s\-\.\/_]*")
        
        # If the skill starts/ends with alphanumeric characters, enforce standard word boundaries.
        # Otherwise, wrap with whitespace/punctuation boundary checks to handle characters like +, #, .
        if skill_lower and skill_lower[0].isalnum():
            escaped = rf"\b{escaped}"
        else:
            escaped = rf"(?<=[\s\-\.\/_]){escaped}"
            
        if skill_lower and skill_lower[-1].isalnum():
            escaped = rf"{escaped}\b"
        else:
            escaped = rf"{escaped}(?=[\s\-\.\/_])"
            
        return re.compile(escaped)

    def _verify_raw_text_fallback(self, skill: str, text: str) -> bool:
        """
        Universal strict boundary check preventing substring pollution (e.g., 'Java' inside 'JavaScript')
        while maintaining structural accuracy across multiple JDs and Resumes.
        """
        skill_lower = skill.strip().lower()
        if not skill_lower:
            return False

        # TIER A: Primary Flexible Pattern Validation (Catches 98% of clean boundaries)
        try:
            if self._compile_flexible_pattern(skill_lower).search(text):
                return True
        except Exception as e:
            logger.warning("Regex failed for %s: %s", skill_lower, e)

        # TIER B: Universal Token-Aware Boundary Check (Replaces the broken compaction 'in' logic)
        # Split text into distinct alphanumeric token groups
        tokens = re.findall(r'[a-z0-9]+', skill_lower)
        if not tokens:
            return False
            
        # Target the primary visual anchor token of the skill (longest component)
        primary_token = max(tokens, key=len)
        
        # Find all occurrences of that primary token in the resume text
        if len(primary_token) > 1:
            for match in re.finditer(rf'\b{re.escape(primary_token)}\b', text):
                # We found a verified, isolated instance of the word root (e.g., standalone '\bjava\b')
                # This guarantees 'java' was detected independently of 'javascript'
                return True

        return False

    # ---------------- MAIN PIPELINE ----------------
    async def score(
        self,
        jd: ParsedJd,
        resume: ParsedResume,
        existing_jd_embedding=None,
        existing_resume_embedding=None,
        database_service=None,
        contest_id: str | None = None,
        js_id: str | None = None
    ):
        candidate_name = getattr(resume, "candidate_name", "Unknown Candidate")
        logger.info("⚡ STARTING MATCH SCORE PIPELINE | Candidate: %s", candidate_name)

        structural = self.structural_engine.compute(jd=jd, resume=resume)

        matched_primary = list(set(structural.matched_primary_skills))
        matched_good = list(set(structural.matched_good_to_have_skills))

        missing_primary = [s for s in (jd.primary_skills or []) if s not in matched_primary]
        missing_good = [s for s in (jd.good_to_have_skills or []) if s not in matched_good]

        norm_text = self._normalize_raw_text(getattr(resume, "raw_text", "") or "")

        # ---------------- TIER 1: RAW TEXT FALLBACK (KEYWORD MATCHING) ----------------
        logger.info("🎯 STARTING TIER 1 EVALUATION: Keyword & Regex Fallback Validation")
        logger.info("📋 Initial missing baseline passing into Tier 1 processing: Primary: %s | Good-To-Have: %s", missing_primary, missing_good)
        
        for skill in missing_primary[:]:
            if self._verify_raw_text_fallback(skill, norm_text):
                if skill not in matched_primary:
                    matched_primary.append(skill)
                logger.info("🎯 [TIER 1 MATCH] | PRIMARY SKILL: '%s' matched via exact keyword token/regex fallback pattern.", skill)
                missing_primary.remove(skill)
            else:
                logger.debug("skip [TIER 1 NO MATCH] | PRIMARY SKILL: '%s' boundary evaluations failed or string missing.", skill)

        for skill in missing_good[:]:
            if self._verify_raw_text_fallback(skill, norm_text):
                if skill not in matched_good:
                    matched_good.append(skill)
                logger.info("🎯 [TIER 1 MATCH] | GOOD-TO-HAVE SKILL: '%s' matched via exact keyword token/regex fallback pattern.", skill)
                missing_good.remove(skill)
            else:
                logger.debug("skip [TIER 1 NO MATCH] | GOOD-TO-HAVE SKILL: '%s' boundary evaluations failed or string missing.", skill)

        # ---------------- TIER 2: HYBRID SEMANTIC MATCHING (VECTOR EMBEDDINGS) ----------------
        logger.info("🤖 STARTING TIER 2 EVALUATION: Vector Space Cosine Similarity Matching")
        resume_skills = getattr(resume, "primary_skills", []) or []
        resume_skills_list = list(set(str(s).strip() for s in resume_skills if s))

        # We need a fallback path or active database context coordinates to load micro-skills mapping safely
        if resume_skills_list and (missing_primary or missing_good) and database_service and contest_id and js_id:
            logger.info("⚡ Executing isolated micro-skills batch lookup via database persistence layer.")
            
            # Fetch entire batch dictionary from collection record fields instantly
            jd_skills_vectors = await database_service.get_or_create_jd_skills_meta(
                semantic_engine=self.semantic_engine,
                contest_id=contest_id,
                raw_jd_skills=(jd.primary_skills or []) + (jd.good_to_have_skills or [])
            )
            
            # 🚀 FIX: Passed contest_id parameter ahead of js_id to fulfill compound query layouts perfectly
            cv_skills_vectors = await database_service.get_or_create_cv_skills_meta(
                semantic_engine=self.semantic_engine,
                contest_id=contest_id,
                js_id=js_id,
                raw_cv_skills=resume_skills_list
            )

            # Updated semantic threshold ceiling floor value strategy requirement
            DYNAMIC_INFERENCE_FLOOR = 0.45

            # Process Missing Primary Skills through Memory Vector Space Comparisons
            for skill in missing_primary[:]:
                req_clean = skill.strip().lower()
                vec = jd_skills_vectors.get(req_clean)
                if not vec or not cv_skills_vectors:
                    continue

                best_score = -1.0
                best_match_skill = None

                for cv_skill_name, cv_vec in cv_skills_vectors.items():
                    score_raw = self.semantic_engine.compute_similarity_score(
                        jd_embedding=vec,
                        resume_embedding=cv_vec,
                        jd_skill_text=skill,
                        resume_skill_text=cv_skill_name
                    ) / 100.0
                    
                    if score_raw > best_score:
                        best_score = score_raw
                        best_match_skill = cv_skill_name

                if best_score >= DYNAMIC_INFERENCE_FLOOR:
                    if skill not in matched_primary:
                        matched_primary.append(skill)
                    if skill in missing_primary:
                        missing_primary.remove(skill)
                    logger.info(
                        "🤖 [TIER 2 MATCH] | PRIMARY SKILL REQUIRED: '%s' ↔️ CANDIDATE HAS: '%s' | "
                        "SIMILARITY: %.4f >= THRESHOLD: %.2f | STATUS: QUALIFIED",
                        skill, best_match_skill, best_score, DYNAMIC_INFERENCE_FLOOR
                    )
                else:
                    logger.debug(
                        "⚠️ [TIER 2 REJECT] | PRIMARY SKILL: '%s' | Best similarity score was only %.4f "
                        "with candidate skill '%s' (Failed threshold: %.2f)",
                        skill, best_score, best_match_skill, DYNAMIC_INFERENCE_FLOOR
                    )

            # Process Missing Good-To-Have Skills through Memory Vector Space Comparisons
            for skill in missing_good[:]:
                req_clean = skill.strip().lower()
                vec = jd_skills_vectors.get(req_clean)
                if not vec or not cv_skills_vectors:
                    continue

                best_score = -1.0
                best_match_skill = None

                for cv_skill_name, cv_vec in cv_skills_vectors.items():
                    score_raw = self.semantic_engine.compute_similarity_score(
                        jd_embedding=vec,
                        resume_embedding=cv_vec,
                        jd_skill_text=skill,
                        resume_skill_text=cv_skill_name
                    ) / 100.0
                    
                    if score_raw > best_score:
                        best_score = score_raw
                        best_match_skill = cv_skill_name

                if best_score >= DYNAMIC_INFERENCE_FLOOR:
                    if skill not in matched_good:
                        matched_good.append(skill)
                    if skill in missing_good:
                        missing_good.remove(skill)
                    logger.info(
                        "🤖 [TIER 2 MATCH] | GOOD-TO-HAVE SKILL REQUIRED: '%s' ↔️ CANDIDATE HAS: '%s' | "
                        "SIMILARITY: %.4f >= THRESHOLD: %.2f | STATUS: QUALIFIED",
                        skill, best_match_skill, best_score, DYNAMIC_INFERENCE_FLOOR
                    )
                else:
                    logger.debug(
                        "⚠️ [TIER 2 REJECT] | GOOD-TO-HAVE SKILL: '%s' | Best similarity score was only %.4f "
                        "with candidate skill '%s' (Failed threshold: %.2f)",
                        skill, best_score, best_match_skill, DYNAMIC_INFERENCE_FLOOR
                    )
        else:
            logger.info("⏭️ SKIPPING TIER 2: No parsed candidate primary skills profile array, missing engine database orchestration instances, or requirements met.")

        # ---------------- TIER 3: LLM IMPLIED ENGINE (CONTEXT DEEP SCAN) ----------------
        still_miss_p = [s for s in (jd.primary_skills or []) if s not in matched_primary]
        still_miss_g = [s for s in (jd.good_to_have_skills or []) if s not in matched_good]

        ctx_score, ctx_matches = await self.contextual_matcher.compute_contextual_match(
            jd=jd, resume=resume
        )

        if settings.llm_implied_skills_enabled and (still_miss_p or still_miss_g):
            logger.info("🔮 STARTING TIER 3 EVALUATION: LLM Implied Skill Deep Context Scan Engine")
            logger.info("🔮 SENDING TO LLM | Remaining Missing Primary: %s | Missing Good-To-Have: %s", still_miss_p, still_miss_g)

            ref = await self.refinement_engine.detect_implied_skills(
                norm_text, still_miss_p, still_miss_g
            )

            for s in ref.implied_primary_matches:
                if s not in matched_primary and self._verify_raw_text_fallback(s, norm_text):
                    matched_primary.append(s)
                    logger.info("🔮 [TIER 3 MATCH] | PRIMARY SKILL: '%s' | STATUS: APPROVED BY LLM CONTEXT CONVERSION ENGINE", s)

            for s in ref.implied_good_to_have_matches:
                if s not in matched_good and self._verify_raw_text_fallback(s, norm_text):
                    matched_good.append(s)
                    logger.info("🔮 [TIER 3 MATCH] | GOOD-TO-HAVE SKILL: '%s' | STATUS: APPROVED BY LLM CONTEXT CONVERSION ENGINE", s)
        else:
            logger.info("⏭️ SKIPPING TIER 3: LLM Refinement disabled or all requirements met via Tier 1 and Tier 2 loops.")

        # ---------------- CALCULATE COMPONENT WEIGHTS ----------------
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

        # ---------------- COMPREHENSIVE PIPELINE AUDIT LOG ----------------
        logger.info(
            "\n======================================================================\n"
            "🏁 PIPELINE EXECUTION SUMMARY COMPLETED FOR CANDIDATE: %s\n"
            "----------------------------------------------------------------------\n"
            "🎯 Final Primary Skills Matched       : %s\n"
            "❌ Remaining Missing Primary          : %s\n"
            "🌟 Final Good-To-Have Skills Matched   : %s\n"
            "❌ Remaining Missing Good-To-Have     : %s\n"
            "💼 Calculated Experience Allocation    : %.2f / %.2f (Total Years: %.1f, Required: %.1f)\n"
            "📍 Calculated Location Allocation      : %.2f / %.2f\n"
            "📊 SCORES BREAKDOWN IN WEIGHTS        : Primary Skills Component: %.2f, Good-To-Have Component: %.2f\n"
            "🚀 COMBINED OVERALL STRUCTURAL SCORE  : %s / 100.0\n"
            "======================================================================",
            candidate_name,
            matched_primary,
            [s for s in (jd.primary_skills or []) if s not in matched_primary],
            matched_good,
            [s for s in (jd.good_to_have_skills or []) if s not in matched_good],
            exp_score, STRUCTURAL_WEIGHTS["experience"], cand_exp, req_exp,
            loc_score, STRUCTURAL_WEIGHTS["location"],
            p_score, g_score,
            overall
        )

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

        # Safe tracking collection recovery
        jd_text = str(jd.raw_text or "").strip()
        resume_text = str(resume.raw_text or "").strip()

        # Enforce is_skill=False context routing to ensure clean tracking structures bypass skill length boundaries
        jd_emb = existing_jd_embedding or (await self.semantic_engine.generate_embedding(jd_text, is_skill=False) if jd_text else [])
        res_emb = existing_resume_embedding or (await self.semantic_engine.generate_embedding(resume_text, is_skill=False) if resume_text else [])

        return res, jd_emb, res_emb