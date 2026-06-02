"""
score.py
Core orchestration engine implementing the 3-tier hybrid escalation matching pipeline.
Optimized with robust raw-text experience parsing and live database context synchronization.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time  # ⏱️ High-resolution timing tracking
import unicodedata
from functools import lru_cache
from typing import Any
import numpy as np

from config import settings
from engine.models import ParsedJd, ParsedResume, STRUCTURAL_WEIGHTS
from engine.structural import StructuralScorer
from engine.semantic import SemanticEngine
from engine.refinement import RefinementEngine
from engine.formatter import build_final_response

logger = logging.getLogger("match_score_api.scorer")


class MatchScoreEngine:

    def __init__(self, db_client=None) -> None:
        self.structural_engine = StructuralScorer()
        self.semantic_engine = SemanticEngine(openai_api_key=settings.OPENAI_API_KEY, db_client=db_client)
        self.refinement_engine = RefinementEngine(openai_api_key=settings.OPENAI_API_KEY)

    # ---------------- SAFE NORMALIZATION ----------------
    def _normalize_raw_text(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text).lower()
        text = re.sub(r"[\u200b\u200c\u200d]", " ", text)
        return " " + " ".join(text.split()) + " "

    def _parse_experience(self, val: Any) -> float:
        """
        Robust string/numeric processing engine to clean and extract exact numerical 
        experience values from dirty parameters (e.g. '5+ years', '3.5', 'Fresher').
        Ens ensures comparison gates work flawlessly without throwing ValueErrors.
        """
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        
        cleaned = str(val).strip().lower()
        if not cleaned or cleaned in ["none", "null", "not specified", "fresher"]:
            return 0.0
            
        try:
            return float(cleaned)
        except ValueError:
            pass
            
        # Fallback regex parsing to extract first clean numerical digit block found
        match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    @lru_cache(maxsize=5000)
    def _compile_flexible_pattern(skill_lower: str):
        """
        Compiles a robust regex pattern handling word boundaries safely,
        even for skills containing special characters (e.g., .net, c++, c#).
        """
        cleaned = skill_lower.strip()
        
        # 1. Handle trailing framework naming conventions (e.g., 'angularjs' -> base 'angular' with optional 'js')
        if cleaned.endswith("js") and len(cleaned) > 4:
            base_tech = cleaned[:-2].strip()
            escaped = re.escape(base_tech).replace(r"\ ", r"[\s\-\.\/_]*") + r"(js)?"
        else:
            escaped = re.escape(cleaned).replace(r"\ ", r"[\s\-\.\/_]*")
            
        # 2. Handle trailing numeric versions dynamically (e.g., 'java 8' can fallback to match 'java')
        if re.search(r"\s+\d+$", cleaned):
            base_token = re.sub(r"\s+\d+$", "", cleaned).strip()
            version_token = re.search(r"\d+$", cleaned).group()
            escaped_base = re.escape(base_token).replace(r"\ ", r"[\s\-\.\/_]*")
            escaped = rf"{escaped_base}(?:[\s\-\.\/_]*{version_token})?"

        # 3. Inject safe isolation lookarounds to block Java matching inside JavaScript / TypeScript
        if "java" in skill_lower and "script" not in skill_lower:
            escaped = rf"\bjava\b(?![- ]?[ss]cript)|{escaped}"

        # 4. Enforce appropriate spatial lookarounds and strict token boundary logic
        if skill_lower and skill_lower[0].isalnum():
            escaped = rf"\b({escaped})"
        else:
            escaped = rf"(?<=[\s\-\.\/_])({escaped})"
            
        if skill_lower and skill_lower[-1].isalnum():
            escaped = rf"{escaped}\b"
        else:
            escaped = rf"{escaped}(?=[\s\-\.\/_])"
            
        return re.compile(escaped)

    def _verify_raw_text_fallback(self, skill: str, text: str) -> bool:
        """
        Universal strict boundary check preventing substring pollution (e.g., 'Java' inside 'JavaScript')
        and split-word pollution.
        """
        skill_lower = skill.strip().lower()
        if not skill_lower:
            return False

        try:
            if self._compile_flexible_pattern(skill_lower).search(text):
                return True
        except Exception as e:
            logger.warning("Regex failed for %s: %s", skill_lower, e)

        if " " in skill_lower:
            return False

        tokens = [t for t in re.findall(r'[a-z0-9]+', skill_lower) if len(t) > 1]
        if not tokens:
            return False
            
        for token in tokens:
            if not re.search(rf'\b{re.escape(token)}\b', text):
                return False

        return True

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

        # 🛠️ THE FIX: Dynamically link database connection pool context to semantic layer if unassigned
        if database_service and not self.semantic_engine.db:
            resolved_db = getattr(database_service, "db", getattr(database_service, "client", None))
            if resolved_db:
                self.semantic_engine.db = resolved_db
                logger.info("💾 [ENGINE AUTO-SYNC] Successfully paired active database driver to Semantic Engine layer.")

        # Ensure min and max experience structural values exist inside the JD metadata tracking block
        if isinstance(jd.metadata, dict):
            jd.metadata["min_experience"] = getattr(jd, "min_experience_years", 0.0)
            jd.metadata["max_experience"] = getattr(jd, "max_experience_years", 0.0)

        # =====================================================================================
        # 🛑 UPFRONT FORCED SKILL EMBEDDING PHASES (Guarantees execution for all pipeline passes)
        # =====================================================================================
        jd_skills_vectors = {}
        cv_skills_vectors = {}
        
        resume_skills = getattr(resume, "primary_skills", []) or []
        resume_skills_list = list(set(str(s).strip() for s in resume_skills if s))
        raw_jd_skills = (jd.primary_skills or []) + (jd.good_to_have_skills or [])

        if database_service and contest_id and js_id:
            logger.info("📡 [UPFRONT EMBEDDING] Initiating mandatory phase-level skill vectorization caches...")
            skills_summary_start = time.perf_counter()

            try:
                logger.info("📡 Checking Jd_Embeddings cache collection for contestId: %s", contest_id)
                jd_skills_vectors = await database_service.get_or_create_jd_skills_meta(
                    semantic_engine=self.semantic_engine,
                    contest_id=contest_id,
                    raw_jd_skills=raw_jd_skills
                )
                logger.info("📦 JD Skills Embedding Resolution complete. Found keys: %s", list(jd_skills_vectors.keys()))
            except Exception as jd_emb_ex:
                logger.error("❌ Failed resolving upfront JD skill embeddings: %s", str(jd_emb_ex))

            try:
                logger.info("📡 Checking Cv_Embeddings cache collection for contestId: %s, jsId: %s", contest_id, js_id)
                cv_skills_vectors = await database_service.get_or_create_cv_skills_meta(
                    semantic_engine=self.semantic_engine,
                    contest_id=contest_id,
                    js_id=js_id,
                    raw_cv_skills=resume_skills_list
                )
                logger.info("📦 CV Skills Embedding Resolution complete. Found keys: %s", list(cv_skills_vectors.keys()))
            except Exception as cv_emb_ex:
                logger.error("❌ Failed resolving upfront CV skill embeddings: %s", str(cv_emb_ex))

            total_skills_elapsed = time.perf_counter() - skills_summary_start
            logger.info("⏱️ [TOTAL TIME SKILLS SUMMARY] Upfront processing complete for both (JD + CV) parsed skills arrays in %.4f seconds.", total_skills_elapsed)
        else:
            logger.warning("⚠️ Upfront caching skipped: Database service instances or orchestration metadata IDs are unassigned.")

        # Execute structural scoring baseline mappings
        structural = self.structural_engine.compute(jd=jd, resume=resume)

        # Base tracking lists from structural engine (Pre-processed initial matches)
        matched_primary = list(set(structural.matched_primary_skills))
        matched_good = list(set(structural.matched_good_to_have_skills))

        # Explicit containers to record EXACTLY which skills get validated per Tier
        tier1_matches = []
        tier2_matches = []
        tier3_matches = []

        missing_primary = [s for s in (jd.primary_skills or []) if s not in matched_primary]
        missing_good = [s for s in (jd.good_to_have_skills or []) if s not in matched_good]

        norm_text = self._normalize_raw_text(getattr(resume, "raw_text", "") or "")
        resume_text = str(resume.raw_text or "").strip()

        # ---------------- TIER 1: RAW TEXT FALLBACK (KEYWORD MATCHING) ----------------
        logger.info("🎯 STARTING TIER 1 EVALUATION: Keyword & Regex Fallback Validation")
        logger.info("📋 Initial missing baseline passing into Tier 1 processing: Primary: %s | Good-To-Have: %s", missing_primary, missing_good)
        
        for skill in missing_primary[:]:
            if self._verify_raw_text_fallback(skill, norm_text):
                if skill not in matched_primary:
                    matched_primary.append(skill)
                tier1_matches.append(f"{skill} (Primary)")
                logger.info("🎯 [TIER 1 MATCH] | PRIMARY SKILL: '%s' matched via exact keyword token/regex fallback pattern.", skill)
                missing_primary.remove(skill)
            else:
                logger.debug("skip [TIER 1 NO MATCH] | PRIMARY SKILL: '%s' boundary evaluations failed or string missing.", skill)

        for skill in missing_good[:]:
            if self._verify_raw_text_fallback(skill, norm_text):
                if skill not in matched_good:
                    matched_good.append(skill)
                tier1_matches.append(f"{skill} (Good-To-Have)")
                logger.info("🎯 [TIER 1 MATCH] | GOOD-TO-HAVE SKILL: '%s' matched via exact keyword token/regex fallback pattern.", skill)
                missing_good.remove(skill)
            else:
                logger.debug("skip [TIER 1 NO MATCH] | GOOD-TO-HAVE SKILL: '%s' boundary evaluations failed or string missing.", skill)

        # ---------------- TIER 2: HYBRID SEMANTIC MATCHING (VECTOR EMBEDDINGS) ----------------
        logger.info("🤖 STARTING TIER 2 EVALUATION: Vector Space Cosine Similarity Matching")

        if (missing_primary or missing_good) and jd_skills_vectors and cv_skills_vectors:
            logger.info("⚡ Executing high-performance matrix vector space comparisons using pre-fetched upfront skill embeddings.")
            
            # 🛠️ Set strictly to 70% matching floor requirement
            DYNAMIC_INFERENCE_FLOOR = 0.70

            # 🛠️ PRE-CONSTRUCT 2D NUMPY MATRIX FOR LIGHTNING-FAST BATCH CALCULATIONS
            cv_skill_names = list(cv_skills_vectors.keys())
            cv_matrix = np.array([cv_skills_vectors[name] for name in cv_skill_names], dtype=np.float64)
            cv_norms = np.linalg.norm(cv_matrix, axis=1)
            cv_norms[cv_norms == 0] = 1.0

            # Process Missing Primary Skills through High-Performance Matrix Vector Space Comparisons
            for skill in missing_primary[:]:
                req_clean = skill.strip().lower()
                vec = jd_skills_vectors.get(req_clean)
                if vec is None or cv_matrix is None or len(cv_skill_names) == 0:
                    logger.warning("⚠️ Missing vectors for primary skill: '%s' (vector in JD map: %s, CV vectors present: %s)", skill, bool(vec), cv_matrix is not None)
                    continue

                jd_vec = np.array(vec, dtype=np.float64)
                jd_norm = np.linalg.norm(jd_vec)
                if jd_norm == 0:
                    jd_norm = 1.0

                dot_products = np.dot(cv_matrix, jd_vec)
                similarities = dot_products / (cv_norms * jd_norm)

                best_idx = np.argmax(similarities)
                best_score = float(similarities[best_idx])
                best_match_skill = cv_skill_names[best_idx]

                if best_score >= DYNAMIC_INFERENCE_FLOOR:
                    if skill not in matched_primary:
                        matched_primary.append(skill)
                    if skill in missing_primary:
                        missing_primary.remove(skill)
                    tier2_matches.append(f"{skill} ↔️ {best_match_skill} (Primary, Sim: {best_score:.2f})")
                    logger.info(
                        "🤖 [TIER 2 MATCH] | PRIMARY SKILL REQUIRED: '%s' ↔️ CANDIDATE HAS: '%s' | "
                        "SIMILARITY: %.4f >= THRESHOLD: %.2f | STATUS: QUALIFIED",
                        skill, best_match_skill, best_score, DYNAMIC_INFERENCE_FLOOR
                    )
                else:
                    logger.info(
                        "⚠️ [TIER 2 REJECT] | PRIMARY SKILL: '%s' | Best similarity score was only %.4f "
                        "with candidate skill '%s' (Failed threshold: %.2f)",
                        skill, best_score, best_match_skill, DYNAMIC_INFERENCE_FLOOR
                    )

            # Process Missing Good-To-Have Skills through High-Performance Matrix Vector Space Comparisons
            for skill in missing_good[:]:
                req_clean = skill.strip().lower()
                vec = jd_skills_vectors.get(req_clean)
                if vec is None or cv_matrix is None or len(cv_skill_names) == 0:
                    logger.warning("⚠️ Missing vectors for good-to-have skill: '%s' (vector in JD map: %s, CV vectors present: %s)", skill, bool(vec), cv_matrix is not None)
                    continue

                jd_vec = np.array(vec, dtype=np.float64)
                jd_norm = np.linalg.norm(jd_vec)
                if jd_norm == 0:
                    jd_norm = 1.0

                dot_products = np.dot(cv_matrix, jd_vec)
                similarities = dot_products / (cv_norms * jd_norm)

                best_idx = np.argmax(similarities)
                best_score = float(similarities[best_idx])
                best_match_skill = cv_skill_names[best_idx]

                if best_score >= DYNAMIC_INFERENCE_FLOOR:
                    if skill not in matched_good:
                        matched_good.append(skill)
                    if skill in missing_good:
                        missing_good.remove(skill)
                    tier2_matches.append(f"{skill} ↔️ {best_match_skill} (Good-To-Have, Sim: {best_score:.2f})")
                    logger.info(
                        "🤖 [TIER 2 MATCH] | GOOD-TO-HAVE SKILL REQUIRED: '%s' ↔️ CANDIDATE HAS: '%s' | "
                        "SIMILARITY: %.4f >= THRESHOLD: %.2f | STATUS: QUALIFIED",
                        skill, best_match_skill, best_score, DYNAMIC_INFERENCE_FLOOR
                    )
                else:
                    logger.info(
                        "⚠️ [TIER 2 REJECT] | GOOD-TO-HAVE SKILL: '%s' | Best similarity score was only %.4f "
                        "with candidate skill '%s' (Failed threshold: %.2f)",
                        skill, best_score, best_match_skill, DYNAMIC_INFERENCE_FLOOR
                    )
        else:
            logger.info("⏭️ SKIPPING TIER 2 MATCHING CALCULATIONS: Target missing skill array empty or upfront embedding matrices unavailable.")

        # ---------------- TIER 3: LLM IMPLIED ENGINE (CONTEXT DEEP SCAN - FORCED EXECUTION) ----------------
        if settings.llm_implied_skills_enabled:
            logger.info("🔮 STARTING TIER 3 EVALUATION (FORCED MODE): LLM Implied Skill Deep Context Scan Engine")
            
            still_miss_p = [s for s in (jd.primary_skills or []) if s not in matched_primary]
            still_miss_g = [s for s in (jd.good_to_have_skills or []) if s not in matched_good]
            
            logger.info("🔮 SENDING TO LLM | Remaining Missing Primary: %s | Missing Good-To-Have: %s", still_miss_p, still_miss_g)

            if still_miss_p or still_miss_g:
                ref = await self.refinement_engine.detect_implied_skills(
                    norm_text, still_miss_p, still_miss_g
                )

                for s in ref.implied_primary_matches:
                    if s not in matched_primary:
                        matched_primary.append(s)
                        tier3_matches.append(f"{s} (Primary)")
                        logger.info("🔮 [TIER 3 MATCH] | PRIMARY SKILL: '%s' | STATUS: APPROVED BY LLM CONTEXT CONVERSION ENGINE", s)

                for s in ref.implied_good_to_have_matches:
                    if s not in matched_good:
                        matched_good.append(s)
                        tier3_matches.append(f"{s} (Good-To-Have)")
                        logger.info("🔮 [TIER 3 MATCH] | GOOD-TO-HAVE SKILL: '%s' | STATUS: APPROVED BY LLM CONTEXT CONVERSION ENGINE", s)
        else:
            logger.info("⏭️ SKIPPING TIER 3: LLM Refinement disabled via settings.")

        # ---------------- CALCULATE COMPONENT WEIGHTS ----------------
        total_p = len(jd.primary_skills or []) or 1
        total_g = len(jd.good_to_have_skills or []) or 1

        p_score = (len(matched_primary) / total_p) * 55.0
        g_score = (len(matched_good) / total_g) * 20.0

        # 🛠️ THE FIX: Safe normalized extraction prevent float conversion discrepancies on structural years rules
        req_exp = self._parse_experience(getattr(jd, "min_experience_years", 0.0))
        cand_exp = self._parse_experience(getattr(resume, "total_experience_years", 0.0))

        # Perform clean numeric comparison against structural bounds with safe rounding threshold logic
        if cand_exp >= req_exp:
            exp_score = 20.0
            exp_pct = 100.0
        else:
            exp_score = 0.0
            exp_pct = 0.0

        # 🛠️ LOCATION RULE GATE: Strict 5 Points All-or-Nothing
        jd_location_raw = str(getattr(jd, "location", "") or "").strip().lower()
        is_location_unspecified = not jd_location_raw or jd_location_raw in ["", "none", "null", "any"]
        is_remote_jd = "remote" in jd_location_raw or "work from home" in jd_location_raw

        if is_location_unspecified or is_remote_jd:
            loc_score = 5.0
            loc_match_pct = 100.0
        else:
            if float(structural.location_match_pct) >= 100.0:
                loc_score = 5.0
                loc_match_pct = 100.0
            else:
                loc_score = 0.0
                loc_match_pct = 0.0

        # Aggregate raw scores together up to an absolute combined 100.0 pool maximum boundary
        overall = min(round(p_score + g_score + exp_score + loc_score, 1), 100.0)

        # ---------------- COMPREHENSIVE PIPELINE AUDIT LOG ----------------
        structural_pre_matches = [
            f"{s} (Primary)" for s in (jd.primary_skills or []) if s in list(set(structural.matched_primary_skills))
        ] + [
            f"{s} (Good-To-Have)" for s in (jd.good_to_have_skills or []) if s in list(set(structural.matched_good_to_have_skills))
        ]

        logger.info(
            "\n======================================================================\n"
            "🏁 PIPELINE EXECUTION SUMMARY COMPLETED FOR CANDIDATE: %s\n"
            "----------------------------------------------------------------------\n"
            "🔍 [TIER BREAKDOWN ANALYSIS]:\n"
            "   🔹 PRE-MATCH (Initial Structural Engine) : %s\n"
            "   🔹 TIER 1    (Keyword/Regex Exact Hits)  : %s\n"
            "   🔹 TIER 2    (Vector Embedding Synthetics): %s\n"
            "   🔹 TIER 3    (LLM Implied Context Scan)   : %s\n"
            "----------------------------------------------------------------------\n"
            "🎯 Final Primary Skills Matched         : %s\n"
            "❌ Remaining Missing Primary           : %s\n"
            "🌟 Final Good-To-Have Skills Matched  : %s\n"
            "❌ Remaining Missing Good-To-Have     : %s\n"
            "💼 Calculated Experience Allocation    : %.2f / 20.0 (Total Years: %.1f, Required: %.1f)\n"
            "📍 Calculated Location Allocation      : %.2f / 5.0\n"
            "📊 SCORES BREAKDOWN IN WEIGHTS         : Primary Skills Component: %.2f / 55.0, Good-To-Have Component: %.2f / 20.0\n"
            "🚀 COMBINED OVERALL STRUCTURAL SCORE  : %s / 100.0\n"
            "======================================================================",
            candidate_name,
            structural_pre_matches if structural_pre_matches else "None",
            tier1_matches if tier1_matches else "None",
            tier2_matches if tier2_matches else "None",
            tier3_matches if tier3_matches else "None",
            matched_primary,
            [s for s in (jd.primary_skills or []) if s not in matched_primary],
            matched_good,
            [s for s in (jd.good_to_have_skills or []) if s not in matched_good],
            exp_score, cand_exp, req_exp,
            loc_score,
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
            location_match_pct=loc_match_pct
        )

        if hasattr(res, "experienceMatch") and isinstance(res.experienceMatch, dict):
            res.experienceMatch["candidateExperience"] = f"{cand_exp} years" if cand_exp else "Not specified"

        if hasattr(res, "jdMetadata") and isinstance(res.jdMetadata, dict):
            res.jdMetadata["min_experience"] = getattr(jd, "min_experience_years", 0.0)
            res.jdMetadata["max_experience"] = getattr(jd, "max_experience_years", 0.0)

        jd_text = str(jd.raw_text or "").strip()
        resume_text = str(resume.raw_text or "").strip()

        # 🛠️ PARALLELIZE INTERNET I/O FOR FULL-TEXT EMBEDDINGS
        async def fetch_jd_macro_embedding() -> list[float]:
            if existing_jd_embedding:
                return existing_jd_embedding
            logger.info("📡 [TEXT EMBEDDING CACHE MISS] Generating fresh macro JD text embedding via OpenAI...")
            return await self.semantic_engine.generate_embedding(jd_text, is_skill=False, context="JD") if jd_text else []

        async def fetch_resume_macro_embedding() -> list[float]:
            if existing_resume_embedding:
                return existing_resume_embedding
            logger.info("📡 [TEXT EMBEDDING CACHE MISS] Generating fresh macro CV text embedding for Candidate: %s via OpenAI...", candidate_name)
            return await self.semantic_engine.generate_embedding(resume_text, is_skill=False, context="CV") if resume_text else []

        jd_emb, res_emb = await asyncio.gather(
            fetch_jd_macro_embedding(),
            fetch_resume_macro_embedding()
        )

        return res, jd_emb, res_emb