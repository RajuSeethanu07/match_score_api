"""
semantic.py
Embedding generation and semantic similarity scoring layer.
Optimized for unified document-skill sub-vector caches inside Cv_Embeddings and Jd_Embeddings.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

import numpy as np
from openai import AsyncOpenAI

from config import settings
from engine.models import ParsedJd, ParsedResume

logger = logging.getLogger("match_score_api.semantic")

# ⚡ FAST IN-MEMORY RUNTIME CACHE (Fallback helper for isolated lookups)
_GLOBAL_SKILL_CACHE: Dict[str, List[float]] = {}


class SemanticEngine:
    """
    Handles:
    - OpenAI embedding generation (Macro full-text and Micro-skills batch arrays)
    - Persistent Database Cache resolutions using unified object fields
    - High-performance local Cosine similarity evaluation via NumPy
    """

    def __init__(
        self,
        openai_api_key: str,
        embedding_model: str = settings.OPENAI_EMBEDDING_MODEL,
        db_client=None,  # Marketplace MongoDB client connection context
    ) -> None:
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.db = db_client

    def _is_valid_skill_format(self, text: str) -> bool:
        """
        Validates if a string is a clean, legitimate skill before persisting to DB.
        Prevents typos, broken metrics, or long paragraphs from cluttering data frames.
        """
        if not text:
            return False

        if len(text) < 2 and text not in ["c", "r"]:
            return False
            
        if len(text) > 50:
            return False
            
        if text.isdigit() or not re.search(r'[a-zA-Z]', text):
            return False
            
        return True

    # ======================================================================
    # MICRO LAYER: BULK SKILLS BATCH EMBEDDING
    # ======================================================================

    async def generate_bulk_skills_embeddings(self, skills: List[str]) -> Dict[str, List[float]]:
        """
        Takes a raw list of parsed skills from a profile document, sanitizes them, 
        and requests vectors from OpenAI in a single efficient batch network call.
        """
        embedded_skills_dict: Dict[str, List[float]] = {}
        skills_to_fetch: List[str] = []

        # Sanitize, clean, and deduplicate the string keys
        for skill in skills:
            if not skill:
                continue
            cleaned = str(skill).strip().lower()
            cleaned = re.sub(r"\s+", " ", cleaned)
            
            if self._is_valid_skill_format(cleaned) and cleaned not in skills_to_fetch:
                # Fast check memory layer first to keep network overhead minimal
                if cleaned in _GLOBAL_SKILL_CACHE:
                    embedded_skills_dict[cleaned] = _GLOBAL_SKILL_CACHE[cleaned]
                else:
                    skills_to_fetch.append(cleaned)

        if not skills_to_fetch:
            return embedded_skills_dict

        try:
            logger.info("🌐 [OPENAI BATCH CALL] | Dispatching concurrent requests for %d skills", len(skills_to_fetch))
            
            # OpenAI natively accepts an array of clean strings for batch processing
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=skills_to_fetch,
                encoding_format="float",
            )

            for idx, item in enumerate(response.data):
                vector = item.embedding
                skill_key = skills_to_fetch[idx]
                
                # Update runtime state and result dictionary simultaneously
                _GLOBAL_SKILL_CACHE[skill_key] = vector
                embedded_skills_dict[skill_key] = vector

            logger.info("✨ Successfully processed batch embeddings for unique skills.")

        except Exception as exc:
            logger.error("❌ Failed batch skill embedding generation via OpenAI: %s", exc)

        return embedded_skills_dict

    # ======================================================================
    # MACRO LAYER: FULL DOCUMENT EMBEDDING
    # ======================================================================

    async def generate_embedding(self, text: str, is_skill: bool = True) -> list[float]:
        """
        Generate OpenAI embedding vector utilizing persistent and in-memory cache structures.
        Aligned to handle macro full-text fields (jd_raw_text and cv_raw_text).
        """
        if not text or not str(text).strip():
            logger.warning("Empty text passed to embedding. Returning zero vector.")
            return [0.0] * 1536

        if is_skill:
            cleaned_text = str(text).strip().lower()
            cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        else:
            cleaned_text = str(text).strip()

        # STEP 1: FAST RAM CACHE VERIFICATION
        if cleaned_text in _GLOBAL_SKILL_CACHE:
            logger.info("✨ [RAM CACHE HIT] | Found memory vector for: '%s'", cleaned_text[:50])
            return _GLOBAL_SKILL_CACHE[cleaned_text]

        # STEP 2: PERSISTENT MONGODB LAYER LOOKUP (Aligned with actual structures)
        if self.db is not None:
            try:
                if is_skill:
                    # Look up by individual skill if invoked via legacy fallback
                    existing_doc = await self.db.Cv_Embeddings.find_one({"cv_raw_text": cleaned_text})
                    vector_key = "cv_embeddings"
                else:
                    # Look up full-text document records using correct database fields
                    existing_doc = await self.db.Jd_Embeddings.find_one({"jd_raw_text": cleaned_text})
                    vector_key = "jd_embeddings"
                
                if existing_doc and vector_key in existing_doc:
                    vector = existing_doc[vector_key]
                    _GLOBAL_SKILL_CACHE[cleaned_text] = vector  # Prime fast RAM
                    logger.info("💾 [MONGO CACHE HIT] | Found vector in Marketplace DB for: '%s'", cleaned_text[:50])
                    return vector
                else:
                    logger.info("🔍 [MONGO CACHE MISS] | No matching vector found in record field for: '%s'", cleaned_text[:50])
            except Exception as e:
                logger.warning("Failed to look up persistent embedding in Marketplace MongoDB: %s", e)
        else:
            logger.warning("⚠️ [CACHE DISABLED] | self.db context is missing.")

        # STEP 3: SANITIZATION SKIPPING FOR ISOLATED SKILLS
        if is_skill and not self._is_valid_skill_format(cleaned_text):
            logger.warning("⏭️ Skipping embedding for invalid skill format: '%s'", cleaned_text[:50])
            return [0.0] * 1536

        # STEP 4: OUTBOUND API CONCURRENT RETRIEVAL
        try:
            logger.info("🌐 [OPENAI API CALL] | Dispatching live outbound request for: '%s'", cleaned_text[:50])

            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_text,
                encoding_format="float",
            )

            if not response.data:
                logger.warning("No data field returned in OpenAI response. Returning zero vector.")
                return [0.0] * 1536

            embedding = response.data[0].embedding
            if not embedding:
                return [0.0] * 1536

            # STEP 5: SYNC BACK TO BOTH CACHE LAYERS
            _GLOBAL_SKILL_CACHE[cleaned_text] = embedding

            if self.db is not None:
                try:
                    if is_skill:
                        await self.db.Cv_Embeddings.update_one(
                            {"cv_raw_text": cleaned_text},
                            {"$set": {"cv_raw_text": cleaned_text, "cv_embeddings": embedding}},
                            upsert=True
                        )
                    else:
                        await self.db.Jd_Embeddings.update_one(
                            {"jd_raw_text": cleaned_text},
                            {"$set": {"jd_raw_text": cleaned_text, "jd_embeddings": embedding}},
                            upsert=True
                        )
                    logger.info("💾 [MONGO CACHE SAVE] | Saved fresh vector to Marketplace Collection for '%s'", cleaned_text[:50])
                except Exception as db_exc:
                    logger.error("Failed to write generated embedding to MongoDB cache: %s", db_exc)

            return embedding

        except Exception as exc:
            logger.exception("Embedding generation failed: %s", str(exc))
            return [0.0] * 1536

    # ======================================================================
    # MATHEMATICAL COMPUTATION LAYER
    # ======================================================================

    def compute_similarity_score(
        self,
        jd_embedding: list[float],
        resume_embedding: list[float],
        jd_skill_text: str = "",
        resume_skill_text: str = "",
    ) -> float:
        """ Runs high-performance vector mapping calculations locally within application RAM """
        try:
            if not jd_embedding or not resume_embedding:
                return 0.0

            vector_a = np.array(jd_embedding, dtype=np.float64)
            vector_b = np.array(resume_embedding, dtype=np.float64)

            norm_a = np.linalg.norm(vector_a)
            norm_b = np.linalg.norm(vector_b)

            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0

            cosine_similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)
            cosine_similarity = float(np.clip(cosine_similarity, -1.0, 1.0))

            # Rescale mathematical range cleanly to standard 0-100 matrix scores
            semantic_score = max(0.0, cosine_similarity) * 100.0
            return round(float(np.clip(semantic_score, 0.0, 100.0)), 2)

        except Exception as exc:
            logger.exception("Semantic similarity computation failed: %s", str(exc))
            return 0.0

    # ======================================================================
    # FULL PIPELINE (MACRO LEVEL CHECK)
    # ======================================================================

    async def semantic_match(
        self,
        jd: ParsedJd,
        resume: ParsedResume,
        existing_jd_embedding: list[float] | None = None,
        existing_resume_embedding: list[float] | None = None,
    ) -> tuple[float, list[float], list[float]]:
        """ Top level domain validation pass used to benchmark macro document alignment """
        try:
            jd_text = self._normalize_text(getattr(jd, "raw_text", "") or "")
            resume_text = self._normalize_text(getattr(resume, "raw_text", "") or "")

            logger.info("SEMANTIC MATCH PIPELINE STARTED | jd_raw_chars=%d | resume_raw_chars=%d", len(jd_text), len(resume_text))

            if existing_jd_embedding is not None:
                jd_emb_final = existing_jd_embedding
            else:
                jd_emb_final = await self.generate_embedding(jd_text, is_skill=False)

            if existing_resume_embedding is not None:
                resume_emb_final = existing_resume_embedding
            else:
                resume_emb_final = await self.generate_embedding(resume_text, is_skill=False)

            semantic_score = self.compute_similarity_score(
                jd_embedding=jd_emb_final,
                resume_embedding=resume_emb_final,
                jd_skill_text="FULL_JD_DOCUMENT",
                resume_skill_text="FULL_RESUME_DOCUMENT",
            )

            logger.info("FINAL DOCUMENT SEMANTIC SCORE => %.2f", semantic_score)
            return semantic_score, jd_emb_final, resume_emb_final

        except Exception as exc:
            logger.exception("Semantic pipeline failed: %s", str(exc))
            return 0.0, [0.0] * 1536, [0.0] * 1536

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text