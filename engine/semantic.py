"""
semantic.py
Embedding generation and semantic similarity scoring layer.
Implements global skill-level caching to eliminate redundant OpenAI API costs.
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

# ⚡ GLOBAL PROCESSED CACHE
_GLOBAL_SKILL_CACHE: Dict[str, List[float]] = {}


class SemanticEngine:
    """
    Handles:
    - OpenAI embedding generation
    - Local Skill-Level Vector Caching
    - Cosine similarity scoring
    - Semantic threshold gating
    - RAW TEXT ONLY semantic modeling
    """

    def __init__(
        self,
        openai_api_key: str,
        embedding_model: str = settings.OPENAI_EMBEDDING_MODEL,
    ) -> None:
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model

    # ======================================================================
    # EMBEDDING GENERATION
    # ======================================================================

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate OpenAI embedding vector from RAW TEXT ONLY utilizing global cache.
        """

        if not text or not str(text).strip():
            logger.warning("Empty text passed to embedding. Returning zero vector.")
            return [0.0] * 1536

        cleaned_text = str(text).strip().lower()
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        # GLOBAL CACHE CHECK
        if cleaned_text in _GLOBAL_SKILL_CACHE:
            logger.debug(
                "Global skill vector cache hit for phrase: '%s'",
                cleaned_text[:50]
            )
            return _GLOBAL_SKILL_CACHE[cleaned_text]

        try:
            logger.info(
                "Cache Miss: Dispatching outbound API task for vector: '%s'",
                cleaned_text[:50]
            )

            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_text,
                encoding_format="float",
            )

            if not response.data:
                logger.warning(
                    "No data field returned in OpenAI response. Returning zero vector."
                )
                return [0.0] * 1536

            embedding = response.data[0].embedding

            if not embedding:
                logger.warning(
                    "Received empty embedding vector. Returning zero vector."
                )
                return [0.0] * 1536

            _GLOBAL_SKILL_CACHE[cleaned_text] = embedding

            logger.info(
                "Embedding generated successfully | text='%s' | dimensions=%d",
                cleaned_text[:50],
                len(embedding)
            )

            return embedding

        except Exception as exc:
            logger.exception("Embedding generation failed: %s", str(exc))
            return [0.0] * 1536

    # ======================================================================
    # COSINE SIMILARITY
    # ======================================================================

    def compute_similarity_score(
        self,
        jd_embedding: list[float],
        resume_embedding: list[float],
        jd_skill: str = "",
        resume_skill: str = "",
    ) -> float:
        """
        Compute semantic cosine similarity score (0 → 100).
        """

        try:
            if not jd_embedding or not resume_embedding:
                logger.warning(
                    "Similarity skipped due to missing embeddings | jd='%s' | resume='%s'",
                    jd_skill,
                    resume_skill,
                )
                return 0.0

            vector_a = np.array(jd_embedding, dtype=np.float64)
            vector_b = np.array(resume_embedding, dtype=np.float64)

            norm_a = np.linalg.norm(vector_a)
            norm_b = np.linalg.norm(vector_b)

            if norm_a == 0.0 or norm_b == 0.0:
                logger.warning(
                    "Similarity skipped due to zero norm vector | jd='%s' | resume='%s'",
                    jd_skill,
                    resume_skill,
                )
                return 0.0

            cosine_similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)
            cosine_similarity = float(np.clip(cosine_similarity, -1.0, 1.0))

            semantic_score = max(0.0, cosine_similarity) * 100.0
            semantic_score = float(np.clip(semantic_score, 0.0, 100.0))

            logger.info(
                "SEMANTIC SIMILARITY | JD='%s' | RESUME='%s' | cosine=%.4f | semantic_score=%.2f",
                jd_skill,
                resume_skill,
                cosine_similarity,
                semantic_score,
            )
            print("==================================================================================================")
            print(f"DEBUG SEMANTIC SIMILARITY | JD='{jd_skill}' | RESUME='{resume_skill}' | cosine={cosine_similarity:.4f} | semantic_score={semantic_score:.2f}")
            print("==================================================================================================")
            return round(semantic_score, 2)

        except Exception as exc:
            logger.exception("Semantic similarity computation failed: %s", str(exc))
            return 0.0

    # ======================================================================
    # FULL PIPELINE (RAW TEXT ONLY)
    # ======================================================================

    async def semantic_match(
        self,
        jd: ParsedJd,
        resume: ParsedResume,
        existing_jd_embedding: list[float] | None = None,
        existing_resume_embedding: list[float] | None = None,
    ) -> tuple[float, list[float], list[float]]:
        """
        Full semantic matching pipeline using RAW TEXT ONLY.

        Returns:
            (semantic_score, jd_embedding, resume_embedding)
        """

        try:
            jd_text = getattr(jd, "raw_text", "") or ""
            resume_text = getattr(resume, "raw_text", "") or ""

            logger.info(
                "SEMANTIC MATCH PIPELINE STARTED | jd_raw_chars=%d | resume_raw_chars=%d",
                len(jd_text),
                len(resume_text),
            )

            if not jd_text:
                logger.warning(
                    "JD raw_text is empty — semantic score will degrade"
                )

            if not resume_text:
                logger.warning(
                    "Resume raw_text is empty — semantic score will degrade"
                )

            # JD EMBEDDING
            if existing_jd_embedding is not None:
                logger.info(
                    "USING CACHED JD EMBEDDING | dimensions=%d",
                    len(existing_jd_embedding),
                )
                jd_emb_final = existing_jd_embedding
            else:
                logger.info("GENERATING NEW JD EMBEDDING")
                jd_emb_final = await self.generate_embedding(jd_text)

            # RESUME EMBEDDING
            if existing_resume_embedding is not None:
                logger.info(
                    "USING CACHED RESUME EMBEDDING | dimensions=%d",
                    len(existing_resume_embedding),
                )
                resume_emb_final = existing_resume_embedding
            else:
                logger.info("GENERATING NEW RESUME EMBEDDING")
                resume_emb_final = await self.generate_embedding(resume_text)

            semantic_score = self.compute_similarity_score(
                jd_embedding=jd_emb_final,
                resume_embedding=resume_emb_final,
                jd_skill="FULL_JD_DOCUMENT",
                resume_skill="FULL_RESUME_DOCUMENT",
            )

            logger.info(
                "FINAL DOCUMENT SEMANTIC SCORE => %.2f",
                semantic_score,
            )

            return semantic_score, jd_emb_final, resume_emb_final

        except Exception as exc:
            logger.exception("Semantic pipeline failed: %s", str(exc))
            return 0.0, [0.0] * 1536, [0.0] * 1536

    # ======================================================================
    # TEXT NORMALIZATION
    # ======================================================================

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text blocks before embedding generation steps.
        """

        if not text:
            return ""

        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        return text