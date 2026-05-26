"""
contextual_semantic.py
Production-ready contextual semantic matching engine using OpenAI embeddings + cosine similarity.
"""

from __future__ import annotations
import logging
from typing import List
import numpy as np
from openai import AsyncOpenAI
from config import settings
from engine.models import ParsedJd, ParsedResume

logger = logging.getLogger("match_score_api.contextual_semantic")

DEFAULT_SIMILARITY_THRESHOLD = 0.72
MAX_CONTEXTS_PER_SIDE = 25
MAX_CONTEXT_LENGTH = 1200
ZERO_VECTOR_SIZE = 1536

class ContextualSemanticMatcher:
    """
    Production-grade semantic context matching engine.
    Responsibilities: Generate embeddings, compute pairwise similarity, detect alignment, and return scores.
    """

    def __init__(
        self,
        openai_api_key: str,
        embedding_model: str = settings.OPENAI_EMBEDDING_MODEL,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    async def compute_contextual_match(self, jd: ParsedJd, resume: ParsedResume) -> tuple[float, list[str]]:
        """Compute contextual semantic alignment score. Returns (contextual_score, contextual_matches)."""
        try:
            jd_contexts = self._prepare_contexts(jd.semantic_contexts)
            resume_contexts = self._prepare_contexts(resume.semantic_contexts)

            if not jd_contexts or not resume_contexts:
                logger.warning("Contextual semantic matching skipped due to empty contexts.")
                return 0.0, []

            logger.info(
                "Generating contextual embeddings | jd_contexts=%d | resume_contexts=%d",
                len(jd_contexts), len(resume_contexts),
            )

            jd_embeddings = await self._generate_embeddings(jd_contexts)
            resume_embeddings = await self._generate_embeddings(resume_contexts)
            similarity_matrix = self._compute_similarity_matrix(jd_embeddings=jd_embeddings, resume_embeddings=resume_embeddings)

            contextual_score = self._calculate_contextual_score(similarity_matrix)
            contextual_matches = self._extract_contextual_matches(
                similarity_matrix=similarity_matrix, jd_contexts=jd_contexts, resume_contexts=resume_contexts
            )

            logger.info("Contextual semantic score computed successfully | score=%.2f | matches=%d", contextual_score, len(contextual_matches))
            return contextual_score, contextual_matches

        except Exception as exc:
            logger.exception("Contextual semantic matching failed: %s", str(exc))
            return 0.0, []

    def _prepare_contexts(self, contexts: List[str]) -> List[str]:
        """Clean, normalize, deduplicate, and limit contexts."""
        cleaned = []
        seen = set()
        for context in contexts:
            if not context: continue
            normalized = str(context).strip()
            if not normalized or normalized in seen: continue

            cleaned.append(normalized[:MAX_CONTEXT_LENGTH])
            seen.add(normalized)
        return cleaned[:MAX_CONTEXTS_PER_SIDE]

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings asynchronously."""
        if not texts: return []
        try:
            response = await self.client.embeddings.create(model=self.embedding_model, input=texts)
            if not response.data:
                logger.warning("No embedding data returned from OpenAI.")
                return [[0.0] * ZERO_VECTOR_SIZE for _ in texts]
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.exception("Embedding generation failed: %s", str(exc))
            return [[0.0] * ZERO_VECTOR_SIZE for _ in texts]

    @staticmethod
    def _cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
        """Compute cosine similarity safely."""
        try:
            a = np.array(vector_a, dtype=np.float64)
            b = np.array(vector_b, dtype=np.float64)
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            if denominator == 0: return 0.0
            return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
        except Exception:
            return 0.0

    def _compute_similarity_matrix(self, jd_embeddings: List[List[float]], resume_embeddings: List[List[float]]) -> np.ndarray:
        """Compute pairwise similarity matrix."""
        matrix = np.zeros((len(jd_embeddings), len(resume_embeddings)))
        for i, jd_vector in enumerate(jd_embeddings):
            for j, resume_vector in enumerate(resume_embeddings):
                matrix[i][j] = self._cosine_similarity(jd_vector, resume_vector)
        return matrix

    def _calculate_contextual_score(self, similarity_matrix: np.ndarray) -> float:
        """Aggregate contextual semantic score using strongest matches."""
        if similarity_matrix.size == 0: return 0.0
        best_matches = []
        for row in similarity_matrix:
            best_similarity = float(np.max(row))
            if best_similarity >= self.similarity_threshold:
                best_matches.append(best_similarity)

        if not best_matches: return 0.0
        score = float(np.mean(best_matches)) * 100.0
        return round(min(score, 100.0), 2)

    def _extract_contextual_matches(
        self, similarity_matrix: np.ndarray, jd_contexts: List[str], resume_contexts: List[str]
    ) -> list[str]:
        """Build explainable contextual semantic matches."""
        matches = []
        if similarity_matrix.size == 0: return matches

        for i, row in enumerate(similarity_matrix):
            best_idx = int(np.argmax(row))
            best_score = float(row[best_idx])
            if best_score < self.similarity_threshold: continue

            jd_text = jd_contexts[i][:140]
            resume_text = resume_contexts[best_idx][:140]
            matches.append(f"JD Context: '{jd_text}' matched Resume Context: '{resume_text}' ({round(best_score * 100, 1)}%)")
        return matches

contextual_semantic_matcher = ContextualSemanticMatcher(
    openai_api_key=settings.OPENAI_API_KEY,
)