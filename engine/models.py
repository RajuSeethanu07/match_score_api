"""
models.py
Core internal engine data models for the Match Score API.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ==============================================================================
# BASE MIXIN (SAFE SERIALIZATION LAYER)
# ==============================================================================

class SafeModelMixin:
    """
    Adds safe dict conversion for slots-based dataclasses.
    Prevents __dict__ crash issues.
    """

    def to_dict(self) -> dict:
        try:
            return asdict(self)
        except Exception:
            # fallback for extreme edge cases
            return {
                k: getattr(self, k, None)
                for k in getattr(self, "__dataclass_fields__", {})
            }

    # backward compatibility for legacy code
    def dict(self) -> dict:
        return self.to_dict()


# ==============================================================================
# PARSED JD MODEL
# ==============================================================================

@dataclass(slots=True)
class ParsedJd(SafeModelMixin):
    """
    Normalized Job Description model.

    Built from:
    - parsed_jobdescription.jobOverview
    - parsed_jobdescription.job_requirements
    """

    raw_text: str

    primary_skills: list[str] = field(default_factory=list)
    good_to_have_skills: list[str] = field(default_factory=list)

    semantic_contexts: list[str] = field(default_factory=list)

    min_experience_years: float = 0.0
    max_experience_years: float = 0.0

    locations: list[str] = field(default_factory=list)

    education: str = ""
    job_title: str = ""
    company_name: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.raw_text = self.raw_text or ""

        self.primary_skills = self.primary_skills or []
        self.good_to_have_skills = self.good_to_have_skills or []
        self.semantic_contexts = self.semantic_contexts or []
        self.locations = self.locations or []

        self.education = self.education or ""
        self.job_title = self.job_title or ""
        self.company_name = self.company_name or ""


# ==============================================================================
# PARSED RESUME MODEL
# ==============================================================================

@dataclass(slots=True)
class ParsedResume(SafeModelMixin):
    """
    Normalized Resume model.
    """

    raw_text: str

    primary_skills: list[str] = field(default_factory=list)

    semantic_contexts: list[str] = field(default_factory=list)
    experience_contexts: list[str] = field(default_factory=list)
    project_contexts: list[str] = field(default_factory=list)

    total_experience_years: float = 0.0
    experience_string_display: str = "Not specified"

    locations: list[str] = field(default_factory=list)

    candidate_name: str = ""
    current_role: str = ""
    current_company: str = ""

    education: str = ""

    email: str = ""
    phone: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.raw_text = self.raw_text or ""

        # IMPROVEMENT 1: Force elements inside primary_skills list to string format and strip trailing white spaces
        if isinstance(self.primary_skills, list):
            self.primary_skills = [str(s).strip() for s in self.primary_skills if s]
        else:
            self.primary_skills = []

        self.semantic_contexts = self.semantic_contexts or []
        self.experience_contexts = self.experience_contexts or []
        self.project_contexts = self.project_contexts or []

        self.locations = self.locations or []

        self.candidate_name = self.candidate_name or "Unknown"
        self.current_role = self.current_role or ""
        self.current_company = self.current_company or ""

        self.education = self.education or ""

        self.email = self.email or ""
        self.phone = self.phone or ""


# ==============================================================================
# LLM REFINEMENT OUTPUT MODEL
# ==============================================================================

@dataclass(slots=True)
class RefinementAdjustment(SafeModelMixin):

    implied_primary_matches: list[str] = field(default_factory=list)
    implied_good_to_have_matches: list[str] = field(default_factory=list)

    matched_pairs: list[dict[str, str]] = field(default_factory=list)

    reasoning: str = ""


# ==============================================================================
# INTERNAL MATCH SCORE RESULT MODEL
# ==============================================================================

@dataclass(slots=True)
class MatchScoreComputation(SafeModelMixin):

    primary_match_pct: float = 0.0
    good_to_have_match_pct: float = 0.0
    experience_match_pct: float = 0.0
    location_match_pct: float = 0.0

    structural_score: float = 0.0

    semantic_score: float = 0.0
    contextual_semantic_score: float = 0.0
    refined_structural_score: float = 0.0

    overall_score: float = 0.0

    matched_primary_skills: list[str] = field(default_factory=list)
    missing_primary_skills: list[str] = field(default_factory=list)

    matched_good_to_have_skills: list[str] = field(default_factory=list)
    missing_good_to_have_skills: list[str] = field(default_factory=list)

    contextual_matches: list[str] = field(default_factory=list)

    jd_embedding: list[float] = field(default_factory=list)
    resume_embedding: list[float] = field(default_factory=list)

    refinement: Optional[RefinementAdjustment] = None


# ==============================================================================
# ENGINE WEIGHT CONFIGURATION
# ==============================================================================

STRUCTURAL_WEIGHTS: dict[str, float] = {
    "primary_skills": 55.0,
    "good_to_have": 20.0,
    "experience": 20.0,
    "location": 5.0,
}


# ==============================================================================
# FINAL PIPELINE BLENDING
# ==============================================================================

# IMPROVEMENT 2: Retained dynamic score distribution models for your blending weights
PIPELINE_BLEND_WEIGHTS: dict[str, float] = {
    "refined_structural": 0.40,
    "semantic_vector": 0.25,
    "contextual_semantic": 0.20,
    "raw_structural": 0.15,
}