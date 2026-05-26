"""
models.py
Core internal engine data models for the Match Score API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ==============================================================================
# PARSED JD MODEL
# ==============================================================================

@dataclass(slots=True)
class ParsedJd:
    """
    Normalized Job Description model.

    Built from:
    - parsed_jobdescription.jobOverview
    - parsed_jobdescription.job_requirements
    """

    # --------------------------------------------------------------------------
    # RAW DOCUMENT CONTENT
    # --------------------------------------------------------------------------

    raw_text: str

    # --------------------------------------------------------------------------
    # STRUCTURED SKILLS
    # --------------------------------------------------------------------------

    primary_skills: list[str] = field(default_factory=list)

    good_to_have_skills: list[str] = field(default_factory=list)

    # --------------------------------------------------------------------------
    # CONTEXTUAL SEMANTIC REQUIREMENTS
    # --------------------------------------------------------------------------

    semantic_contexts: list[str] = field(default_factory=list)

    # --------------------------------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------------------------------

    min_experience_years: float = 0.0

    max_experience_years: float = 0.0

    # --------------------------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------------------------

    locations: list[str] = field(default_factory=list)

    # --------------------------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------------------------

    education: str = ""

    # --------------------------------------------------------------------------
    # JOB PROFILE
    # --------------------------------------------------------------------------

    job_title: str = ""

    company_name: str = ""

    # --------------------------------------------------------------------------
    # EXTRA METADATA
    # --------------------------------------------------------------------------

    metadata: dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------------------
    # PRODUCTION SAFETY NORMALIZATION
    # --------------------------------------------------------------------------
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
class ParsedResume:
    """
    Normalized Resume model.

    Built from:
    - completeResumeDetails.parsed_resume.ResumeParserData
    - personal_info
    - work_experience
    """

    # --------------------------------------------------------------------------
    # RAW DOCUMENT CONTENT
    # --------------------------------------------------------------------------

    raw_text: str

    # --------------------------------------------------------------------------
    # PRIMARY EXTRACTED SKILLS
    # --------------------------------------------------------------------------

    primary_skills: list[str] = field(default_factory=list)

    # --------------------------------------------------------------------------
    # CONTEXTUAL EXPERIENCE INTELLIGENCE
    # --------------------------------------------------------------------------

    semantic_contexts: list[str] = field(default_factory=list)

    experience_contexts: list[str] = field(default_factory=list)

    project_contexts: list[str] = field(default_factory=list)

    # --------------------------------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------------------------------

    total_experience_years: float = 0.0

    experience_string_display: str = "Not specified"

    # --------------------------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------------------------

    locations: list[str] = field(default_factory=list)

    # --------------------------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------------------------

    candidate_name: str = ""

    current_role: str = ""

    current_company: str = ""

    # --------------------------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------------------------

    education: str = ""

    # --------------------------------------------------------------------------
    # CONTACT
    # --------------------------------------------------------------------------

    email: str = ""

    phone: str = ""

    # --------------------------------------------------------------------------
    # EXTRA METADATA
    # --------------------------------------------------------------------------

    metadata: dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------------------
    # PRODUCTION SAFETY NORMALIZATION
    # --------------------------------------------------------------------------
    def __post_init__(self):
        self.raw_text = self.raw_text or ""

        self.primary_skills = self.primary_skills or []
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
class RefinementAdjustment:
    """
    LLM implied-skill detection response (Tier 3).
    Skills must be evidenced in candidate raw_text; lists use JD requirement spelling.
    """

    implied_primary_matches: list[str] = field(default_factory=list)

    implied_good_to_have_matches: list[str] = field(default_factory=list)

    matched_pairs: list[dict[str, str]] = field(default_factory=list)

    reasoning: str = ""


# ==============================================================================
# INTERNAL MATCH SCORE RESULT MODEL
# ==============================================================================

@dataclass(slots=True)
class MatchScoreComputation:
    """
    Internal scoring aggregation model.

    Used between:
    - structural.py
    - semantic.py
    - refinement.py
    - formatter.py
    - contextual_semantic.py
    """

    # --------------------------------------------------------------------------
    # RAW STRUCTURAL SCORES
    # --------------------------------------------------------------------------

    primary_match_pct: float = 0.0

    good_to_have_match_pct: float = 0.0

    experience_match_pct: float = 0.0

    location_match_pct: float = 0.0

    structural_score: float = 0.0

    # --------------------------------------------------------------------------
    # SEMANTIC VECTOR SCORES
    # --------------------------------------------------------------------------

    semantic_score: float = 0.0

    contextual_semantic_score: float = 0.0

    refined_structural_score: float = 0.0

    overall_score: float = 0.0

    # --------------------------------------------------------------------------
    # SKILL MAPPING
    # --------------------------------------------------------------------------

    matched_primary_skills: list[str] = field(default_factory=list)

    missing_primary_skills: list[str] = field(default_factory=list)

    matched_good_to_have_skills: list[str] = field(default_factory=list)

    missing_good_to_have_skills: list[str] = field(default_factory=list)

    # --------------------------------------------------------------------------
    # CONTEXTUAL EXPERIENCE MATCHING
    # --------------------------------------------------------------------------

    contextual_matches: list[str] = field(default_factory=list)

    # --------------------------------------------------------------------------
    # EMBEDDING CACHE VECTORS
    # --------------------------------------------------------------------------

    jd_embedding: list[float] = field(default_factory=list)

    resume_embedding: list[float] = field(default_factory=list)

    # --------------------------------------------------------------------------
    # LLM REFINEMENT
    # --------------------------------------------------------------------------

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

PIPELINE_BLEND_WEIGHTS: dict[str, float] = {
    "refined_structural": 0.40,
    "semantic_vector": 0.25,
    "contextual_semantic": 0.20,
    "raw_structural": 0.15,
}