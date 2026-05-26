"""
formatter.py
Builds the final ATS response payload structure.
"""

from __future__ import annotations

from schema import (
    MatchScoreResponse,
    ScoringDataContainer,
    FinalScoresEnvelope,
    SnsScoringResult,
    ResumeMatchBlock,
    ScoreBreakdownBlock,
    SkillsMatchBlock,
    ExperienceMatchBlock,
    LocationMatchBlock,
)

from engine.models import (
    ParsedJd,
    ParsedResume,
    STRUCTURAL_WEIGHTS,
)


def build_final_response(
    overall_score: float,
    jd: ParsedJd,
    resume: ParsedResume,
    matched_primary: list[str],
    missing_primary: list[str],
    matched_good: list[str],
    missing_good: list[str],
    primary_score: float,
    good_score: float,
    experience_score: float,
    location_score: float,
    experience_match_pct: float,
    location_match_pct: float,
    semantic_score: float = 0.0,
    contextual_semantic_score: float = 0.0,
    contextual_matches: list[str] | None = None,
) -> MatchScoreResponse:

    contextual_matches = contextual_matches or []

    # ==========================================================
    # EXPERIENCE FORMATTING
    # ==========================================================

    if jd.max_experience_years > 0:
        required_experience = (
            f"{int(jd.min_experience_years)}-{int(jd.max_experience_years)} years"
        )
    else:
        required_experience = f"{int(jd.min_experience_years)}+ years"

    if float(resume.total_experience_years or 0.0) > 0.0:
        candidate_experience = f"{resume.total_experience_years} years"
    elif resume.experience_string_display and resume.experience_string_display != "Not specified":
        candidate_experience = resume.experience_string_display
    else:
        candidate_experience = "Not specified"

    if experience_match_pct >= 100:
        experience_explanation = (
            f"Candidate has {candidate_experience} "
            f"which meets the minimum required "
            f"{int(jd.min_experience_years)} years."
        )
    else:
        experience_explanation = (
            f"Candidate has {candidate_experience} "
            f"which does not meet the minimum required "
            f"{int(jd.min_experience_years)} years."
        )

    # ==========================================================
    # LOCATION FORMATTING
    # ==========================================================

    job_location = ", ".join(jd.locations) if jd.locations else "Not specified"
    candidate_location = ", ".join(resume.locations) if resume.locations else "Not specified"

    # ==========================================================
    # CONTEXTUAL EXPLANATION
    # ==========================================================

    contextual_explanation = (
        f"{len(contextual_matches)} contextual semantic alignments detected "
        f"between resume experience and JD expectations."
        if contextual_matches
        else "No strong contextual semantic alignments detected."
    )

    # ==========================================================
    # 🚨 FIX: CLEAN RESUME METADATA OUTPUT (IMPORTANT CHANGE)
    # ==========================================================

    resume_metadata = {
        "mongo_id": resume.metadata.get("mongo_id", "") if resume.metadata else "",
        "first_name": resume.metadata.get("first_name", "") if resume.metadata else "",
        "skills": resume.primary_skills,              # ✅ FIXED HERE
        "locations": resume.locations,                # ✅ FIXED HERE
        "experience": resume.total_experience_years,  # ✅ FIXED HERE
    }

    jd_metadata = jd.metadata if isinstance(jd.metadata, dict) else {}

    # ==========================================================
    # FINAL RESPONSE
    # ==========================================================

    response = MatchScoreResponse(
        message="Contest processed successfully",
        data=ScoringDataContainer(
            finalScores=FinalScoresEnvelope(
                snsScoringResult=SnsScoringResult(
                    resumeMatch=ResumeMatchBlock(
                        OverallScore=round(overall_score, 1),
                        scoreOutOf=100,
                    ),
                    scoreBreakdown=ScoreBreakdownBlock(
                        primarySkillsMatch=SkillsMatchBlock(
                            score=round(primary_score, 1),
                            weight=STRUCTURAL_WEIGHTS["primary_skills"],
                            matchedSkills=matched_primary,
                            missingSkills=missing_primary,
                            explanation=(
                                f"Candidate matched "
                                f"{len(matched_primary)} out of "
                                f"{len(jd.primary_skills)} primary skills."
                            ),
                            requirements=jd.primary_skills,
                        ),
                        goodToHave=SkillsMatchBlock(
                            score=round(good_score, 1),
                            weight=STRUCTURAL_WEIGHTS["good_to_have"],
                            matchedSkills=matched_good,
                            missingSkills=missing_good,
                            explanation=(
                                f"Candidate matched "
                                f"{len(matched_good)} out of "
                                f"{len(jd.good_to_have_skills)} "
                                f"good-to-have skills."
                            ),
                            requirements=jd.good_to_have_skills,
                        ),
                        experienceMatch=ExperienceMatchBlock(
                            score=round(experience_score, 1),
                            weight=STRUCTURAL_WEIGHTS["experience"],
                            requiredExperience=required_experience,
                            candidateExperience=candidate_experience,
                            explanation=experience_explanation,
                        ),
                        locationMatch=LocationMatchBlock(
                            score=round(location_score, 1),
                            weight=STRUCTURAL_WEIGHTS["location"],
                            jobLocationRequirement=job_location,
                            candidateLocation=candidate_location,
                            matched=location_match_pct > 0,
                        ),
                    ),
                ),
            ),

            # ✅ FIXED OUTPUT (NO MORE EMPTY SKILLS)
            jdMetadata=jd_metadata,
            resumeMetadata=resume_metadata,
        ),
    )

    # ==========================================================
    # DEBUG / OBSERVABILITY
    # ==========================================================

    response._semantic_score = round(semantic_score, 2)
    response._contextual_semantic_score = round(contextual_semantic_score, 2)
    response._contextual_matches = contextual_matches
    response._contextual_explanation = contextual_explanation

    return response