# schema.py
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field

# 🚀 UPDATED: Removed recruiterId as the pipeline now relies strictly 
# on the contest-to-jobseeker mapping for data resolution.
class MatchScoreRequest(BaseModel):
    contestId: str
    jsId: str

class SkillsMatchBlock(BaseModel):
    score: float
    weight: float
    matchedSkills: list[str] = Field(default_factory=list)
    missingSkills: list[str] = Field(default_factory=list)
    explanation: str
    requirements: list[str] = Field(default_factory=list)

class ExperienceMatchBlock(BaseModel):
    score: float
    weight: float
    requiredExperience: str
    candidateExperience: str
    explanation: str

class LocationMatchBlock(BaseModel):
    score: float
    weight: float
    jobLocationRequirement: str
    candidateLocation: str
    matched: bool

class ScoreBreakdownBlock(BaseModel):
    primarySkillsMatch: SkillsMatchBlock
    goodToHave: SkillsMatchBlock
    experienceMatch: ExperienceMatchBlock
    locationMatch: LocationMatchBlock

class ResumeMatchBlock(BaseModel):
    OverallScore: float
    scoreOutOf: int = 100

class SnsScoringResult(BaseModel):
    resumeMatch: ResumeMatchBlock
    scoreBreakdown: ScoreBreakdownBlock

class FinalScoresEnvelope(BaseModel):
    snsScoringResult: SnsScoringResult

class ScoringDataContainer(BaseModel):
    finalScores: FinalScoresEnvelope
    jdMetadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Audit tracking metadata dictionary retrieved from the InhouseJdParser db in the records collection"
    )
    resumeMetadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Audit tracking metadata dictionary retrieved from the InhouseResumeParser db in the records collection"
    )

class MatchScoreResponse(BaseModel):
    message: str = "Contest processed successfully"
    data: ScoringDataContainer