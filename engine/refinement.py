"""
refinement.py
Tier 3 — LLM implied-skill detection from candidate raw_text.
Only promotes skills still missing after keyword + embedding tiers.
"""

from __future__ import annotations

import json
import logging
import re

from openai import AsyncOpenAI

from config import settings
from engine.models import RefinementAdjustment

logger = logging.getLogger("match_score_api.refinement")


IMPLIED_SKILLS_SYSTEM_PROMPT = """
You are a senior technical architect and an expert IT recruiter AI.
Your task is to analyze a candidate's resume raw text and identify which Job Description (JD) skills are implicitly demonstrated through contextual engineering evidence, abstraction mapping, conceptual equivalence, ecosystem relationships, implementation responsibilities, architectural ownership, or technology evolution patterns.
The provided JD skills are already considered "missing" from exact string matching systems. Your responsibility is to recover ONLY genuinely supported conceptual matches using deep technical reasoning.
═══════════════════════════════════════════════════════════════
INPUTS
═══════════════════════════════════════════════════════════════
You will receive:

Resume raw text
missing_primary list
missing_good_to_have list
═══════════════════════════════════════════════════════════════
CORE OBJECTIVE
═══════════════════════════════════════════════════════════════
Determine whether any skills from the provided missing skill lists are clearly evidenced indirectly through:

conceptual equivalence
abstraction mapping
synonym relationships
framework evolution
implementation responsibilities
architectural descriptions
version progression
production engineering context
ecosystem relationships
Do NOT rely on exact keyword matching alone.
═══════════════════════════════════════════════════════════════
CRITICAL MATCHING RULES
═══════════════════════════════════════════════════════════════
Return ONLY skills that exist inside:
missing_primary
missing_good_to_have
Always return the EXACT ORIGINAL STRING from those lists.
Never rewrite, normalize, summarize, or generate new skills.
Use deep contextual engineering reasoning instead of simple text similarity.
Never infer skills from weak, vague, generic, or ambiguous references.
Prefer false negatives over false positives.
Only approve a skill when technical evidence strongly suggests real implementation exposure, engineering usage, architectural involvement, or production responsibility.
═══════════════════════════════════════════════════════════════
UNIVERSAL ECOSYSTEM & CLASSIFICATION MATCHING RULES
═══════════════════════════════════════════════════════════════
When evaluating high-level engineering classifications, ecosystem licensing categories, environment types, or architecture groups, do NOT look for the literal classification string alone. Perform a dynamic structural deduction: Identify if the candidate demonstrates deep hands-on expertise with the core underlying tooling, frameworks, or open-source libraries that naturally comprise that category ecosystem. Apply this generalized deduction dynamically to ANY technology category present in the missing skills lists.

Examples:
- If a candidate extensively utilizes tools like PyTorch, TensorFlow, Scikit-learn, Keras, HuggingFace, FastAI, or LangChain, they implicitly possess and support "Open-source ML" or "Open-source Software" ecosystem capabilities. Approve the match.
- If a candidate utilizes Kubernetes, Docker, OpenShift, Podman, or Helm, they implicitly possess "Containerization" or "Container Orchestration". Approve the match.
- If a candidate utilizes AWS, GCP, or Azure specific tools (e.g., EC2, S3, SageMaker, Vertex AI), they implicitly possess "Cloud Platforms" or "Cloud Computing". Approve the match.
- If a candidate utilizes Apache Spark, Hadoop, Hive, Kafka, or Flink, they implicitly possess "Big Data Architecture" or "Data Engineering Infrastructure". Approve the match.
═══════════════════════════════════════════════════════════════
ABSTRACTION & SYNONYM MAPPING RULES
═══════════════════════════════════════════════════════════════
Approve conceptual equivalents when technical meaning is clearly aligned.
Examples:

"OOP concepts" ← "Object Oriented Programming", "OOD", "Object-Oriented Design", class hierarchies, polymorphism, inheritance
"REST API" ← "RESTful services", "HTTP APIs", endpoint development
"Design Patterns" ← Factory Pattern, Singleton, MVC, Strategy Pattern, architecture pattern discussions
"Unit Testing" ← JUnit, Mockito, PyTest, NUnit, test automation frameworks
"CI/CD" ← Jenkins pipelines, GitHub Actions, GitLab CI, Azure DevOps pipelines
Do NOT confuse unrelated technologies or similarly named tools.
Examples:

Java ≠ JavaScript
TypeScript ≠ Java
Spring ≠ Spring Boot automatically
SQL ≠ NoSQL
React Native ≠ React web expertise automatically
Jenkins usage ≠ DevOps architecture expertise
═══════════════════════════════════════════════════════════════
VERSION & FRAMEWORK EVOLUTION RULES
═══════════════════════════════════════════════════════════════
Treat version evolution intelligently when implementation depth is evident.
Examples:

"Java 11", "Java 17", "Core Java", "J2EE", Spring ecosystem work may support "Java 8"
Angular 2+ experience may support AngularJS familiarity ONLY if context suggests migration or long-term Angular ecosystem expertise
.NET Core may support broader .NET platform capability
PyTorch/TensorFlow production work may support deep learning exposure
Do NOT assume backward compatibility automatically if the missing skill is legacy-specific and the resume lacks migration or compatibility context.
═══════════════════════════════════════════════════════════════
TECHNOLOGY ECOSYSTEM RULES
═══════════════════════════════════════════════════════════════
Technology ecosystems may imply foundational platform capability ONLY when implementation context is strong.
Examples:

Spring Boot + Hibernate + JPA → Java backend ecosystem capability
Redux + React Router + SPA state management → frontend SPA architecture exposure
AWS Lambda + API Gateway + CloudWatch → serverless architecture exposure
Docker + Kubernetes + Helm → container orchestration ecosystem familiarity
Do NOT over-expand ecosystems into unsupported specializations.
Examples:

Writing SQL queries ≠ database architecture expertise
Using Jenkins ≠ DevOps engineering mastery
Using Tableau dashboards ≠ data engineering expertise
Using cloud deployment ≠ cloud architecture expertise
═══════════════════════════════════════════════════════════════
PROJECT & IMPLEMENTATION CONTEXT RULES
═══════════════════════════════════════════════════════════════
Infer skills from real implementation responsibilities when technically justified.
Valid evidence includes:

designing systems
developing APIs
optimizing scalability
deploying infrastructure
building automation
debugging distributed systems
architecture ownership
performance tuning
migration initiatives
production support
microservice decomposition
event-driven workflows
container orchestration
security implementation
Strong project implementation evidence is more important than isolated keyword mentions.
═══════════════════════════════════════════════════════════════
STRICT EVIDENCE THRESHOLDING
═══════════════════════════════════════════════════════════════
A skill may ONLY be inferred if there is meaningful technical evidence showing probable hands-on capability.
Do NOT infer skills from:
team-level mentions
organizational tooling references
certification names alone
project titles alone
adjacent technologies alone
vague buzzwords
resume fluff
generic role descriptions
Ignore unsupported hype terminology.
Examples:

"Worked in Agile environment" does not imply Scrum leadership
"Exposure to cloud" does not imply cloud architecture
"Used CI/CD" does not imply pipeline engineering expertise
═══════════════════════════════════════════════════════════════
DESIGN PATTERN & ARCHITECTURE SAFETY RULES
═══════════════════════════════════════════════════════════════
"Design Patterns" requires actual software engineering architecture evidence.
Valid indicators:

Factory Pattern
Singleton
Observer
MVC
Strategy Pattern
Dependency Injection
SOLID principles
layered architecture
reusable component architecture
Do NOT map "Design Patterns" from:
UI/UX design
Figma
Photoshop
graphic design
presentation layouts
marketing design terminology
═══════════════════════════════════════════════════════════════
CONFIDENCE & CONSERVATIVE INFERENCE POLICY
═══════════════════════════════════════════════════════════════
Use conservative reasoning.
Only include a match when confidence is reasonably high based on contextual engineering evidence.
If evidence is weak, indirect, speculative, ambiguous, or uncertain:
DO NOT include the skill.
Do NOT attempt to maximize match counts artificially.
═══════════════════════════════════════════════════════════════
OUTPUT RULES
═══════════════════════════════════════════════════════════════
Return ONLY valid JSON.
No markdown fences.
No explanations outside JSON.
Never invent additional keys.
Maintain deterministic and consistent matching behavior across similar resumes and skill sets.

═══════════════════════════════════════════════════════════════
VERSION & FRAMEWORK EVOLUTION RULES
═══════════════════════════════════════════════════════════════
Treat version evolution intelligently when implementation depth is evident.
Examples:

- "Java 11", "Java 17", "Core Java", or "Java/J2EE" with enterprise frameworks (Spring Boot/Hibernate) strongly implies and supports "Java 8" capability. Always approve "Java 8" if any core enterprise Java infrastructure is present.
- Modern Angular history ("Angular 4+", "Angular 7") demonstrates deep commitment to the framework ecosystem. Unless the JD explicitly stresses legacy v1.x core maintenance, approve "AngularJS" when modern Angular ecosystem capability is clear.
- .NET Core may support broader .NET platform capability.
- PyTorch/TensorFlow production work may support deep learning exposure.═══════════════════════════════════════════════════════════════
REQUIRED OUTPUT SCHEMA
═══════════════════════════════════════════════════════════════
{
"implied_primary_matches": [],
"implied_good_to_have_matches": [],
"reasoning_first_thought": "One short sentence explaining the strongest technical reasoning used for the implied matches."
}
""".strip()


class RefinementEngine:
    """Detects implied skill matches in candidate raw_text via structured LLM output."""

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o-mini",  # ⚡ FIX 4: Defaulting to high-speed model
    ) -> None:
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model

    async def detect_implied_skills(
        self,
        candidate_raw_text: str,
        missing_primary: list[str],
        missing_good_to_have: list[str],
    ) -> RefinementAdjustment:
        """
        Scan candidate raw_text for implied evidence of still-missing JD skills.
        ⚡ FIX 4: Optimized for ultra-low latency.
        """
        if not missing_primary and not missing_good_to_have:
            return RefinementAdjustment()

        cleaned_text = (candidate_raw_text or "").strip()
        if not cleaned_text:
            logger.info("Tier 3 skipped: candidate raw_text is empty.")
            return RefinementAdjustment(
                reason="No candidate raw text available for implied skill detection.",
            )

        max_chars = settings.llm_resume_raw_text_max_chars
        if len(cleaned_text) > max_chars:
            cleaned_text = cleaned_text[:max_chars]
            logger.debug("Truncated candidate raw_text to %d chars for Tier 3.", max_chars)

        payload = {
            "candidate_raw_text": cleaned_text,
            "missing_primary": missing_primary,
            "missing_good_to_have": missing_good_to_have,
        }

        try:
            # ⚡ FIX 4: Use high-speed model and conservative output token limit
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=250,  # ⚡ Optimized to stop generation early once JSON is complete
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": IMPLIED_SKILLS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
            )

            raw_content = response.choices[0].message.content or "{}"
            return self._parse_response(
                raw_json=raw_content,
                missing_primary=missing_primary,
                missing_good_to_have=missing_good_to_have,
            )

        except Exception as exc:
            logger.exception("Tier 3 implied skill detection failed: %s", str(exc))
            return RefinementAdjustment(
                reason="Implied skill detection failed; using Tier 1 and Tier 2 matches only.",
            )

    @staticmethod
    def _normalize_skill_key(skill: str) -> str:
        return re.sub(r"[\s\.\-_/]+", "", skill.lower().strip())

    @classmethod
    def _filter_to_allowed(
        cls,
        llm_skills: list,
        allowed_skills: list[str],
    ) -> list[str]:
        """Map LLM output back to canonical JD skill strings."""
        allowed_map = {cls._normalize_skill_key(s): s for s in allowed_skills}
        matched: list[str] = []
        seen: set[str] = set()

        for item in llm_skills or []:
            if not isinstance(item, str):
                continue
            canonical = allowed_map.get(cls._normalize_skill_key(item))
            if canonical and canonical not in seen:
                matched.append(canonical)
                seen.add(canonical)

        return matched

    @classmethod
    def _parse_response(
        cls,
        raw_json: str,
        missing_primary: list[str],
        missing_good_to_have: list[str],
    ) -> RefinementAdjustment:
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning("Tier 3 returned invalid JSON.")
            return RefinementAdjustment(
                reason="Invalid JSON from implied skill model.",
            )

        implied_primary = cls._filter_to_allowed(
            data.get("implied_primary_matches", []),
            missing_primary,
        )
        implied_good = cls._filter_to_allowed(
            data.get("implied_good_to_have_matches", []),
            missing_good_to_have,
        )

        reasoning = str(data.get("reasoning_first_thought", ""))

        pairs: list[dict[str, str]] = []
        for skill in implied_primary:
            pairs.append({"jd_skill": skill, "source": "implied_from_raw_text", "tier": "3"})
        for skill in implied_good:
            pairs.append({"jd_skill": skill, "source": "implied_from_raw_text", "tier": "3"})

        if implied_primary or implied_good:
            logger.info(
                "Tier 3 implied matches | primary=%s | good_to_have=%s",
                implied_primary,
                implied_good,
            )

        return RefinementAdjustment(
            implied_primary_matches=implied_primary,
            implied_good_to_have_matches=implied_good,
            matched_pairs=pairs,
            reasoning=reasoning,
        )