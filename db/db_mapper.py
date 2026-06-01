from __future__ import annotations
import logging
import re
from typing import Any
from engine.models import ParsedJd, ParsedResume

logger = logging.getLogger(__name__)


class DBMapper:

    # ==========================================================
    # SAFE UTILITIES
    # ==========================================================
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip().split("+")[0].strip())
        except Exception:
            return default

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(v).strip() for v in value if v)
        return str(value).strip()

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()

    # ==========================================================
    # SAFE RESUME ROOT HANDLING (UPDATED FOR DEEP NESTING)
    # ==========================================================
    @staticmethod
    def _get_resume_root(doc: dict) -> dict:
        if not isinstance(doc, dict):
            return {}

        # Step 1: Handle completeResumeDetails encapsulation wrapper
        complete_details = doc.get("completeResumeDetails")
        if isinstance(complete_details, dict):
            parsed = complete_details.get("parsed_resume")
        else:
            parsed = doc.get("parsed_resume")

        if isinstance(parsed, dict):
            # Case 1: deeply wrapped parser payload structure
            if isinstance(parsed.get("ResumeParserData"), dict):
                return parsed["ResumeParserData"]

            # Case 2: flattened dictionary schema
            return parsed

        return {}

    # ==========================================================
    # RESUME SKILLS
    # ==========================================================
    @staticmethod
    def _extract_resume_skills(parsed_resume_data: dict) -> list[str]:
        extracted: set[str] = set()

        if not isinstance(parsed_resume_data, dict):
            return []

        skill_block = parsed_resume_data.get("SkillBlock")
        if isinstance(skill_block, str):
            # Clean trailing periods and split by comma
            extracted.update(s.strip().rstrip(".") for s in skill_block.split(",") if s.strip())

        legacy = parsed_resume_data.get("SegregatedSkill", [])
        if isinstance(legacy, list):
            for item in legacy:
                if isinstance(item, dict):
                    skill = item.get("Skill") or item.get("FormattedName")
                    if skill:
                        extracted.add(str(skill).strip())

        experiences = parsed_resume_data.get("SegregatedExperience", [])
        if isinstance(experiences, list):
            for exp in experiences:
                if not isinstance(exp, dict):
                    continue

                related = exp.get("RelatedSkills")

                if isinstance(related, list):
                    extracted.update(str(s).strip() for s in related if s)
                elif isinstance(related, str):
                    extracted.update(s.strip() for s in related.split(",") if s.strip())

                projects = exp.get("Projects", [])
                if isinstance(projects, list):
                    for proj in projects:
                        if not isinstance(proj, dict):
                            continue

                        used = proj.get("UsedSkills")

                        if isinstance(used, list):
                            extracted.update(str(s).strip() for s in used if s)
                        elif isinstance(used, str):
                            extracted.update(s.strip() for s in used.split(",") if s.strip())

        return sorted(extracted)

    # ==========================================================
    # JD SKILLS
    # ==========================================================
    @staticmethod
    def _extract_jd_skills(skills: Any) -> list[str]:
        if isinstance(skills, str):
            return [s.strip() for s in skills.split(",") if s.strip()]
        if isinstance(skills, list):
            return [str(s).strip() for s in skills if str(s).strip()]
        return []

    # ==========================================================
    # JD CONTEXTS (SAFE)
    # ==========================================================
    @classmethod
    def _extract_jd_contexts(cls, parsed_data: dict) -> list[str]:
        contexts = []
        try:
            reqs = parsed_data.get("job_requirements") or {}

            fields = [
                parsed_data.get("jobOverview"),
                parsed_data.get("keyResponsibilities"),
                reqs.get("job_summary"),
                reqs.get("must_have"),
                reqs.get("good_to_have"),
                reqs.get("responsibilities"),
                reqs.get("required_skills"),
                reqs.get("education"),
                reqs.get("job_title_or_role"),
            ]

            for field in fields:
                if isinstance(field, str) and field.strip():
                    contexts.append(cls._clean_text(field))
                elif isinstance(field, list):
                    for item in field:
                        if item:
                            contexts.append(cls._clean_text(str(item)))

            return list(dict.fromkeys(contexts))

        except Exception as e:
            logger.exception(e)
            return []

    # ==========================================================
    # RESUME CONTEXTS
    # ==========================================================
    @classmethod
    def _extract_resume_experience_contexts(cls, parsed_resume_data: dict) -> list[str]:
        contexts = []
        experiences = parsed_resume_data.get("SegregatedExperience", [])

        if not isinstance(experiences, list):
            return []

        for exp in experiences:
            if not isinstance(exp, dict):
                continue

            # Safe extraction out of inner dict structures like Employer and JobProfile
            employer_data = exp.get("Employer")
            employer = employer_data.get("EmployerName") if isinstance(employer_data, dict) else ""
            
            profile_data = exp.get("JobProfile")
            designation = profile_data.get("Title") if isinstance(profile_data, dict) else ""
            related_skills = profile_data.get("RelatedSkills") if isinstance(profile_data, dict) else ""

            parts = [
                cls._normalize_text(designation or exp.get("Designation")),
                cls._normalize_text(employer or exp.get("Employer")),
                cls._normalize_text(exp.get("JobDescription") or exp.get("JobProfile") or exp.get("Summary")),
                cls._normalize_text(related_skills or exp.get("RelatedSkills")),
            ]

            text = cls._clean_text(" | ".join([p for p in parts if p]))
            if text:
                contexts.append(text)

            projects = exp.get("Projects", [])
            if isinstance(projects, list):
                for proj in projects:
                    if not isinstance(proj, dict):
                        continue

                    proj_text = cls._clean_text(" | ".join([
                        cls._normalize_text(proj.get("ProjectName")),
                        cls._normalize_text(proj.get("ProjectDescription")),
                        cls._normalize_text(proj.get("UsedSkills")),
                    ]))

                    if proj_text:
                        contexts.append(proj_text)

        return list(dict.fromkeys(contexts))

    # ==========================================================
    # RESUME LOCATIONS
    # ==========================================================
    @staticmethod
    def _extract_resume_locations(parsed: dict, doc: dict) -> list[str]:
        locations = set()

        addresses = parsed.get("Address", [])
        if isinstance(addresses, list):
            for addr in addresses:
                if isinstance(addr, dict):
                    if addr.get("City"):
                        locations.add(addr["City"].strip())
                    if addr.get("State"):
                        locations.add(addr["State"].strip())

        experiences = parsed.get("SegregatedExperience", [])
        if isinstance(experiences, list):
            for exp in experiences:
                loc = exp.get("Location")
                if isinstance(loc, dict):
                    if loc.get("City"):
                        locations.add(loc["City"].strip())

        salary = doc.get("salary_insights", {}).get("preferredLocation", [])
        if isinstance(salary, list):
            for loc in salary:
                if isinstance(loc, dict) and loc.get("label"):
                    locations.add(loc["label"].strip())

        return sorted(locations)

    # ==========================================================
    # JD MAPPING
    # ==========================================================
    @classmethod
    def map_parsed_jd(cls, doc: dict, raw_text: str = "") -> ParsedJd:

        parsed = doc.get("parsed_jobdescription") or {}
        reqs = parsed.get("job_requirements") or {}

        job_title = cls._normalize_text(reqs.get("job_title_or_role"))
        location = cls._normalize_text(reqs.get("work_location"))

        must = cls._extract_jd_skills(reqs.get("must_have"))
        good = cls._extract_jd_skills(reqs.get("good_to_have"))

        final_raw_text = (
            raw_text
            or doc.get("jd_raw_text")
            or doc.get("Jd_Embeddings", {}).get("jd_raw_text")
            or ""
        ).strip()

        if not final_raw_text:
            final_raw_text = cls._normalize_text(parsed.get("jobOverview"))

        semantic_contexts = cls._extract_jd_contexts(parsed)

        return ParsedJd(
            raw_text=final_raw_text,
            primary_skills=must,
            good_to_have_skills=good,
            min_experience_years=cls._safe_float(reqs.get("years_experience_minimum")),
            max_experience_years=cls._safe_float(reqs.get("years_experience_maximum")),
            locations=[location] if location else [],
            semantic_contexts=semantic_contexts,
            job_title=job_title,
            company_name=cls._normalize_text(doc.get("companyName")),
            metadata={
                "mongo_id": str(doc.get("_id") or ""),
                "title": job_title,
                "must_have": must,
                "good_to_have": good,
                "location": location,
            },
        )

    # ==========================================================
    # RESUME MAPPING
    # ==========================================================
    @classmethod
    def map_parsed_resume(cls, doc: dict, raw_text: str = "") -> ParsedResume:

        parsed = cls._get_resume_root(doc)
        personal = doc.get("personal_info") or {}

        # Handle potential nesting for cases where personal_info lives inside completeResumeDetails
        if not personal and isinstance(doc.get("completeResumeDetails"), dict):
            personal = doc["completeResumeDetails"].get("personal_info") or {}

        # Fallback to structural parsed values if personal_info overrides are completely absent
        first = cls._normalize_text(personal.get("firstName") or parsed.get("Name", {}).get("FirstName", ""))
        last = cls._normalize_text(personal.get("lastName") or parsed.get("Name", {}).get("LastName", ""))
        
        candidate_name = f"{first} {last}".strip()
        if not candidate_name or candidate_name == "Unknown":
            candidate_name = cls._normalize_text(parsed.get("Name", {}).get("FullName", "Unknown"))

        experience = cls._safe_float(personal.get("totalExperience"))

        final_raw_text = (
            raw_text
            or doc.get("cv_raw_text")
            or doc.get("Cv_embeddings", {}).get("cv_raw_text")
            or ""
        ).strip()

        skills = cls._extract_resume_skills(parsed)
        semantic_contexts = cls._extract_resume_experience_contexts(parsed)
        locations = cls._extract_resume_locations(parsed, doc)

        if not final_raw_text:
            # Fallback to joining all parsed job profile/descriptions contexts into structural text block
            final_raw_text = " ".join(semantic_contexts) if semantic_contexts else cls._normalize_text(parsed.get("RawText"))

        email = cls._normalize_text(personal.get("email"))
        emails = parsed.get("Email", [])
        if not email and isinstance(emails, list) and emails:
            email = emails[0].get("EmailAddress", "")

        phone = cls._normalize_text(personal.get("phoneNumber"))
        phones = parsed.get("PhoneNumber", [])
        if not phone and isinstance(phones, list) and phones:
            phone = phones[0].get("FormattedNumber") or phones[0].get("Number") or ""

        # Safe evaluation checking against both structural and misspelled variations
        education = cls._normalize_text(personal.get("educationalQualifucation") or personal.get("educationalQualification"))

        return ParsedResume(
            raw_text=final_raw_text,
            primary_skills=skills,
            semantic_contexts=semantic_contexts,
            total_experience_years=experience,
            experience_string_display=f"{experience} years" if experience else "Not specified",
            locations=locations,
            candidate_name=candidate_name,
            email=email,
            phone=phone,
            education=education,
            metadata={
                "mongo_id": str(doc.get("_id") or ""),
                "first_name": first,
                "skills": skills,
                "locations": locations,
                "experience": experience,
            },
        )


db_mapper = DBMapper()