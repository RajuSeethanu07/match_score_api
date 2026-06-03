"""
Database Service Layer for Match Score API
Handles all MongoDB operations using Motor async client
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorClient

from config import settings

logger = logging.getLogger(__name__)


class DatabaseService:
    """Production-ready async MongoDB service for Match Score API"""

    def __init__(self, mongo_uri: str = settings.mongodb_uri):
        if not mongo_uri:
            raise ValueError("MONGO_CLUSTER_URI must be provided")

        self.client: AsyncIOMotorClient = AsyncIOMotorClient(
            mongo_uri,
            maxPoolSize=50,
            minPoolSize=10,
            serverSelectionTimeoutMS=5000
        )

    class InvalidIdError(ValueError):
        pass

    class DocumentNotFoundError(LookupError):
        pass

    # ==========================================================
    # DB ACCESSORS
    # ==========================================================
    @property
    def marketplace_db(self):
        return self.client[settings.mongodb_db_marketplace]

    @property
    def inhouse_jd_parser_db(self):
        return self.client[settings.mongodb_db_jd_parser]

    # ==========================================================
    # ID NORMALIZER
    # ==========================================================
    def _normalize_and_convert_id(self, value: Any) -> ObjectId | str:
        if not value:
            return value

        if isinstance(value, ObjectId):
            return value

        clean_str = str(value).strip().replace('"', '').replace("'", "")

        try:
            return ObjectId(clean_str)
        except (InvalidId, ValueError):
            logger.warning("Invalid ObjectId: %s", clean_str)
            return clean_str

    # ==========================================================
    # JD FETCH (CLEAN DATA OVER WIRE)
    # ==========================================================
    async def get_parsed_jd(self, contest_id: str) -> dict:
        target_contest_id = self._normalize_and_convert_id(contest_id)

        if not isinstance(target_contest_id, ObjectId):
            raise self.InvalidIdError(f"Invalid contestId format: {contest_id}")

        contest_doc = await self.marketplace_db.contests.find_one(
            {"contestId": target_contest_id},
            {"jdId": 1, "jd_id": 1, "jdkey": 1, "jdKey": 1, "jdUrl": 1, "details": 1, "_id": 0}
        )

        if not contest_doc:
            raise self.DocumentNotFoundError("Contest config not found")

        raw_jd_id = contest_doc.get("jdId") or contest_doc.get("jd_id")
        target_jd_id = self._normalize_and_convert_id(raw_jd_id)

        parsed_jd_doc = await self.inhouse_jd_parser_db.records.find_one(
            {"_id": target_jd_id}
        )

        if not parsed_jd_doc:
            raise self.DocumentNotFoundError("Parsed JD not found")

        parsed_jd_doc["details"] = contest_doc.get("details", {}) or {}
        parsed_jd_doc["jdKey"] = (
            contest_doc.get("jdKey")
            or contest_doc.get("jdkey")
            or contest_doc.get("jdUrl")
            or ""
        )
        parsed_jd_doc["jdUrl"] = contest_doc.get("jdUrl") or ""

        return parsed_jd_doc

    # ==========================================================
    # RESUME FETCH (CLEAN RAW DATA OVER WIRE)
    # ==========================================================
    async def get_parsed_resume(self, js_id: str) -> dict:
        target_js_id = self._normalize_and_convert_id(js_id)

        if not isinstance(target_js_id, ObjectId):
            raise self.InvalidIdError(f"Invalid jsId format: {js_id}")

        resume_doc = await self.marketplace_db.jobSeekerProfile.find_one(
            {"_id": target_js_id}
        )

        if not resume_doc:
            raise self.DocumentNotFoundError(f"Profile data completely missing for candidate id: {js_id}")

        logger.info("JOB SEEKER PROFILE FETCHED SUCCESSFULLY | jsId=%s", js_id)
        return resume_doc

    # ==========================================================
    # CV METADATA
    # ==========================================================
    async def get_cv_meta_metadata(self, contest_id: str, js_id: str) -> str:
        """
        Fetches the CV URL from recruiterAddProfiles strictly mapping 
        contestId and nested jobseekerDetails.jsId without recruiterId filter.
        """
        target_contest_id = self._normalize_and_convert_id(contest_id)
        target_js_id = self._normalize_and_convert_id(js_id)

        profile_doc = await self.marketplace_db.recruiterAddProfiles.find_one({
            "contestId": target_contest_id,
            "jobseekerDetails.jsId": target_js_id
        })

        if not profile_doc:
            raise self.DocumentNotFoundError("Recruiter profile mapping not found")

        root_cv = profile_doc.get("cv")
        if root_cv:
            if isinstance(root_cv, dict):
                return root_cv.get("url") or root_cv.get("secureUrl") or root_cv.get("s3Url") or ""
            return str(root_cv)

        jobseeker_details = profile_doc.get("jobseekerDetails", [])
        if isinstance(jobseeker_details, list):
            for item in jobseeker_details:
                if not isinstance(item, dict):
                    continue

                item_js_id = self._normalize_and_convert_id(item.get("jsId"))
                if str(item_js_id) != str(target_js_id):
                    continue

                cv_data = item.get("cv") or item.get("resume") or item.get("cvUrl")

                if isinstance(cv_data, dict):
                    return cv_data.get("url") or cv_data.get("secureUrl") or cv_data.get("s3Url") or ""
                return str(cv_data or "").strip()

        raise self.DocumentNotFoundError("CV metadata missing inside profile map array")

    # ==========================================================
    # MICRO-SKILL OBJECT EMBEDDING COMPONENT INTEGRATIONS
    # ==========================================================
    async def get_or_create_jd_skills_meta(
        self,
        semantic_engine: Any,
        contest_id: str,
        raw_jd_skills: List[str]
    ) -> Dict[str, List[float]]:
        """
        Ensures JD skill embeddings are generated once and cached.
        Saves an explicit null marker to the DB if no skills exist to prevent redundant runs.
        """
        try:
            query_id = self._normalize_and_convert_id(contest_id)

            doc = await self.marketplace_db.Jd_Embeddings.find_one(
                {"contestId": query_id},
                {"jd_skills_embeddings": 1}
            )

            if doc and "jd_skills_embeddings" in doc:
                logger.info("✨ [MONGO JD SKILLS HIT] | contestId=%s", contest_id)
                cached_vectors = doc.get("jd_skills_embeddings")
                # Downstream execution layer safety check
                return cached_vectors if isinstance(cached_vectors, dict) else {}

            fresh_vectors = None
            if raw_jd_skills:
                logger.info("🔍 [MONGO JD SKILLS MISS] | Generating bulk OpenAI vectors")
                fresh_vectors = await semantic_engine.generate_bulk_skills_embeddings(raw_jd_skills)
            else:
                logger.warning("⚠️ No JD skills available. Saving null cache marker.")
                fresh_vectors = None  # Saves as JSON null in MongoDB

            now = datetime.now(timezone.utc)
            await self.marketplace_db.Jd_Embeddings.update_one(
                {"contestId": query_id},
                {
                    "$set": {
                        "jd_skills_embeddings": fresh_vectors,
                        "updatedAt": now
                    },
                    "$setOnInsert": {
                        "createdAt": now
                    }
                },
                upsert=True
            )
            
            return fresh_vectors if fresh_vectors is not None else {}

        except Exception as e:
            logger.exception("❌ get_or_create_jd_skills_meta failed: %s", e)
            return {}

    async def get_or_create_cv_skills_meta(
        self,
        semantic_engine: Any,
        contest_id: str,
        js_id: str,
        raw_cv_skills: List[str]
    ) -> Dict[str, List[float]]:
        """
        Ensures CV skill embeddings are generated once and cached.
        Saves an explicit null marker to the DB if no skills exist to prevent redundant runs.
        """
        try:
            query_contest_id = self._normalize_and_convert_id(contest_id)
            query_js_id = self._normalize_and_convert_id(js_id)

            doc = await self.marketplace_db.Cv_Embeddings.find_one(
                {
                    "contestId": query_contest_id,
                    "jsId": query_js_id
                },
                {"cv_skills_embeddings": 1}
            )

            if doc and "cv_skills_embeddings" in doc:
                logger.info("✨ [MONGO CV SKILLS HIT] | contestId=%s jsId=%s", contest_id, js_id)
                cached_vectors = doc.get("cv_skills_embeddings")
                # Downstream execution layer safety check
                return cached_vectors if isinstance(cached_vectors, dict) else {}

            fresh_vectors = None
            if raw_cv_skills:
                logger.info("🔍 [MONGO CV SKILLS MISS] | Generating bulk OpenAI vectors")
                fresh_vectors = await semantic_engine.generate_bulk_skills_embeddings(raw_cv_skills)
            else:
                logger.warning("⚠️ No CV skills available for jsId=%s. Saving null cache marker.", js_id)
                fresh_vectors = None  # Saves as JSON null in MongoDB

            now = datetime.now(timezone.utc)
            await self.marketplace_db.Cv_Embeddings.update_one(
                {
                    "contestId": query_contest_id,
                    "jsId": query_js_id
                },
                {
                    "$set": {
                        "cv_skills_embeddings": fresh_vectors,
                        "updatedAt": now
                    },
                    "$setOnInsert": {
                        "createdAt": now
                    }
                },
                upsert=True
            )
            
            return fresh_vectors if fresh_vectors is not None else {}

        except Exception as e:
            logger.exception("❌ get_or_create_cv_skills_meta failed: %s", e)
            return {}

    async def close(self) -> None:
        if self.client:
            self.client.close()


database_service = DatabaseService()