"""
Database Service Layer for Match Score API
Handles all MongoDB operations using Motor async client
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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

    @property
    def inhouse_resume_parser_db(self):
        # ⚡ FIXED: Pointing directly to the Marketplace DB since separate DB doesn't exist
        return self.client[settings.mongodb_db_marketplace]

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

        # ⚡ FIXED: Pointing primary target directly to Marketplace.jobSeekerProfile
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
    async def get_cv_meta_metadata(self, contest_id: str, recruiter_id: str, js_id: str) -> str:
        target_contest_id = self._normalize_and_convert_id(contest_id)
        target_recruiter_id = self._normalize_and_convert_id(recruiter_id)
        target_js_id = self._normalize_and_convert_id(js_id)

        profile_doc = await self.marketplace_db.recruiterAddProfiles.find_one({
            "contestId": target_contest_id,
            "recruiterId": target_recruiter_id,
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
                if item_js_id != target_js_id:
                    continue

                cv_data = item.get("cv") or item.get("resume") or item.get("cvUrl")

                if isinstance(cv_data, dict):
                    return cv_data.get("url") or cv_data.get("secureUrl") or cv_data.get("s3Url") or ""
                return str(cv_data or "")

        raise self.DocumentNotFoundError("CV metadata missing inside profile map")

    # ==========================================================
    # CACHE FUNCTIONS
    # ==========================================================
    async def get_jd_cache(self, contest_id: str) -> dict:
        target_contest_id = self._normalize_and_convert_id(contest_id)

        cache_doc = await self.marketplace_db.Jd_Embeddings.find_one(
            {"contestId": {"$in": [target_contest_id, str(target_contest_id)]}},
            {"jd_embeddings": 1, "jd_raw_text": 1, "_id": 0}
        )

        return {
            "embedding": cache_doc.get("jd_embeddings") if cache_doc else None,
            "raw_text": cache_doc.get("jd_raw_text", "") if cache_doc else ""
        }

    async def cache_jd_data(self, contest_id: str, embeddings: List[float], raw_text: str) -> None:
        target_contest_id = self._normalize_and_convert_id(contest_id)

        await self.marketplace_db.Jd_Embeddings.update_one(
            {"contestId": target_contest_id},
            {
                "$set": {
                    "jd_embeddings": embeddings,
                    "jd_raw_text": raw_text,
                    "updatedAt": datetime.now(timezone.utc)
                },
                "$setOnInsert": {"createdAt": datetime.now(timezone.utc)}
            },
            upsert=True
        )

    async def get_cv_cache(self, contest_id: str, js_id: str) -> dict:
        target_contest_id = self._normalize_and_convert_id(contest_id)
        target_js_id = self._normalize_and_convert_id(js_id)

        cache_doc = await self.marketplace_db.Cv_Embeddings.find_one(
            {
                "contestId": {"$in": [target_contest_id, str(target_contest_id)]},
                "jsId": {"$in": [target_js_id, str(target_js_id)]}
            },
            {"cv_embeddings": 1, "cv_raw_text": 1, "_id": 0}
        )

        return {
            "embedding": cache_doc.get("cv_embeddings") if cache_doc else None,
            "raw_text": cache_doc.get("cv_raw_text", "") if cache_doc else ""
        }

    async def cache_cv_data(self, contest_id: str, js_id: str, embeddings: List[float], raw_text: str) -> None:
        target_contest_id = self._normalize_and_convert_id(contest_id)
        target_js_id = self._normalize_and_convert_id(js_id)

        await self.marketplace_db.Cv_Embeddings.update_one(
            {"contestId": target_contest_id, "jsId": target_js_id},
            {
                "$set": {
                    "cv_embeddings": embeddings,
                    "cv_raw_text": raw_text,
                    "updatedAt": datetime.now(timezone.utc)
                },
                "$setOnInsert": {"createdAt": datetime.now(timezone.utc)}
            },
            upsert=True
        )

    # ==========================================================
    # NEW MICRO-SKILL OBJECT EMBEDDING COMPONENT INTEGRATIONS
    # ==========================================================
    async def get_or_create_jd_skills_meta(
        self, semantic_engine: Any, contest_id: str, raw_jd_skills: List[str]
    ) -> Dict[str, List[float]]:
        """
        Ensures JD requirements are batch embedded and cached inside Jd_Embeddings record.
        Uses upsert mechanics to handle edge race condition states smoothly.
        """
        try:
            query_id = self._normalize_and_convert_id(contest_id)

            doc = await self.marketplace_db.Jd_Embeddings.find_one({"contestId": query_id})
            
            if doc and "jd_skills_embeddings" in doc and doc["jd_skills_embeddings"]:
                logger.info("✨ [MONGO JD SKILLS HIT] | Loaded vectors directly from Jd_Embeddings.")
                return doc["jd_skills_embeddings"]

            logger.info("🔍 [MONGO JD SKILLS MISS] | Generating bulk OpenAI vectors for JD skills")
            fresh_vectors = await semantic_engine.generate_bulk_skills_embeddings(raw_jd_skills)

            if fresh_vectors:
                await self.marketplace_db.Jd_Embeddings.update_one(
                    {"contestId": query_id},
                    {
                        "$set": {
                            "jd_skills_embeddings": fresh_vectors,
                            "updatedAt": datetime.now(timezone.utc)
                        },
                        "$setOnInsert": {"createdAt": datetime.now(timezone.utc)}
                    },
                    upsert=True
                )
                logger.info("💾 [MONGO JD SKILLS SAVE] | Injected embedded map into Jd_Embeddings.")

            return fresh_vectors

        except Exception as e:
            logger.error("❌ Database tracking failed inside get_or_create_jd_skills_meta: %s", e)
            return {}

    async def get_or_create_cv_skills_meta(
        self, semantic_engine: Any, contest_id: str, js_id: str, raw_cv_skills: List[str]
    ) -> Dict[str, List[float]]:
        """
        Ensures Candidate profile components are batch embedded and cached inside Cv_Embeddings record.
        Uses explicit compound targeting fields to align with baseline structural tracking layouts.
        """
        try:
            query_contest_id = self._normalize_and_convert_id(contest_id)
            query_js_id = self._normalize_and_convert_id(js_id)

            doc = await self.marketplace_db.Cv_Embeddings.find_one({
                "contestId": query_contest_id, 
                "jsId": query_js_id
            })
            
            if doc and "cv_skills_embeddings" in doc and doc["cv_skills_embeddings"]:
                logger.info("✨ [MONGO CV SKILLS HIT] | Loaded vectors directly from Cv_Embeddings.")
                return doc["cv_skills_embeddings"]

            logger.info("🔍 [MONGO CV SKILLS MISS] | Generating bulk OpenAI vectors for CV skills")
            fresh_vectors = await semantic_engine.generate_bulk_skills_embeddings(raw_cv_skills)

            if fresh_vectors:
                await self.marketplace_db.Cv_Embeddings.update_one(
                    {"contestId": query_contest_id, "jsId": query_js_id},
                    {
                        "$set": {
                            "cv_skills_embeddings": fresh_vectors,
                            "updatedAt": datetime.now(timezone.utc)
                        },
                        "$setOnInsert": {"createdAt": datetime.now(timezone.utc)}
                    },
                    upsert=True
                )
                logger.info("💾 [MONGO CV SKILLS SAVE] | Injected embedded map into Cv_Embeddings.")

            return fresh_vectors

        except Exception as e:
            logger.error("❌ Database tracking failed inside get_or_create_cv_skills_meta: %s", e)
            return {}

    async def close(self) -> None:
        if self.client:
            self.client.close()


database_service = DatabaseService()