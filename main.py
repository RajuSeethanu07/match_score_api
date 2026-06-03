"""
main.py
FastAPI gateway application controller for the Match Score engine.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, TYPE_CHECKING

import uvicorn
from fastapi import APIRouter, BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from db.db_mapper import db_mapper, DBMapper
from db.db_service import DatabaseService
from engine.score import MatchScoreEngine
from engine.models import ParsedJd, ParsedResume
from schema import MatchScoreRequest, MatchScoreResponse

# --- Logging Safety ---
def _quiet_shutdown(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, (KeyboardInterrupt, asyncio.CancelledError)):
        return
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = _quiet_shutdown
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- S3 Parser ---
try:
    from s3_parser import s3_parser as global_s3_parser
    logger.info("S3 PARSER IMPORTED SUCCESSFULLY")
except Exception as e:
    logger.exception("FAILED TO IMPORT S3 PARSER: %s", str(e))

    class MockS3Parser:
        async def extract_text(self, s3_url: str, label: str = "") -> str:
            return ""
        async def close(self) -> None:
            pass

    global_s3_parser = MockS3Parser()


class AppState:
    db_service: DatabaseService
    s3_parser: Any
    score_engine: MatchScoreEngine


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState()
    state.db_service = DatabaseService(settings.mongodb_uri)
    state.s3_parser = global_s3_parser
    state.score_engine = MatchScoreEngine()

    app.state.services = state
    logger.info("APPLICATION SERVICES INITIALIZED")

    yield

    await state.db_service.close()
    await state.s3_parser.close()
    logger.info("APPLICATION SERVICES CLOSED")


router = APIRouter(tags=["scoring"])


# ==========================================================
# SAFE TYPE GUARD HELPERS (PRODUCTION PATCH)
# ==========================================================
def _convert_to_dict(doc: Any) -> dict:
    """
    Robust internal utility to convert objects into standard Python dictionaries.
    Handles Pydantic v1/v2 models, slotted/non-slotted dataclasses, and custom objects.
    """
    if doc is None:
        return {}
    if isinstance(doc, dict):
        return doc

    # 1. Pydantic Model Check (V2 model_dump or V1 dict)
    if hasattr(doc, "model_dump") and callable(getattr(doc, "model_dump")):
        return doc.model_dump()
    if hasattr(doc, "dict") and callable(getattr(doc, "dict")):
        return doc.dict()

    # 2. Standard Dataclass Check
    if is_dataclass(doc):
        return asdict(doc)

    # 3. Custom Slotted / Attribute Fallback
    if hasattr(doc, "__slots__"):
        return {slot: getattr(doc, slot) for slot in doc.__slots__ if hasattr(doc, slot)}
    if hasattr(doc, "__dict__"):
        return dict(doc.__dict__)

    logger.warning("Document type %s could not be safely parsed; returning empty dict", type(doc))
    return {}


def ensure_jd_dict(jd_doc: Any) -> dict:
    """
    Ensures DBMapper always receives a dict safely resolving
    potential slotted or Pydantic instances of ParsedJd.
    """
    return _convert_to_dict(jd_doc)


def ensure_resume_dict(res_doc: Any) -> dict:
    """
    Ensures DBMapper always receives a dict safely resolving
    potential slotted or Pydantic instances of ParsedResume.
    """
    return _convert_to_dict(res_doc)


# ==========================================================
# ENDPOINT
# ==========================================================
@router.post("/match-score/", response_model=MatchScoreResponse)
async def match_score(
    payload: MatchScoreRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    svc: AppState = request.app.state.services

    logger.info(
        "MATCH SCORE REQUEST RECEIVED | contestId=%s | jsId=%s",
        payload.contestId,
        payload.jsId
    )

    # 1. Load Data
    parsed_jd_task = svc.db_service.get_parsed_jd(payload.contestId)
    parsed_resume_task = svc.db_service.get_parsed_resume(payload.jsId)

    parsed_jd_doc, js_doc = await asyncio.gather(
        parsed_jd_task,
        parsed_resume_task
    )

    jd_raw_text = ""
    cv_raw_text = ""

    # 2. S3 Deep Scan
    if settings.s3_deep_scan_enabled:
        tasks, labels = [], []

        if not jd_raw_text:
            # Safely cast doc to dictionary before parsing fields
            clean_jd_dict = ensure_jd_dict(parsed_jd_doc)
            jd_details = clean_jd_dict.get("details", {}) or {}
            
            jd_key = (
                clean_jd_dict.get("jdKey")
                or clean_jd_dict.get("jdkey")
                or clean_jd_dict.get("jdUrl")
                or jd_details.get("jdKey")
                or jd_details.get("jdkey")
                or jd_details.get("jdUrl")
                or ""
            )

            if jd_key:
                tasks.append(svc.s3_parser.extract_text(jd_key, "jd"))
                labels.append("jd")

        if not cv_raw_text:
            try:
                # 🚀 UPDATED: Fetching CV URL strictly using contestId and jsId
                cv_url = await svc.db_service.get_cv_meta_metadata(
                    contest_id=payload.contestId,
                    js_id=payload.jsId
                )
                if cv_url:
                    tasks.append(svc.s3_parser.extract_text(cv_url, "cv"))
                    labels.append("cv")
            except Exception as ex:
                logger.exception("FAILED CV METADATA RESOLUTION: %s", str(ex))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            data = dict(zip(labels, results))

            if isinstance(data.get("jd"), str):
                jd_raw_text = data["jd"]
            if isinstance(data.get("cv"), str):
                cv_raw_text = data["cv"]

    # 3. INTERCEPT & SANITIZE DATA TYPE STRUCTURES
    parsed_jd_doc = ensure_jd_dict(parsed_jd_doc)
    js_doc = ensure_resume_dict(js_doc)

    parsed_jd = db_mapper.map_parsed_jd(parsed_jd_doc, raw_text=jd_raw_text)
    parsed_resume = db_mapper.map_parsed_resume(js_doc, raw_text=cv_raw_text)

    # 4. Scoring Engine Execution (Wired Up and Updated with Core State Context Mappings)
    response = await svc.score_engine.score(
        jd=parsed_jd,
        resume=parsed_resume,
        database_service=svc.db_service,  # 🌟 Forward db state management class instance
        contest_id=payload.contestId,     # 🌟 Forward core job tracking identifier
        js_id=payload.jsId                # 🌟 Forward candidate tracking identifier
    )

    return response


# ==========================================================
# APP FACTORY
# ==========================================================
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title, 
        version=settings.api_version, 
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    app.include_router(router, prefix="/api/v1")
    return app


if __name__ == "__main__":
    try:
        uvicorn.run(
            "main:create_app",
            host=settings.host,
            port=settings.port,
            workers=settings.workers,
            factory=True
        )
    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        pass