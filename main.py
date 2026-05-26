"""
main.py
FastAPI gateway application controller for the Match Score engine.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, TYPE_CHECKING

import uvicorn
from fastapi import APIRouter, BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from db.db_mapper import db_mapper
from db.db_service import DatabaseService
from engine.score import MatchScoreEngine
from schema import MatchScoreRequest, MatchScoreResponse

if TYPE_CHECKING:
    from engine.models import ParsedJd, ParsedResume

# --- Setup Logging & Exception Handling ---
def _quiet_shutdown(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, (KeyboardInterrupt, asyncio.CancelledError)):
        return
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = _quiet_shutdown
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- S3 Parser Initialization ---
try:
    from s3_parser import s3_parser as global_s3_parser
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

# --- Application Lifespan ---
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

# --- Scoring Endpoint ---
@router.post("/match-score/", response_model=MatchScoreResponse)
async def match_score(
    payload: MatchScoreRequest, 
    background_tasks: BackgroundTasks, 
    request: Request
):
    svc: AppState = request.app.state.services
    logger.info("MATCH SCORE REQUEST RECEIVED | contestId=%s | jsId=%s", payload.contestId, payload.jsId)

    # 1. Load Data
    jd_cache_task = svc.db_service.get_jd_cache(payload.contestId)
    cv_cache_task = svc.db_service.get_cv_cache(payload.contestId, payload.jsId)
    parsed_jd_task = svc.db_service.get_parsed_jd(payload.contestId)
    parsed_resume_task = svc.db_service.get_parsed_resume(payload.jsId)

    jd_cache, cv_cache, parsed_jd_doc, js_doc = await asyncio.gather(
        jd_cache_task, cv_cache_task, parsed_jd_task, parsed_resume_task
    )

    cached_jd_vec = jd_cache.get("embedding")
    cached_cv_vec = cv_cache.get("embedding")
    jd_raw_text = jd_cache.get("raw_text", "") or ""
    cv_raw_text = cv_cache.get("raw_text", "") or ""

    # 2. Conditional S3 Deep Scan
    if settings.s3_deep_scan_enabled:
        tasks, labels = [], []

        if not jd_raw_text:
            jd_details = parsed_jd_doc.get("details", {})
            jd_key = (
                parsed_jd_doc.get("jdKey") or parsed_jd_doc.get("jdkey") or 
                parsed_jd_doc.get("jdUrl") or jd_details.get("jdKey") or 
                jd_details.get("jdkey") or jd_details.get("jdUrl") or ""
            )
            if jd_key:
                tasks.append(svc.s3_parser.extract_text(jd_key, "jd"))
                labels.append("jd")

        if not cv_raw_text:
            try:
                cv_url = await svc.db_service.get_cv_meta_metadata(
                    contest_id=payload.contestId, 
                    recruiter_id=payload.recruiterId, 
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
            
            if isinstance(data.get("jd"), str): jd_raw_text = data["jd"]
            if isinstance(data.get("cv"), str): cv_raw_text = data["cv"]

    # 3. Mapping & Engine Execution
    parsed_jd = db_mapper.map_parsed_jd(parsed_jd_doc, raw_text=jd_raw_text)
    parsed_resume = db_mapper.map_parsed_resume(js_doc, raw_text=cv_raw_text)

    response, generated_jd_vec, generated_cv_vec = await svc.score_engine.score(
        jd=parsed_jd, 
        resume=parsed_resume, 
        existing_jd_embedding=cached_jd_vec, 
        existing_resume_embedding=cached_cv_vec
    )

    # 4. Cache Update
    try:
        if (not cached_jd_vec and generated_jd_vec) or (not jd_cache.get("raw_text") and jd_raw_text):
            await svc.db_service.cache_jd_data(
                contest_id=payload.contestId, 
                embeddings=generated_jd_vec or cached_jd_vec or [], 
                raw_text=jd_raw_text
            )
            
        if (not cached_cv_vec and generated_cv_vec) or (not cv_cache.get("raw_text") and cv_raw_text):
            await svc.db_service.cache_cv_data(
                contest_id=payload.contestId, 
                js_id=payload.jsId, 
                embeddings=generated_cv_vec or cached_cv_vec or [], 
                raw_text=cv_raw_text
            )
    except Exception as e: 
        logger.exception("FAILED TO CACHE EMBEDDINGS/TEXT: %s", str(e))

    return response

# --- App Factory & Entry Point ---
def create_app() -> FastAPI:
    app = FastAPI(title="Match Score API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, 
        allow_origins=["*"], 
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