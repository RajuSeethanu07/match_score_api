"""
Production-Ready Configuration Management for Match Score API
Pydantic Settings v2 based environment configuration layer
"""

from __future__ import annotations
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized application configuration.
    Features: Pydantic v2 compatible, .env auto loading, typed validation, production defaults.
    """
    # 1. SYSTEM / RUNTIME CONFIGURATION
    DEBUG: bool = Field(default=False, validation_alias="DEBUG")
    ENV: str = Field(default="production", validation_alias="ENV")
    LOG_LEVEL: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    HOST: str = Field(default="0.0.0.0", validation_alias="HOST")
    PORT: int = Field(default=8000, validation_alias="PORT")
    WORKERS: int = Field(default=1, validation_alias="WORKERS")

    # 2. MONGODB CONFIGURATION
    MONGO_CLUSTER_URI: str = Field(..., validation_alias="MONGO_CLUSTER_URI")
    MONGO_MARKETPLACE_DB: str = Field(default="Marketplace", validation_alias="MONGO_MARKETPLACE_DB")
    MONGO_JD_PARSER_DB: str = Field(default="InhouseJdParser", validation_alias="MONGO_JD_PARSER_DB")

    # 3. OPENAI CONFIGURATION
    OPENAI_API_KEY: str = Field(..., validation_alias="OPENAI_API_KEY")
    OPENAI_LLM_MODEL: str = Field(default="gpt-4o", validation_alias="OPENAI_LLM_MODEL")
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", validation_alias="OPENAI_EMBEDDING_MODEL")

    # 4. MATCH ENGINE CONFIGURATION
    # Tier 2 cosine floor (0.0–1.0). 0.78 reduces false positives vs old hardcoded 0.70.
    TIER2_EMBEDDING_SIMILARITY_THRESHOLD: float = Field(
        default=0.78, validation_alias="TIER2_EMBEDDING_SIMILARITY_THRESHOLD"
    )
    # Legacy name kept for .env compatibility; if set and TIER2_* unset, tier2 uses value/100.
    SEMANTIC_THRESHOLD_VALUE: float = Field(default=78.0, validation_alias="SEMANTIC_THRESHOLD_VALUE")
    CONTEXTUAL_SIMILARITY_THRESHOLD: float = Field(default=0.72, validation_alias="CONTEXTUAL_SIMILARITY_THRESHOLD")
    MAX_CONTEXTS_PER_SIDE: int = Field(default=25, validation_alias="MAX_CONTEXTS_PER_SIDE")
    EMBEDDING_CACHE_ENABLED: bool = Field(default=True, validation_alias="EMBEDDING_CACHE_ENABLED")
    S3_DEEP_SCAN_ENABLED: bool = Field(default=True, validation_alias="S3_DEEP_SCAN_ENABLED")
    LLM_IMPLIED_SKILLS_ENABLED: bool = Field(default=True, validation_alias="LLM_IMPLIED_SKILLS_ENABLED")
    LLM_RESUME_RAW_TEXT_MAX_CHARS: int = Field(default=14000, validation_alias="LLM_RESUME_RAW_TEXT_MAX_CHARS")

    # 5. S3 / AWS CONFIGURATION
    AWS_ACCESS_KEY_ID: str = Field(..., validation_alias="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = Field(..., validation_alias="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    S3_BUCKET: str = Field(..., validation_alias="S3_BUCKET")
    S3_STREAM_TIMEOUT_SECONDS: int = Field(default=60, validation_alias="S3_STREAM_TIMEOUT_SECONDS")
    S3_SIGNED_URL_EXPIRY_SECONDS: int = Field(default=300, validation_alias="S3_SIGNED_URL_EXPIRY_SECONDS")

    # 6. FASTAPI / UVICORN CONFIGURATION
    API_TITLE: str = Field(default="Match Score API", validation_alias="API_TITLE")
    API_VERSION: str = Field(default="1.0.0", validation_alias="API_VERSION")

    # 7. PRODUCTION SECURITY CONFIGURATION
    ALLOWED_ORIGINS: list[str] = Field(default=["*"], validation_alias="ALLOWED_ORIGINS")

    # 8. PYDANTIC SETTINGS CONFIGURATION
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # 9. INTERNAL CLEAN PROPERTY ALIASES
    @property
    def mongodb_uri(self) -> str: return self.MONGO_CLUSTER_URI
    @property
    def mongodb_db_marketplace(self) -> str: return self.MONGO_MARKETPLACE_DB
    @property
    def mongodb_db_jd_parser(self) -> str: return self.MONGO_JD_PARSER_DB
    @property
    def openai_api_key(self) -> str: return self.OPENAI_API_KEY
    @property
    def openai_llm_model(self) -> str: return self.OPENAI_LLM_MODEL
    @property
    def openai_embedding_model(self) -> str: return self.OPENAI_EMBEDDING_MODEL
    @property
    def semantic_threshold(self) -> float: return self.SEMANTIC_THRESHOLD_VALUE
    @property
    def tier2_embedding_similarity_threshold(self) -> float:
        return float(self.TIER2_EMBEDDING_SIMILARITY_THRESHOLD)
    @property
    def contextual_similarity_threshold(self) -> float: return self.CONTEXTUAL_SIMILARITY_THRESHOLD
    @property
    def max_contexts_per_side(self) -> int: return self.MAX_CONTEXTS_PER_SIDE
    @property
    def embedding_cache_enabled(self) -> bool: return self.EMBEDDING_CACHE_ENABLED
    @property
    def s3_deep_scan_enabled(self) -> bool: return self.S3_DEEP_SCAN_ENABLED
    @property
    def llm_implied_skills_enabled(self) -> bool: return self.LLM_IMPLIED_SKILLS_ENABLED
    @property
    def llm_resume_raw_text_max_chars(self) -> int: return self.LLM_RESUME_RAW_TEXT_MAX_CHARS
    @property
    def s3_timeout(self) -> int: return self.S3_STREAM_TIMEOUT_SECONDS
    @property
    def signed_url_expiry(self) -> int: return self.S3_SIGNED_URL_EXPIRY_SECONDS
    @property
    def aws_access_key_id(self) -> str: return self.AWS_ACCESS_KEY_ID
    @property
    def aws_secret_access_key(self) -> str: return self.AWS_SECRET_ACCESS_KEY
    @property
    def aws_region(self) -> str: return self.AWS_REGION
    @property
    def s3_bucket(self) -> str: return self.S3_BUCKET
    @property
    def host(self) -> str: return self.HOST
    @property
    def port(self) -> int: return self.PORT
    @property
    def workers(self) -> int: return self.WORKERS
    @property
    def log_level(self) -> str: return self.LOG_LEVEL.upper()
    @property
    def api_title(self) -> str: return self.API_TITLE
    @property
    def api_version(self) -> str: return self.API_VERSION
    @property
    def allowed_origins(self) -> list[str]: return self.ALLOWED_ORIGINS

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance to prevent repeated .env parsing and disk access."""
    return Settings()

settings = get_settings()