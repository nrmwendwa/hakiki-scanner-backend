"""
Configuration module for Hakiki Scanner API.

Loads settings from environment / .env. Centralizes all LLM, API, and
model configuration so no subsystem reads `os.getenv` directly.
"""

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class LLMSettings(BaseModel):
    """LLM gateway configuration.

    Task-based routing: `route_*` fields pick which logical model
    handles each pipeline step. Swapping providers is a config change.
    """

    primary_provider: str = "gemini"

    gemini_api_key: Optional[str] = None
    gemini_model: str = os.getenv("GEMINI_MODEL")
    gemini_vision_model: str = os.getenv("GEMINI_MODEL")

    route_ocr: str = "gemini-vision"
    route_image_analysis: str = "gemini-vision"
    route_claim_extraction: str = "gemini-text"
    route_claim_verification: str = "gemini-text"
    route_reasoning: str = "gemini-text"

    request_timeout_s: int = 30
    max_retries: int = 3
    retry_backoff_s: float = 1.5

    def require_gemini_key(self) -> str:
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to backend/.env."
            )
        return self.gemini_api_key


class Settings(BaseModel):
    """Application settings - loads from .env file and environment variables."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    api_title: str = "Hakiki Scanner API"
    api_version: str = "1.2.0"

    max_upload_size_mb: int = 10
    allowed_image_formats: tuple = ("jpeg", "jpg", "png", "webp")
    allowed_document_formats: tuple = ("pdf", "txt", "md")

    min_image_size_pixels: int = 32
    max_image_size_pixels: int = 4096

    cors_origins: str = (
        "http://localhost,http://localhost:3000,http://localhost:5173,"
        "http://localhost:8080,http://127.0.0.1:3000,"
        "http://127.0.0.1:5173,http://127.0.0.1:8080"
    )

    log_level: str = "INFO"

    llm: LLMSettings = Field(default_factory=LLMSettings)

    class Config:
        case_sensitive = False

    def get_cors_origins_list(self) -> list[str]:
        if isinstance(self.cors_origins, str):
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return self.cors_origins if isinstance(self.cors_origins, list) else []

@lru_cache()
def get_settings() -> Settings:
    return Settings(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        debug=os.getenv("DEBUG", "False").lower() == "true",
        workers=int(os.getenv("WORKERS", "4")),
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")),
        cors_origins=os.getenv(
            "CORS_ORIGINS",
            "http://localhost,http://localhost:3000,http://localhost:5173,"
            "http://localhost:8080,http://127.0.0.1:3000,"
            "http://127.0.0.1:5173,http://127.0.0.1:8080",
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        llm=LLMSettings(
            primary_provider=os.getenv("LLM_PRIMARY_PROVIDER", "gemini"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL"),
            gemini_vision_model=os.getenv("GEMINI_MODEL"),
            route_ocr=os.getenv("LLM_ROUTE_OCR", "gemini-vision"),
            route_image_analysis=os.getenv("LLM_ROUTE_IMAGE_ANALYSIS", "gemini-vision"),
            route_claim_extraction=os.getenv("LLM_ROUTE_CLAIM_EXTRACTION", "gemini-text"),
            route_claim_verification=os.getenv("LLM_ROUTE_CLAIM_VERIFICATION", "gemini-text"),
            route_reasoning=os.getenv("LLM_ROUTE_REASONING", "gemini-text"),
            request_timeout_s=int(os.getenv("LLM_TIMEOUT_S", "30")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            retry_backoff_s=float(os.getenv("LLM_RETRY_BACKOFF_S", "1.5")),
        ),
    )
