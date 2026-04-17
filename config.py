"""
Configuration module for Hakiki Scanner API
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

# Load .env file
load_dotenv()


class Settings(BaseModel):
    """Application settings - loads from .env file and environment variables"""

    # API settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    api_title: str = "Hakiki Scanner API"
    api_version: str = "1.0.0"

    # Model settings
    model_path: str = "models/efficientnet3class_full_model.pth"
    device: str = "cuda"  # cuda or cpu
    model_confidence_threshold: float = 0.30

    # File upload settings
    max_upload_size_mb: int = 10
    allowed_image_formats: tuple = ("jpeg", "jpg", "png", "webp")

    # Image validation settings
    min_image_size_pixels: int = 32
    max_image_size_pixels: int = 4096

    # CORS settings
    cors_origins: str = "http://localhost,http://localhost:3000,http://localhost:5173,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:8080"

    # Logging
    log_level: str = "INFO"

    class Config:
        case_sensitive = False

    def get_cors_origins_list(self) -> list[str]:
        """Get CORS origins as list"""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return self.cors_origins if isinstance(self.cors_origins, list) else []

    def get_model_path(self) -> Path:
        """Get absolute path to model file"""
        model_path = Path(self.model_path)
        if not model_path.is_absolute():
            # Make relative to backend directory
            model_path = Path(__file__).parent / model_path
        return model_path

    def validate_model_exists(self) -> bool:
        """Check if model file exists"""
        return self.get_model_path().exists()


@lru_cache()
def get_settings() -> Settings:
    """Get application settings from environment variables"""
    return Settings(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        debug=os.getenv("DEBUG", "False").lower() == "true",
        workers=int(os.getenv("WORKERS", "4")),
        model_path=os.getenv("MODEL_PATH", "models/efficientnet3class_full_model.pth"),
        device=os.getenv("DEVICE", "cuda"),
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")),
        cors_origins=os.getenv(
            "CORS_ORIGINS",
            "http://localhost,http://localhost:3000,http://localhost:5173,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:8080"
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
