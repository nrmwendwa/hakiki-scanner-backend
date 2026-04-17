"""
FastAPI backend for Hakiki Scanner - Face authentication detection system

Production-ready API for detecting AI-generated vs real faces.
Provides REST endpoints for image analysis with detailed confidence scores.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import get_settings
from model_service import ModelService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global model service
model_service: ModelService = None


class PredictionScores(BaseModel):
    """Individual class prediction scores"""

    real: float
    suspicious: float
    fake: float


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""

    verdict: Literal["real", "suspicious", "fake"]
    confidence: float
    scores: PredictionScores


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""

    status: str
    model_loaded: bool
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    global model_service
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("Starting Hakiki Scanner API...")
    logger.info(f"API Version: {settings.api_version}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info(f"Model Path: {settings.get_model_path()}")
    logger.info(f"Device: {settings.device}")
    logger.info(f"CORS Origins: {settings.cors_origins}")
    logger.info("=" * 60)

    try:
        model_service = ModelService(model_path=str(settings.get_model_path()))
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        logger.error("Application will not start without the model")
        raise

    yield

    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down Hakiki Scanner API...")
    if model_service:
        model_service.cleanup()
        logger.info("✓ Cleanup completed")
    logger.info("=" * 60)


# Create FastAPI app
app = FastAPI(
    title="Hakiki Scanner API",
    description="Deep learning API for detecting AI-generated faces vs real faces",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Info Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint - Verify API and model status
    
    Returns:
        HealthResponse with status, model state, and version info
    """
    return HealthResponse(
        status="ok",
        model_loaded=model_service is not None,
        version="1.0.0",
    )


@app.get("/info", tags=["Information"])
async def get_info():
    """
    Get API and model information
    
    Returns:
        Dictionary with API version, model path, device, and config info
    """
    settings = get_settings()
    return {
        "api": {
            "name": "Hakiki Scanner API",
            "version": settings.api_version,
            "title": settings.api_title,
            "debug": settings.debug,
        },
        "model": {
            "path": str(settings.get_model_path()),
            "device": str(model_service.device) if model_service else "unknown",
            "loaded": model_service is not None,
        },
        "configuration": {
            "max_upload_size_mb": settings.max_upload_size_mb,
            "allowed_formats": settings.allowed_image_formats,
            "min_image_size_px": settings.min_image_size_pixels,
            "max_image_size_px": settings.max_image_size_pixels,
            "cors_origins": settings.cors_origins,
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "info": "/info",
        }
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API welcome message"""
    return {
        "message": "Hakiki Scanner API - Face Authenticity Detection",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)",
        "info": "/info",
    }


# ============================================================================
# Prediction Endpoint
# ============================================================================


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(image: UploadFile = File(...)) -> PredictionResponse:
    """
    Analyze an image to detect if it's real, suspicious, or fake (AI-generated)

    The model classifies the image as:
    - **real**: Genuine/real face detected
    - **suspicious**: Potentially manipulated or unclear
    - **fake**: AI-generated or synthetic face detected

    Args:
        image: Uploaded image file (JPG, PNG, WebP - max 10MB)

    Returns:
        PredictionResponse with verdict, confidence percentage, and detailed scores

    Raises:
        HTTPException: For invalid format, oversized file, or processing errors
    """
    settings = get_settings()
    
    if model_service is None:
        logger.error("Model service not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service not available",
        )

    # Validate file type
    if not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    file_ext = image.filename.split(".")[-1].lower()
    if file_ext not in settings.allowed_image_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Allowed formats: {', '.join(settings.allowed_image_formats)}",
        )

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only image files are accepted.",
        )

    # Check file size
    try:
        image_data = await image.read()
        file_size = len(image_data)
        max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds the maximum limit of {settings.max_upload_size_mb}MB",
            )
        await image.seek(0)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking file size: {e}")

    try:
        # Read image bytes
        image_bytes = await image.read()
        logger.info(f"Processing image: {image.filename} ({len(image_bytes)} bytes)")

        # Get prediction from model
        result = model_service.predict(image_bytes)
        logger.info(
            f"Prediction: {result['verdict']} "
            f"(confidence: {result['confidence']:.2f}%, "
            f"real={result['scores']['real']:.2f}%, "
            f"suspicious={result['scores']['suspicious']:.2f}%, "
            f"fake={result['scores']['fake']:.2f}%)"
        )

        return PredictionResponse(
            verdict=result["verdict"],
            confidence=result["confidence"],
            scores=PredictionScores(**result["scores"]),
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during prediction",
        )


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "status_code": 500},
    )


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    logger.info(f"Starting server at {settings.host}:{settings.port}")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.log_level.lower(),
    )
