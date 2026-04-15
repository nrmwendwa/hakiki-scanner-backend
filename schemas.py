"""
Pydantic models for request/response validation.
"""

from typing import Literal
from pydantic import BaseModel, Field


class PredictionScores(BaseModel):
    """Model scores for each prediction class."""

    real: float = Field(..., ge=0, le=100, description="Confidence score for real images")
    suspicious: float = Field(..., ge=0, le=100, description="Confidence score for suspicious images")
    fake: float = Field(..., ge=0, le=100, description="Confidence score for fake/AI-generated images")

    class Config:
        json_schema_extra = {
            "example": {
                "real": 85.5,
                "suspicious": 10.2,
                "fake": 4.3
            }
        }


class PredictionResult(BaseModel):
    """Response model for the prediction endpoint."""

    verdict: Literal["real", "suspicious", "fake"] = Field(
        ..., description="Classification verdict"
    )
    confidence: float = Field(
        ..., ge=0, le=100, description="Confidence percentage of the prediction"
    )
    scores: PredictionScores = Field(..., description="Individual class scores")

    class Config:
        json_schema_extra = {
            "example": {
                "verdict": "real",
                "confidence": 85.5,
                "scores": {
                    "real": 85.5,
                    "suspicious": 10.2,
                    "fake": 4.3
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid image format. Allowed formats: jpeg, jpg, png, webp",
                "error_code": "INVALID_FORMAT"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool
    version: str
