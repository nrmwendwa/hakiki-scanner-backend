"""Shared pipeline schemas used across image, text, and document flows."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class InputType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class ExtractedClaim(BaseModel):
    statement: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    numeric_value: Optional[float] = None
    date: Optional[str] = None
    topic: str = ""


class ImageMetadata(BaseModel):
    width: int
    height: int
    format: str
    mode: str
    size_bytes: int
    phash: str
    sha256: str
    exif: dict = Field(default_factory=dict)


class ImageAnalysis(BaseModel):
    scene_description: str = ""
    manipulation_signals: list[str] = Field(default_factory=list)
    source_indicators: list[str] = Field(default_factory=list)
    ai_generated_likelihood: Literal["low", "medium", "high"] = "low"


class OCRResult(BaseModel):
    extracted_text: str = ""
    language: str = ""
    visible_claims: list[str] = Field(default_factory=list)
    has_text: bool = False


class DeepfakeScores(BaseModel):
    verdict: Literal["real", "suspicious", "fake"]
    confidence: float
    real: float
    suspicious: float
    fake: float


class PipelineResult(BaseModel):
    input_type: InputType
    raw_text: str = ""
    claims: list[ExtractedClaim] = Field(default_factory=list)
    image_metadata: Optional[ImageMetadata] = None
    image_analysis: Optional[ImageAnalysis] = None
    deepfake_scores: Optional[DeepfakeScores] = None
    ocr: Optional[OCRResult] = None
    errors: list[str] = Field(default_factory=list)
    trace: dict = Field(default_factory=dict)
