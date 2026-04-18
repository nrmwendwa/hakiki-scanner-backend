"""Schemas for the validation decision layer."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from pipelines.schemas import DeepfakeScores


class DecisionVerdict(str, Enum):
    VALID = "valid"
    SUSPICIOUS = "suspicious"
    INVALID = "invalid"


class EvidenceItem(BaseModel):
    claim: str
    matched_source: str = ""
    matched_url: str = ""
    similarity: float = 0.0
    verdict_contribution: str


class DecisionResult(BaseModel):
    verdict: DecisionVerdict
    confidence: float = Field(ge=0, le=100)
    reasoning: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    signals: dict = Field(default_factory=dict)
    input_type: str
    deepfake_scores: Optional[DeepfakeScores] = None
    pipeline_errors: list[str] = Field(default_factory=list)
    trace: dict = Field(default_factory=dict)
