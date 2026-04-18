"""Pydantic models for LLM gateway requests and responses."""

from typing import Literal, Optional, Any
from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMImagePart(BaseModel):
    data: bytes
    mime_type: str


class LLMRequest(BaseModel):
    task: str
    messages: list[LLMMessage]
    images: list[LLMImagePart] = Field(default_factory=list)
    response_schema: Optional[dict] = None
    max_output_tokens: int = 2048
    temperature: float = 0.1


class LLMUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    structured: Optional[dict] = None
    model: str
    provider: str
    latency_ms: int
    usage: LLMUsage = Field(default_factory=LLMUsage)
