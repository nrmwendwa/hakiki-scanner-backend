"""Text claim-extraction pipeline."""

from __future__ import annotations

import logging
import re
from typing import Any

from llm.prompts import claim_extraction
from llm.registry import get_provider_for_task

from .schemas import ExtractedClaim, InputType, PipelineResult

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    stripped = text.strip()
    collapsed = _WHITESPACE_RE.sub(" ", stripped)
    return collapsed


def _parse_claims(structured: dict | None) -> list[ExtractedClaim]:
    if not structured:
        return []
    raw_claims: list[Any] = []
    if isinstance(structured, dict):
        for key in ("claims", "extracted_claims", "results"):
            if key in structured and isinstance(structured[key], list):
                raw_claims = structured[key]
                break
    claims: list[ExtractedClaim] = []
    for item in raw_claims:
        if isinstance(item, str):
            claims.append(ExtractedClaim(statement=item))
            continue
        if not isinstance(item, dict):
            continue
        try:
            claims.append(ExtractedClaim(**item))
        except Exception:
            statement = str(item.get("statement") or item.get("text") or "").strip()
            if statement:
                claims.append(ExtractedClaim(statement=statement))
    return claims


def run_text_pipeline(text: str) -> PipelineResult:
    normalized = _normalize_text(text or "")
    if not normalized:
        raise ValueError("Empty text")

    result = PipelineResult(input_type=InputType.TEXT, raw_text=normalized)

    try:
        provider = get_provider_for_task("claim_extraction")
        request = claim_extraction.build_request(normalized)
        response = provider.generate(request)
        result.claims = _parse_claims(response.structured)
        result.trace["claim_extraction"] = {
            "model": response.model,
            "provider": response.provider,
            "latency_ms": response.latency_ms,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    except Exception as exc:
        logger.exception("Claim extraction failed")
        result.errors.append(f"claim_extraction: {exc}")

    return result
