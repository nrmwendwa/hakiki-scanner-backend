"""Verify a claim by searching the web and synthesising a verdict with the LLM."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from llm.prompts import claim_verification
from llm.registry import get_provider_for_task

from .search import search_web

logger = logging.getLogger(__name__)

_VALID_LABELS = {"imethibitishwa", "ya_uongo", "haijathibitishwa"}


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _no_evidence(claim: str, detail: str) -> dict:
    return {
        "label": "haijathibitishwa",
        "confidence": 0.0,
        "source": "",
        "url": "",
        "details": detail,
        "similarity_score": 0.0,
    }


def verify_claim_online(claim: str, max_results: int = 6) -> dict:
    """Verify a single claim against DDG search results using the LLM.

    Returns a dict compatible with the legacy ``verify_claim`` shape so the
    decision engine needs no changes.
    """
    claim = (claim or "").strip()
    if not claim:
        return _no_evidence(claim, "Dai tupu.")

    results = search_web(claim, max_results=max_results)
    if not results:
        return _no_evidence(claim, "Hakuna matokeo ya utafutaji mtandaoni.")

    try:
        provider = get_provider_for_task("claim_verification")
        response = provider.generate(claim_verification.build_request(claim, results))
    except Exception as exc:
        logger.warning("claim_verification LLM call failed: %s", exc)
        return _no_evidence(claim, f"Uthibitishaji mtandaoni umeshindwa: {exc}")

    parsed: Any = response.structured or {}
    if not isinstance(parsed, dict):
        return _no_evidence(claim, "Majibu ya mfumo hayajapangika vizuri.")

    label = str(parsed.get("label") or "").strip()
    if label not in _VALID_LABELS:
        label = "haijathibitishwa"

    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(100.0, confidence))

    best_url = str(parsed.get("best_url") or "").strip()
    best_source = str(parsed.get("best_source") or "").strip()
    if not best_source and best_url:
        best_source = _host(best_url)
    rationale = str(parsed.get("rationale") or "").strip()

    return {
        "label": label,
        "confidence": round(confidence, 1),
        "source": best_source,
        "url": best_url,
        "details": rationale,
        "similarity_score": round(confidence / 100.0, 3),
    }
