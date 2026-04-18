"""Deterministic fusion of pipeline signals into a final verdict.

The engine combines three signals (claims, image authenticity, source trust),
renormalizes weights over present signals, and thresholds the weighted score
to produce a verdict. Verification for textual claims is delegated to
``verification.verify_claim_online`` - the engine itself never asks an LLM
for the final verdict.
"""

from __future__ import annotations

import logging
from typing import Optional

from pipelines.schemas import InputType, PipelineResult
from verification import verify_claim_online

from .schemas import DecisionResult, DecisionVerdict, EvidenceItem

logger = logging.getLogger(__name__)

_LABEL_TO_SCORE = {
    "imethibitishwa": 1.0,
    "haijathibitishwa": 0.5,
    "ya_uongo": 0.0,
}

_BASE_WEIGHTS = {
    "claim_score": 0.5,
    "image_authenticity": 0.3,
    "source_trust": 0.2,
}

_INPUT_TYPE_SW = {
    "text": "maandishi",
    "image": "picha",
    "document": "hati",
    "unknown": "taarifa isiyojulikana",
}


def _score_claims(result: PipelineResult) -> tuple[Optional[float], list[EvidenceItem], int, int]:
    if not result.claims:
        return None, [], 0, 0

    evidence: list[EvidenceItem] = []
    weighted_values: list[float] = []
    matched_count = 0
    contradicted_count = 0

    for claim in result.claims:
        try:
            verification = verify_claim_online(claim.statement)
        except Exception as exc:
            logger.warning("verify_claim_online failed for claim: %s", exc)
            continue

        label = verification.get("label", "haijathibitishwa")
        mapped = _LABEL_TO_SCORE.get(label, 0.5)

        raw_confidence = verification.get("confidence", 0) or 0
        weight = (raw_confidence / 100.0) if raw_confidence else 0.5
        weighted_values.append(mapped * weight + 0.5 * (1 - weight) if weight < 1 else mapped)

        matched_source = verification.get("source", "") or ""
        matched_url = verification.get("url", "") or ""
        similarity = float(verification.get("similarity_score") or 0.0)

        if label == "imethibitishwa":
            contribution = f"imethibitishwa na {matched_source}" if matched_source else "imethibitishwa"
            matched_count += 1
        elif label == "ya_uongo":
            contribution = f"imepingwa na {matched_source}" if matched_source else "imegunduliwa kuwa ya uongo"
            contradicted_count += 1
        else:
            contribution = f"haijathibitishwa (chanzo: {matched_source})" if matched_source else "haijathibitishwa"

        evidence.append(
            EvidenceItem(
                claim=claim.statement,
                matched_source=matched_source,
                matched_url=matched_url,
                similarity=similarity,
                verdict_contribution=contribution,
            )
        )

    if not weighted_values:
        return None, evidence, matched_count, contradicted_count

    claim_score = sum(weighted_values) / len(weighted_values)
    return claim_score, evidence, matched_count, contradicted_count


def _score_image_authenticity(result: PipelineResult) -> Optional[float]:
    if result.image_analysis is None:
        return None

    likelihood = result.image_analysis.ai_generated_likelihood
    base = {"low": 1.0, "medium": 0.5, "high": 0.0}.get(likelihood, 0.5)
    penalty = 0.15 * len(result.image_analysis.manipulation_signals)
    return max(0.0, base - penalty)


def _score_source_trust(
    result: PipelineResult,
    evidence: list[EvidenceItem],
) -> Optional[float]:
    if result.input_type == InputType.IMAGE:
        if result.image_analysis is None:
            return None
        indicators = result.image_analysis.source_indicators
        if not indicators:
            return 0.5
        return min(1.0, 0.2 * len(indicators))

    if result.input_type in (InputType.TEXT, InputType.DOCUMENT):
        if not evidence:
            return 0.5
        matched = sum(1 for e in evidence if e.matched_source)
        return matched / len(evidence)

    return None


def _classify(score: float) -> DecisionVerdict:
    if score >= 0.75:
        return DecisionVerdict.VALID
    if score >= 0.4:
        return DecisionVerdict.SUSPICIOUS
    return DecisionVerdict.INVALID


def _build_reasoning(
    result: PipelineResult,
    verdict: DecisionVerdict,
    evidence: list[EvidenceItem],
    matched_count: int,
    contradicted_count: int,
    signals: dict,
) -> str:
    parts: list[str] = []
    input_type_sw = _INPUT_TYPE_SW.get(result.input_type.value, result.input_type.value)
    claim_count = len(result.claims)

    if claim_count:
        parts.append(
            f"Tumechambua madai {claim_count} kutoka kwa {input_type_sw}."
        )
    else:
        parts.append(f"Tumechambua {input_type_sw} lakini hatukupata madai yoyote.")

    if evidence:
        pieces = []
        if matched_count:
            pieces.append(f"{matched_count} yamelingana na vyanzo vilivyothibitishwa")
        if contradicted_count:
            sources = {e.matched_source for e in evidence if "imepingwa" in e.verdict_contribution or "uongo" in e.verdict_contribution}
            source_hint = f" (vyanzo: {', '.join(sorted(s for s in sources if s))})" if sources and any(sources) else ""
            pieces.append(f"{contradicted_count} yamegunduliwa kuwa ya uongo{source_hint}")
        unverified = len(evidence) - matched_count - contradicted_count
        if unverified > 0:
            pieces.append(f"{unverified} hayajathibitishwa")
        if pieces:
            parts.append("; ".join(pieces).capitalize() + ".")

    if signals.get("image_authenticity") is not None:
        img_score = signals["image_authenticity"]
        if img_score >= 0.75:
            parts.append("Viashiria vya uhalisia wa picha vinaonekana safi.")
        elif img_score >= 0.4:
            parts.append("Picha ina viashiria mchanganyiko vya uhalisia.")
        else:
            parts.append("Picha inaonyesha viashiria vikali vya kuchezewa au kutengenezwa na AI.")

    verdict_reason_map = {
        DecisionVerdict.VALID: "Uamuzi wa jumla: taarifa ni halali kulingana na viashiria imara.",
        DecisionVerdict.SUSPICIOUS: "Uamuzi wa jumla: inatia shaka kutokana na viashiria mchanganyiko au dhaifu.",
        DecisionVerdict.INVALID: "Uamuzi wa jumla: taarifa si halali kutokana na ushahidi unaopingana.",
    }

    raw_score = signals.get("raw_score")
    if raw_score is None:
        parts.append("Hakuna data ya kutosha kutoa uamuzi wa uhakika; tumeweka kama inatia shaka.")
    else:
        parts.append(verdict_reason_map[verdict])

    if result.errors:
        parts.append(
            f"Mfumo umeripoti makosa {len(result.errors)} wakati wa uchambuzi."
        )

    return " ".join(parts)


def decide(result: PipelineResult) -> DecisionResult:
    claim_score, evidence, matched_count, contradicted_count = _score_claims(result)
    image_authenticity = _score_image_authenticity(result)
    source_trust = _score_source_trust(result, evidence)

    present: dict[str, float] = {}
    if claim_score is not None:
        present["claim_score"] = claim_score
    if image_authenticity is not None:
        present["image_authenticity"] = image_authenticity
    if source_trust is not None:
        present["source_trust"] = source_trust

    weights_used = {name: _BASE_WEIGHTS[name] for name in present}

    if not present:
        signals = {
            "claim_score": claim_score,
            "image_authenticity": image_authenticity,
            "source_trust": source_trust,
            "weights_used": weights_used,
            "raw_score": None,
        }
        input_type_sw = _INPUT_TYPE_SW.get(result.input_type.value, result.input_type.value)
        reasoning = (
            f"Tumechambua {input_type_sw} lakini hatukupata madai, uchambuzi wa picha, "
            "wala viashiria vya chanzo cha kuaminika. Hakuna data ya kutosha kutoa uamuzi wa uhakika."
        )
        if result.errors:
            reasoning += f" Mfumo umeripoti makosa {len(result.errors)} wakati wa uchambuzi."
        return DecisionResult(
            verdict=DecisionVerdict.SUSPICIOUS,
            confidence=0.0,
            reasoning=reasoning,
            evidence=evidence,
            signals=signals,
            input_type=result.input_type.value,
            deepfake_scores=result.deepfake_scores,
            pipeline_errors=list(result.errors),
            trace=dict(result.trace),
        )

    total_weight = sum(weights_used.values())
    raw_score = sum(weights_used[name] * present[name] for name in present) / total_weight

    verdict = _classify(raw_score)
    confidence = round(raw_score * 100, 1)

    signals = {
        "claim_score": claim_score,
        "image_authenticity": image_authenticity,
        "source_trust": source_trust,
        "weights_used": weights_used,
        "raw_score": raw_score,
    }

    reasoning = _build_reasoning(
        result, verdict, evidence, matched_count, contradicted_count, signals
    )

    return DecisionResult(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        evidence=evidence,
        signals=signals,
        input_type=result.input_type.value,
        deepfake_scores=result.deepfake_scores,
        pipeline_errors=list(result.errors),
        trace=dict(result.trace),
    )
