"""
Text Verification Module for Hakiki Scanner

Ported from the colleague's Flask-based fact-checker.
Provides claim verification using:
  1. Similarity matching against a verified claims dataset
  2. ML classification fallback (Naive Bayes + TF-IDF)
"""

import csv
import re
import sys
import logging
from pathlib import Path
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Add text_model directory to path so we can import the classifier
TEXT_MODEL_DIR = Path(__file__).parent / "text_model"
sys.path.insert(0, str(TEXT_MODEL_DIR))

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tanzania_publicinfo_dataset.csv"
SIMILARITY_THRESHOLD = 0.6


def load_verified_claims() -> list[dict]:
    """Load verified claims from the CSV dataset"""
    claims = []

    if not DATA_PATH.exists():
        logger.warning(f"Dataset not found at {DATA_PATH}")
        return claims

    try:
        with DATA_PATH.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                statement = (
                    row.get("statement")
                    or row.get("text")
                    or row.get("title")
                    or ""
                )
                claims.append({
                    "statement": statement.strip(),
                    "label": row.get("label", "").strip().lower(),
                    "source": row.get("source", ""),
                    "url": row.get("url", ""),
                    "category": row.get("category", ""),
                })
        logger.info(f"Loaded {len(claims)} verified claims from dataset")
    except Exception as e:
        logger.error(f"Error loading verified claims: {e}")

    return claims


def find_similar_claims(
    input_text: str, claims: list[dict], threshold: float = SIMILARITY_THRESHOLD
) -> list[dict]:
    """Find claims similar to input text using sequence matching"""
    matches = []
    input_lower = input_text.lower().strip()

    for claim in claims:
        claim_text = claim.get("statement", "").lower().strip()
        if not claim_text:
            continue

        similarity = SequenceMatcher(None, input_lower, claim_text).ratio()

        if similarity >= threshold:
            matches.append({"claim": claim, "similarity": similarity})

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches


def extract_numbers(text: str) -> list[float]:
    """Extract numbers from text for statistical comparison"""
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return [float(n) for n in numbers]


def compare_statistics(input_text: str, claim_text: str) -> bool | None:
    """Compare statistical claims — checks if numbers are within 10% tolerance"""
    input_nums = extract_numbers(input_text)
    claim_nums = extract_numbers(claim_text)

    if not input_nums or not claim_nums:
        return None

    for input_num in input_nums:
        for claim_num in claim_nums:
            if claim_num == 0 and input_num == 0:
                return True
            if max(input_num, claim_num) > 0:
                if abs(input_num - claim_num) / max(input_num, claim_num) < 0.1:
                    return True
    return False


def verify_claim(input_text: str) -> dict:
    """
    Main verification logic.

    1. Loads the verified claims dataset
    2. Checks for similar claims via string similarity
    3. Falls back to ML classification if no match found

    Returns a dict with: label, confidence, source, details, scores (optional)
    """
    claims = load_verified_claims()

    if not claims:
        return {
            "label": "haijathibitishwa",
            "confidence": 0.0,
            "source": "Hakuna data",
            "details": "Dataset ni tupu. Endesha scraper.py kukusanya data.",
        }

    # Try similarity matching first
    matches = find_similar_claims(input_text, claims)

    if matches:
        best_match = matches[0]
        claim = best_match["claim"]
        similarity = best_match["similarity"]

        is_statistical_match = compare_statistics(input_text, claim["statement"])

        if (is_statistical_match and similarity >= 0.75) or similarity > 0.85:
            label = claim.get("label", "unverified")
            confidence = min(similarity + 0.2, 0.95)
        else:
            label = "unverified"
            confidence = similarity * 0.7

        # Normalize labels to Swahili
        label_map = {
            "verified": "imethibitishwa",
            "false": "ya_uongo",
            "unverified": "haijathibitishwa",
        }
        display_label = label_map.get(label, label)

        return {
            "label": display_label,
            "confidence": round(confidence * 100, 1),
            "source": claim.get("source", "Haijulikani"),
            "url": claim.get("url", ""),
            "details": f"Imelinganishwa na: {claim['statement']}",
            "similarity_score": round(similarity, 3),
        }

    # Fallback: ML classifier
    try:
        from fake_news_classifier import predict_statement

        ml_scores = predict_statement(input_text)
        best_label_name, best_label_score = max(ml_scores.items(), key=lambda x: x[1])

        # If the classifier is degenerate (single-class model) or unconfident,
        # say so honestly instead of surfacing a meaningless verdict.
        if best_label_name == "unverified" or best_label_score < 0.55:
            return {
                "label": "haijathibitishwa",
                "confidence": 0.0,
                "source": "Hakuna data ya kutosha",
                "details": (
                    "Dai halijapatikana kwenye orodha ya madai yaliyothibitishwa, "
                    "na mfumo wa ML hauna data ya kutosha kutoa uamuzi wa uhakika."
                ),
                "scores": {k: round(v * 100, 1) for k, v in ml_scores.items()},
            }

        label_map = {
            "verified": "imethibitishwa",
            "false": "ya_uongo",
            "unverified": "haijathibitishwa",
        }
        display_label = label_map.get(best_label_name, best_label_name)

        return {
            "label": display_label,
            "confidence": round(best_label_score * 100, 1),
            "source": "ML Classifier",
            "details": "Hakuna mechi ya moja kwa moja, inatumia uainishaji wa kujifunza kwa mashine.",
            "scores": {k: round(v * 100, 1) for k, v in ml_scores.items()},
        }
    except Exception as e:
        logger.error(f"ML classifier failed: {e}")
        return {
            "label": "haijathibitishwa",
            "confidence": 0.0,
            "source": "Hitilafu",
            "details": f"Haikuweza kuthibitisha: {str(e)}",
        }
