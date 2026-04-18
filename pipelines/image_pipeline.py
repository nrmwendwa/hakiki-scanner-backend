"""Image pipeline: validate, extract metadata, OCR, analyze, extract claims."""

from __future__ import annotations

import hashlib
import io
import logging
import os
from typing import Any

from config import get_settings
from llm.prompts import claim_extraction, image_analysis, ocr
from llm.registry import get_provider_for_task
from llm.schemas import LLMImagePart

from .schemas import (
    DeepfakeScores,
    ExtractedClaim,
    ImageAnalysis,
    ImageMetadata,
    InputType,
    OCRResult,
    PipelineResult,
)

logger = logging.getLogger(__name__)

_PIL_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}


def _validate_size(size_bytes: int, max_mb: int) -> None:
    if size_bytes <= 0:
        raise ValueError("Empty image payload")
    if size_bytes > max_mb * 1024 * 1024:
        raise ValueError(f"Image exceeds maximum size of {max_mb} MB")


def _validate_format(pil_format: str | None, allowed: tuple[str, ...]) -> str:
    if not pil_format:
        raise ValueError("Unable to determine image format")
    normalized = pil_format.lower()
    if normalized == "jpeg":
        canonical = "jpeg"
    else:
        canonical = normalized
    allowed_norm = {f.lower() for f in allowed}
    if canonical not in allowed_norm and not (
        canonical == "jpeg" and ("jpg" in allowed_norm or "jpeg" in allowed_norm)
    ):
        raise ValueError(
            f"Image format '{pil_format}' not allowed. Allowed: {sorted(allowed_norm)}"
        )
    return canonical


def _extract_exif(img: Any) -> dict:
    exif: dict = {}
    getter = getattr(img, "_getexif", None)
    if not callable(getter):
        return exif
    try:
        raw = getter()
    except Exception:
        logger.debug("EXIF extraction failed", exc_info=True)
        return exif
    if not raw:
        return exif

    try:
        from PIL.ExifTags import TAGS
    except Exception:
        TAGS = {}

    for tag_id, value in raw.items():
        name = TAGS.get(tag_id, str(tag_id)) if TAGS else str(tag_id)
        try:
            if isinstance(value, bytes):
                exif[str(name)] = value.decode("utf-8", errors="replace")
            else:
                exif[str(name)] = str(value)
        except Exception:
            exif[str(name)] = repr(value)
    return exif


def _strip_exif_bytes(img: Any, pil_format: str) -> bytes:
    buffer = io.BytesIO()
    save_format = pil_format.upper()
    save_kwargs: dict[str, Any] = {}
    working = img

    if save_format in {"JPEG", "WEBP"}:
        if working.mode not in ("RGB", "L"):
            working = working.convert("RGB")
        if save_format == "JPEG":
            save_kwargs["quality"] = 92
            save_kwargs["optimize"] = True
    elif save_format == "PNG":
        save_kwargs["optimize"] = True

    data = working.info
    if isinstance(data, dict):
        working.info = {k: v for k, v in data.items() if k.lower() != "exif"}

    working.save(buffer, format=save_format, **save_kwargs)
    return buffer.getvalue()


def _compute_phash(img: Any) -> str:
    try:
        import imagehash
    except Exception as exc:
        logger.warning("imagehash unavailable: %s", exc)
        return ""
    try:
        return str(imagehash.phash(img))
    except Exception:
        logger.exception("phash computation failed")
        return ""


def _parse_ocr(structured: dict | None, fallback_text: str) -> OCRResult:
    if not structured:
        return OCRResult(
            extracted_text=fallback_text or "",
            has_text=bool(fallback_text),
        )
    try:
        parsed = OCRResult(**structured)
    except Exception:
        extracted = str(structured.get("extracted_text") or fallback_text or "")
        language = str(structured.get("language") or "")
        visible = structured.get("visible_claims") or []
        if not isinstance(visible, list):
            visible = []
        visible_claims = [str(v) for v in visible]
        parsed = OCRResult(
            extracted_text=extracted,
            language=language,
            visible_claims=visible_claims,
            has_text=bool(extracted),
        )
    if parsed.extracted_text and not parsed.has_text:
        parsed.has_text = True
    return parsed


def _derive_image_scores(analysis: ImageAnalysis) -> DeepfakeScores:
    """Map Gemini's likelihood + manipulation signals to a real/suspicious/fake breakdown."""
    base = {
        "low": (80.0, 15.0, 5.0),
        "medium": (30.0, 50.0, 20.0),
        "high": (10.0, 20.0, 70.0),
    }.get(analysis.ai_generated_likelihood, (33.0, 34.0, 33.0))
    real, suspicious, fake = base

    shift = min(real, 5.0 * len(analysis.manipulation_signals))
    real -= shift
    suspicious += shift * 0.4
    fake += shift * 0.6

    total = real + suspicious + fake or 1.0
    real = real * 100.0 / total
    suspicious = suspicious * 100.0 / total
    fake = fake * 100.0 / total

    pairs = [("real", real), ("suspicious", suspicious), ("fake", fake)]
    verdict_name, confidence = max(pairs, key=lambda p: p[1])

    return DeepfakeScores(
        verdict=verdict_name,  # type: ignore[arg-type]
        confidence=round(confidence, 2),
        real=round(real, 2),
        suspicious=round(suspicious, 2),
        fake=round(fake, 2),
    )


def _parse_image_analysis(structured: dict | None, fallback_text: str) -> ImageAnalysis:
    if not structured:
        return ImageAnalysis(scene_description=fallback_text or "")
    try:
        return ImageAnalysis(**structured)
    except Exception:
        scene = str(structured.get("scene_description") or fallback_text or "")
        signals = structured.get("manipulation_signals") or []
        sources = structured.get("source_indicators") or []
        likelihood = structured.get("ai_generated_likelihood") or "low"
        if likelihood not in ("low", "medium", "high"):
            likelihood = "low"
        return ImageAnalysis(
            scene_description=scene,
            manipulation_signals=[str(s) for s in signals] if isinstance(signals, list) else [],
            source_indicators=[str(s) for s in sources] if isinstance(sources, list) else [],
            ai_generated_likelihood=likelihood,
        )


def _parse_claims(structured: dict | None) -> list[ExtractedClaim]:
    if not structured:
        return []
    raw: list[Any] = []
    if isinstance(structured, dict):
        for key in ("claims", "extracted_claims", "results"):
            value = structured.get(key)
            if isinstance(value, list):
                raw = value
                break
    out: list[ExtractedClaim] = []
    for item in raw:
        if isinstance(item, str):
            out.append(ExtractedClaim(statement=item))
            continue
        if not isinstance(item, dict):
            continue
        try:
            out.append(ExtractedClaim(**item))
        except Exception:
            statement = str(item.get("statement") or item.get("text") or "").strip()
            if statement:
                out.append(ExtractedClaim(statement=statement))
    return out


def run_image_pipeline(image_bytes: bytes, filename: str = "") -> PipelineResult:
    settings = get_settings()
    _validate_size(len(image_bytes), settings.max_upload_size_mb)

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f"Pillow is required for the image pipeline: {exc}") from exc

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as exc:
        raise ValueError(f"Unable to decode image: {exc}") from exc

    pil_format = img.format or ""
    _validate_format(pil_format, settings.allowed_image_formats)

    width, height = img.size
    if width < settings.min_image_size_pixels or height < settings.min_image_size_pixels:
        raise ValueError(
            f"Image too small: {width}x{height} < min {settings.min_image_size_pixels}px"
        )
    if width > settings.max_image_size_pixels or height > settings.max_image_size_pixels:
        raise ValueError(
            f"Image too large: {width}x{height} > max {settings.max_image_size_pixels}px"
        )

    exif = _extract_exif(img)
    phash = _compute_phash(img)
    sha256 = hashlib.sha256(image_bytes).hexdigest()

    mime_type = _PIL_FORMAT_TO_MIME.get(pil_format.upper(), "application/octet-stream")

    try:
        clean_bytes = _strip_exif_bytes(img, pil_format)
    except Exception:
        logger.exception("Failed to strip EXIF; falling back to original bytes minus metadata warning")
        clean_bytes = image_bytes

    metadata = ImageMetadata(
        width=width,
        height=height,
        format=pil_format,
        mode=img.mode,
        size_bytes=len(image_bytes),
        phash=phash,
        sha256=sha256,
        exif=exif,
    )

    result = PipelineResult(
        input_type=InputType.IMAGE,
        image_metadata=metadata,
    )

    if filename:
        result.trace["filename"] = os.path.basename(filename)

    image_part = LLMImagePart(data=clean_bytes, mime_type=mime_type)

    ocr_text = ""
    try:
        provider = get_provider_for_task("ocr")
        response = provider.generate(ocr.build_request(image_part))
        parsed_ocr = _parse_ocr(response.structured, response.text)
        result.ocr = parsed_ocr
        ocr_text = parsed_ocr.extracted_text
        result.raw_text = ocr_text
        result.trace["ocr"] = {
            "model": response.model,
            "provider": response.provider,
            "latency_ms": response.latency_ms,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    except Exception as exc:
        logger.exception("OCR step failed")
        result.errors.append(f"ocr: {exc}")

    try:
        provider = get_provider_for_task("image_analysis")
        response = provider.generate(image_analysis.build_request(image_part))
        result.image_analysis = _parse_image_analysis(response.structured, response.text)
        result.deepfake_scores = _derive_image_scores(result.image_analysis)
        result.trace["image_analysis"] = {
            "model": response.model,
            "provider": response.provider,
            "latency_ms": response.latency_ms,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    except Exception as exc:
        logger.exception("Image analysis step failed")
        result.errors.append(f"image_analysis: {exc}")

    if ocr_text.strip():
        try:
            provider = get_provider_for_task("claim_extraction")
            response = provider.generate(claim_extraction.build_request(ocr_text))
            result.claims = _parse_claims(response.structured)
            result.trace["claim_extraction"] = {
                "model": response.model,
                "provider": response.provider,
                "latency_ms": response.latency_ms,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except Exception as exc:
            logger.exception("Claim extraction from OCR text failed")
            result.errors.append(f"claim_extraction: {exc}")

    return result
