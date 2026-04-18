"""Document pipeline: PDF / TXT / MD -> text -> claim extraction."""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

from config import get_settings

from .schemas import InputType, PipelineResult
from .text_pipeline import run_text_pipeline

logger = logging.getLogger(__name__)


def _detect_format(doc_bytes: bytes, filename: str) -> Optional[str]:
    head = doc_bytes[:8] if doc_bytes else b""
    if len(head) >= 4 and head[0:4] == b"%PDF":
        return "pdf"

    _, ext = os.path.splitext(filename or "")
    ext = ext.lower().lstrip(".")
    if ext in {"pdf", "txt", "md", "markdown"}:
        return "md" if ext == "markdown" else ext

    if doc_bytes:
        try:
            doc_bytes.decode("utf-8")
            return "txt"
        except UnicodeDecodeError:
            return None
    return None


def _extract_pdf_text(doc_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError(f"pypdf is required for PDF extraction: {exc}") from exc

    reader = PdfReader(io.BytesIO(doc_bytes))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            logger.exception("PDF page extraction failed")
            pages.append("")
    return "\n\n".join(p for p in pages if p)


def _decode_text(doc_bytes: bytes) -> str:
    try:
        return doc_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return doc_bytes.decode("latin-1", errors="replace")


def run_document_pipeline(doc_bytes: bytes, filename: str = "") -> PipelineResult:
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if not doc_bytes:
        raise ValueError("Empty document payload")
    if len(doc_bytes) > max_bytes:
        raise ValueError(f"Document exceeds max size of {settings.max_upload_size_mb} MB")

    fmt = _detect_format(doc_bytes, filename)
    allowed = tuple(f.lower() for f in settings.allowed_document_formats)
    if fmt not in allowed:
        raise ValueError(
            f"Document format '{fmt}' not allowed. Allowed: {list(allowed)}"
        )

    if fmt == "pdf":
        extracted = _extract_pdf_text(doc_bytes)
    else:
        extracted = _decode_text(doc_bytes)

    if not extracted or not extracted.strip():
        raise ValueError("Document contained no extractable text")

    result = run_text_pipeline(extracted)
    result.input_type = InputType.DOCUMENT
    if filename:
        result.trace["filename"] = os.path.basename(filename)
    result.trace["document_format"] = fmt
    return result
