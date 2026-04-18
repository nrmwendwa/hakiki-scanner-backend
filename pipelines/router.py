"""Input type detection and routing for the multimodal pipelines."""

from __future__ import annotations

import logging
import os
from typing import Optional

from config import get_settings

from .schemas import InputType

logger = logging.getLogger(__name__)

try:
    import magic as _magic
except Exception:
    _magic = None


_IMAGE_MIME_PREFIXES = ("image/",)
_DOCUMENT_MIMES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
}
_TEXT_MIME_PREFIXES = ("text/",)

_IMAGE_EXTS = {"jpeg", "jpg", "png", "webp", "gif", "bmp", "tiff"}
_DOCUMENT_EXTS = {"pdf", "txt", "md", "markdown"}


def _sniff_magic_bytes(raw: bytes) -> Optional[InputType]:
    if not raw:
        return None
    head = raw[:16]

    if len(head) >= 3 and head[0:3] == b"\xff\xd8\xff":
        return InputType.IMAGE
    if len(head) >= 8 and head[0:8] == b"\x89PNG\r\n\x1a\n":
        return InputType.IMAGE
    if len(head) >= 12 and head[0:4] == b"RIFF" and head[8:12] == b"WEBP":
        return InputType.IMAGE
    if len(head) >= 6 and (head[0:6] == b"GIF87a" or head[0:6] == b"GIF89a"):
        return InputType.IMAGE
    if len(head) >= 2 and head[0:2] == b"BM":
        return InputType.IMAGE
    if len(head) >= 4 and (head[0:4] == b"II*\x00" or head[0:4] == b"MM\x00*"):
        return InputType.IMAGE

    if len(head) >= 4 and head[0:4] == b"%PDF":
        return InputType.DOCUMENT

    return None


def _from_content_type(content_type: str) -> InputType:
    ct = content_type.lower().split(";", 1)[0].strip()
    if any(ct.startswith(p) for p in _IMAGE_MIME_PREFIXES):
        return InputType.IMAGE
    if ct in _DOCUMENT_MIMES:
        return InputType.DOCUMENT
    if any(ct.startswith(p) for p in _TEXT_MIME_PREFIXES):
        return InputType.TEXT
    return InputType.UNKNOWN


def _from_filename(filename: str) -> InputType:
    _, ext = os.path.splitext(filename)
    ext = ext.lower().lstrip(".")
    if not ext:
        return InputType.UNKNOWN
    if ext in _IMAGE_EXTS:
        return InputType.IMAGE
    if ext == "pdf":
        return InputType.DOCUMENT
    if ext in {"txt", "md", "markdown"}:
        return InputType.DOCUMENT
    return InputType.UNKNOWN


def detect_input_type(
    filename: Optional[str],
    content_type: Optional[str],
    raw_bytes: Optional[bytes],
) -> InputType:
    if raw_bytes:
        if _magic is not None:
            try:
                sniffed = _magic.from_buffer(raw_bytes[:4096], mime=True)
                if sniffed:
                    guess = _from_content_type(sniffed)
                    if guess != InputType.UNKNOWN:
                        return guess
            except Exception:
                logger.debug("python-magic sniff failed, falling back to headers", exc_info=True)
        header_guess = _sniff_magic_bytes(raw_bytes)
        if header_guess is not None:
            return header_guess

    if content_type:
        guess = _from_content_type(content_type)
        if guess != InputType.UNKNOWN:
            return guess

    if filename:
        guess = _from_filename(filename)
        if guess != InputType.UNKNOWN:
            return guess

    return InputType.UNKNOWN


def route_input(
    *,
    filename: Optional[str],
    content_type: Optional[str],
    data: Optional[bytes],
    text: Optional[str],
) -> tuple[InputType, dict]:
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    if data is not None and len(data) > max_bytes:
        raise ValueError(
            f"Upload exceeds max size of {settings.max_upload_size_mb} MB"
        )

    if data is None and text is not None and text.strip():
        return InputType.TEXT, {"text": text}

    input_type = detect_input_type(filename, content_type, data)

    if input_type == InputType.IMAGE:
        if data is None:
            raise ValueError("Image input requires binary data")
        return InputType.IMAGE, {"image_bytes": data}

    if input_type == InputType.DOCUMENT:
        if data is None:
            raise ValueError("Document input requires binary data")
        return InputType.DOCUMENT, {"document_bytes": data}

    if input_type == InputType.TEXT:
        if text is not None:
            return InputType.TEXT, {"text": text}
        if data is not None:
            try:
                decoded = data.decode("utf-8")
            except UnicodeDecodeError:
                decoded = data.decode("latin-1", errors="replace")
            return InputType.TEXT, {"text": decoded}

    return InputType.UNKNOWN, {}
