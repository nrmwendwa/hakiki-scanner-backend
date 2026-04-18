"""Gemini provider implementation using the google-genai unified SDK."""

import json
import logging
import re
import time
from typing import Any


_CODE_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*|\s*```\s*$", re.MULTILINE)


def _parse_json_lenient(text: str) -> Any:
    """Parse the first JSON value in ``text``, tolerating markdown fences and trailing content."""
    cleaned = _CODE_FENCE_RE.sub("", text).strip()
    start = min(
        (i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1),
        default=-1,
    )
    if start > 0:
        cleaned = cleaned[start:]
    value, _end = json.JSONDecoder().raw_decode(cleaned)
    return value

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import get_settings

from .base import LLMProvider
from .schemas import LLMRequest, LLMResponse, LLMUsage

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM call fails unrecoverably."""


class _TransientLLMError(RuntimeError):
    """Internal marker for retryable failures."""


_NON_RETRYABLE_STATUS = ("400", "401", "403", "404", "permission_denied",
                         "unauthenticated", "invalid_argument", "not_found",
                         "failed_precondition", "service_disabled")

def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (ValueError, TypeError, PermissionError)):
        return False
    msg = str(exc).lower()
    if any(m in msg for m in _NON_RETRYABLE_STATUS):
        return False
    transient_markers = (
        "timeout",
        "timed out",
        "temporar",
        "unavailable",
        "deadline",
        "rate limit",
        "resource exhausted",
        "429",
        "500",
        "502",
        "503",
        "504",
        "connection",
        "reset",
    )
    if any(m in msg for m in transient_markers):
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name or "connection" in name:
        return True
    return False


class GeminiProvider(LLMProvider):
    """Google Gemini provider. One instance per concrete model."""

    name = "gemini"

    def __init__(self, model: str):
        self.model = model
        settings = get_settings()
        self._settings = settings
        api_key = settings.llm.require_gemini_key()
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=settings.llm.request_timeout_s * 1000),
        )

    def generate(self, req: LLMRequest) -> LLMResponse:
        settings = self._settings
        max_attempts = max(1, settings.llm.max_retries)
        backoff = max(0.1, settings.llm.retry_backoff_s)

        @retry(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=backoff),
            retry=retry_if_exception_type(_TransientLLMError),
        )
        def _invoke() -> Any:
            try:
                return self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                if _is_transient(exc):
                    raise _TransientLLMError(str(exc)) from exc
                raise

        system_parts: list[str] = []
        contents: list[Any] = []
        for msg in req.messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                contents.append(msg.content)

        for image in req.images:
            contents.append(
                types.Part.from_bytes(data=image.data, mime_type=image.mime_type)
            )

        config_kwargs: dict[str, Any] = {
            "temperature": req.temperature,
            "max_output_tokens": req.max_output_tokens,
        }
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)
        if req.response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = req.response_schema

        config = types.GenerateContentConfig(**config_kwargs)

        start = time.perf_counter()
        try:
            resp = _invoke()
        except _TransientLLMError as exc:
            raise LLMError(f"Gemini call failed after retries: {exc}") from exc
        except (ValueError, TypeError) as exc:
            raise
        except Exception as exc:
            raise LLMError(f"Gemini call failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - start) * 1000)

        text = getattr(resp, "text", None) or ""

        structured = None
        if req.response_schema is not None and text:
            try:
                structured = _parse_json_lenient(text)
            except json.JSONDecodeError as exc:
                raise LLMError(
                    f"Gemini returned non-JSON content despite schema: {exc}"
                ) from exc

        usage_meta = getattr(resp, "usage_metadata", None)
        usage = LLMUsage(
            input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
        )

        logger.info(
            "llm.generate task=%s provider=%s model=%s latency_ms=%d input_tokens=%d output_tokens=%d",
            req.task,
            self.name,
            self.model,
            latency_ms,
            usage.input_tokens,
            usage.output_tokens,
        )

        return LLMResponse(
            text=text,
            structured=structured,
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            usage=usage,
        )
