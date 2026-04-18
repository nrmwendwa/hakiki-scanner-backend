"""Prompt and schema for OCR-style text extraction from images."""

from ..schemas import LLMImagePart, LLMMessage, LLMRequest


SYSTEM_PROMPT = (
    "You are an OCR and visual text analysis assistant. Transcribe ALL "
    "visible text in the provided image verbatim, preserving original "
    "wording, punctuation, capitalization, and line order. Do not translate, "
    "summarize, correct spelling, or omit text that looks like noise.\n\n"
    "After transcription, identify any discrete visible claims or "
    "statements the image appears to make (e.g. headlines, captions, "
    "quotes, statistics) and list them as short standalone sentences.\n\n"
    "Detect the primary language of the visible text using an ISO-639-1 "
    "code when possible (e.g. 'en', 'sw'). If the image contains no text, "
    "set has_text to false and return empty strings or arrays for the "
    "remaining fields.\n\n"
    "Respond ONLY with JSON matching the provided schema."
)


RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "extracted_text": {"type": "string"},
        "language": {"type": "string"},
        "visible_claims": {
            "type": "array",
            "items": {"type": "string"},
        },
        "has_text": {"type": "boolean"},
    },
    "required": ["extracted_text", "language", "visible_claims", "has_text"],
}


def build_request(image: LLMImagePart) -> LLMRequest:
    return LLMRequest(
        task="ocr",
        messages=[
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    "Extract all visible text from this image verbatim, "
                    "then list any visible claims. Return JSON only."
                ),
            ),
        ],
        images=[image],
        response_schema=RESPONSE_SCHEMA,
    )
