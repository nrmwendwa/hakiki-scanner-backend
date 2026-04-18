"""Prompt and schema for forensic-style image analysis."""

from ..schemas import LLMImagePart, LLMMessage, LLMRequest


SYSTEM_PROMPT = (
    "You are an image forensics and context analysis assistant. Examine "
    "the provided image and produce a structured assessment.\n\n"
    "Cover three dimensions:\n"
    "1. Scene description: a concise factual description of what is depicted "
    "(subjects, setting, actions, notable objects).\n"
    "2. Manipulation signals: visual cues suggesting AI generation or "
    "digital manipulation. Look for warped anatomy (hands, teeth, ears), "
    "inconsistent lighting or shadows, unnatural textures, blurred or "
    "nonsensical backgrounds, broken text, symmetry artifacts, and splicing "
    "seams. List each distinct signal as a short phrase; if none are "
    "evident, return an empty list.\n"
    "3. Source indicators: any logos, watermarks, channel bugs, URLs, "
    "captions, or stylistic markers that hint at the origin or publisher. "
    "Return an empty list if none are present.\n\n"
    "Finally, rate the overall likelihood that the image is AI-generated "
    "or substantially manipulated as 'low', 'medium', or 'high'. Be "
    "conservative: choose 'low' when no clear signals are present.\n\n"
    "IMPORTANT: Write all free-text fields (scene_description, manipulation_signals, "
    "source_indicators) in Swahili (Kiswahili). Keep the ai_generated_likelihood "
    "value exactly as 'low', 'medium', or 'high' (do not translate it).\n\n"
    "Respond ONLY with JSON matching the provided schema."
)


RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "scene_description": {"type": "string"},
        "manipulation_signals": {
            "type": "array",
            "items": {"type": "string"},
        },
        "source_indicators": {
            "type": "array",
            "items": {"type": "string"},
        },
        "ai_generated_likelihood": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
    },
    "required": [
        "scene_description",
        "manipulation_signals",
        "source_indicators",
        "ai_generated_likelihood",
    ],
}


def build_request(image: LLMImagePart) -> LLMRequest:
    return LLMRequest(
        task="image_analysis",
        messages=[
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    "Analyze this image for manipulation signals, source "
                    "indicators, and scene content. Return JSON only."
                ),
            ),
        ],
        images=[image],
        response_schema=RESPONSE_SCHEMA,
    )
