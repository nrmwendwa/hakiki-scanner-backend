"""Prompt and schema for extracting discrete factual claims from text."""

from ..schemas import LLMMessage, LLMRequest


SYSTEM_PROMPT = (
    "You are a claim extraction assistant. Given input text, identify every "
    "discrete, verifiable factual claim it makes. Split compound sentences "
    "so that each extracted claim expresses exactly one assertion. Ignore "
    "opinions, rhetorical questions, and purely subjective statements.\n\n"
    "IMPORTANT: All natural-language string values you return (statement, "
    "subject, predicate, object, topic) MUST be written in Swahili (Kiswahili). "
    "If the input text is in English or another language, translate the extracted "
    "claims into Swahili before returning them. Keep proper nouns and numeric "
    "values in their original form.\n\n"
    "For each claim return:\n"
    "- statement: the claim as a clean standalone Swahili sentence\n"
    "- subject: the primary entity the claim is about (in Swahili)\n"
    "- predicate: the relation or verb phrase (in Swahili)\n"
    "- object: the target entity, value, or condition (in Swahili)\n"
    "- numeric_value: any numeric quantity mentioned in the claim, or null\n"
    "- date: any date or time reference in ISO-8601 where possible, or null\n"
    "- topic: a short Swahili topical label (e.g. afya, uchumi, siasa, michezo)\n\n"
    "Respond ONLY with JSON matching the provided schema. If no claims are "
    "present, return an empty claims array."
)


RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "numeric_value": {"type": "number", "nullable": True},
                    "date": {"type": "string", "nullable": True},
                    "topic": {"type": "string"},
                },
                "required": [
                    "statement",
                    "subject",
                    "predicate",
                    "object",
                    "topic",
                ],
            },
        }
    },
    "required": ["claims"],
}


def build_request(text: str) -> LLMRequest:
    return LLMRequest(
        task="claim_extraction",
        messages=[
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    "Extract all factual claims from the following text:\n\n"
                    f"{text}"
                ),
            ),
        ],
        response_schema=RESPONSE_SCHEMA,
    )
