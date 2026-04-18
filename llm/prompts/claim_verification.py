"""Prompt and schema for verifying a claim against web-search snippets."""

from ..schemas import LLMMessage, LLMRequest


SYSTEM_PROMPT = (
    "You are a fact-checking assistant. You will receive ONE claim and a list of "
    "web-search results (title, URL, snippet, and a 'trusted' flag indicating "
    "whether the domain is an official or reputable source).\n\n"
    "Decide whether the snippets support, contradict, or are insufficient to "
    "judge the claim. Prefer trusted sources when available; treat untrusted "
    "sources as weak evidence only.\n\n"
    "Return exactly one of these labels:\n"
    "- 'imethibitishwa' if the claim is clearly supported by (a) at least one "
    "trusted snippet, OR (b) strong agreement across three or more independent "
    "untrusted snippets with no contradicting evidence.\n"
    "- 'ya_uongo' if the claim is clearly contradicted by at least one trusted "
    "snippet, or by strong multi-source agreement among untrusted snippets.\n"
    "- 'haijathibitishwa' only if evidence is genuinely missing or mixed.\n\n"
    "Choose the single best supporting-or-contradicting source and return its "
    "name (domain or publisher) and URL. Rationale must be in Swahili "
    "(Kiswahili). Confidence is 0-100 reflecting how strong the evidence is.\n\n"
    "Respond ONLY with JSON matching the provided schema."
)


RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["imethibitishwa", "ya_uongo", "haijathibitishwa"],
        },
        "confidence": {"type": "number"},
        "best_source": {"type": "string"},
        "best_url": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "confidence", "best_source", "best_url", "rationale"],
}


def _format_results(results: list[dict]) -> str:
    if not results:
        return "(no results)"
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        trusted = "TRUSTED" if r.get("trusted") else "untrusted"
        lines.append(
            f"[{i}] ({trusted}) {r.get('title', '')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Snippet: {r.get('snippet', '')}"
        )
    return "\n\n".join(lines)


def build_request(claim: str, results: list[dict]) -> LLMRequest:
    user_content = (
        f"Claim:\n{claim}\n\n"
        f"Web search results:\n{_format_results(results)}\n\n"
        "Return JSON only."
    )
    return LLMRequest(
        task="claim_verification",
        messages=[
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_content),
        ],
        response_schema=RESPONSE_SCHEMA,
    )
