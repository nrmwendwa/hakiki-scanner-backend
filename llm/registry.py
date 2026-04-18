"""Provider registry and task-based routing."""

from functools import lru_cache

from config import get_settings

from .base import LLMProvider
from .gemini import GeminiProvider


@lru_cache(maxsize=8)
def get_provider(name: str) -> LLMProvider:
    """Get a provider instance by short name.

    Supported names: "gemini-text", "gemini-vision".
    Extend here when adding new providers (anthropic, openai, local).
    """
    settings = get_settings()
    if name == "gemini-text":
        return GeminiProvider(model=settings.llm.gemini_model)
    if name == "gemini-vision":
        return GeminiProvider(model=settings.llm.gemini_vision_model)
    raise ValueError(f"Unknown LLM provider: {name}")


def get_provider_for_task(task: str) -> LLMProvider:
    """Route a logical task to a concrete provider via config."""
    settings = get_settings()
    route_map = {
        "ocr": settings.llm.route_ocr,
        "image_analysis": settings.llm.route_image_analysis,
        "claim_extraction": settings.llm.route_claim_extraction,
        "claim_verification": settings.llm.route_claim_verification,
        "reasoning": settings.llm.route_reasoning,
    }
    if task not in route_map:
        raise ValueError(f"Unknown task: {task}. Known: {list(route_map)}")
    return get_provider(route_map[task])
