"""Abstract provider interface for the LLM gateway."""

from abc import ABC, abstractmethod
from .schemas import LLMRequest, LLMResponse


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, req: LLMRequest) -> LLMResponse:
        ...
