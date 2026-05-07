"""Teacher abstraction so OpenAI and Anthropic clients are interchangeable."""
from __future__ import annotations

import abc
from dataclasses import dataclass

from src.teachers.prompts import PromptSpec


@dataclass
class TeacherResponse:
    summary: str
    input_tokens: int
    output_tokens: int
    raw: dict   # provider-specific metadata, useful for debugging


class BaseTeacher(abc.ABC):
    name: str = "base"

    def __init__(self, model: str, temperature: float = 0.3, max_retries: int = 5):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    @abc.abstractmethod
    def summarize(self, article: str, prompt: PromptSpec) -> TeacherResponse:
        """Return a single summary for one article. Raise on hard failure."""

    def cache_key_components(self) -> dict[str, str | float]:
        """Fields to include in the on-disk cache key alongside article id + prompt."""
        return {"teacher": self.name, "model": self.model, "temperature": self.temperature}
