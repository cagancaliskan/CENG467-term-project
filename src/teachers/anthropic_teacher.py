"""Anthropic Messages API teacher (Claude Haiku 3.5 default)."""
from __future__ import annotations

import os
from typing import Any

from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential,
)

from src.teachers.base import BaseTeacher, TeacherResponse
from src.teachers.prompts import PromptSpec


class AnthropicTeacher(BaseTeacher):
    name = "anthropic"

    def __init__(self, model: str = "claude-3-haiku-20240307", temperature: float = 0.3,
                 max_retries: int = 5, api_key: str | None = None):
        super().__init__(model=model, temperature=temperature, max_retries=max_retries)
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        from anthropic import APIConnectionError, APIError, APITimeoutError, RateLimitError
        self._retryable = (RateLimitError, APIConnectionError, APITimeoutError, APIError)

    def summarize(self, article: str, prompt: PromptSpec) -> TeacherResponse:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(min=1, max=30),
            retry=retry_if_exception_type(self._retryable),
        )
        def _call() -> Any:
            return self._client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=prompt.max_output_tokens,
                system=prompt.system,
                messages=[{"role": "user",
                            "content": prompt.user_template.format(article=article)}],
            )

        msg = _call()
        # Anthropic returns a list of content blocks; concatenate the text-typed ones.
        text_parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        text = "".join(text_parts).strip()
        usage = msg.usage
        return TeacherResponse(
            summary=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            raw={"stop_reason": msg.stop_reason, "model": msg.model},
        )
