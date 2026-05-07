"""OpenAI Chat Completions teacher (GPT-4o-mini default)."""
from __future__ import annotations

import os
from typing import Any

from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential,
)

from src.teachers.base import BaseTeacher, TeacherResponse
from src.teachers.prompts import PromptSpec


class OpenAITeacher(BaseTeacher):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3,
                 max_retries: int = 5, api_key: str | None = None):
        super().__init__(model=model, temperature=temperature, max_retries=max_retries)
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        # Build a dynamic retry decorator scoped to this instance's max_retries.
        from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
        self._retryable = (RateLimitError, APIConnectionError, APITimeoutError, APIError)

    def summarize(self, article: str, prompt: PromptSpec) -> TeacherResponse:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(min=1, max=30),
            retry=retry_if_exception_type(self._retryable),
        )
        def _call() -> Any:
            return self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=prompt.max_output_tokens,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user_template.format(article=article)},
                ],
            )

        resp = _call()
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        usage = resp.usage
        return TeacherResponse(
            summary=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw={"finish_reason": choice.finish_reason, "model": resp.model},
        )
