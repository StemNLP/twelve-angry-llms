"""Client for OpenAI and any OpenAI-compatible endpoint.

Because ``base_url`` is configurable, this one client covers OpenAI itself
plus OpenRouter, Together, Groq, Fireworks, local vLLM or Ollama servers,
Hugging Face inference endpoints, and anything else that speaks the
chat-completions API. Judges hosted behind different endpoints simply use
different client instances.
"""

from __future__ import annotations

import os
from typing import Sequence

from .base import Message, Usage


class OpenAICompatibleClient:
    """Chat-completions client.

    ``api_key`` falls back to the environment variable named by
    ``api_key_env`` (default ``OPENAI_API_KEY``); pass a different env name
    for other providers, e.g. ``api_key_env="OPENROUTER_API_KEY"`` with
    ``base_url="https://openrouter.ai/api/v1"``. Local servers that need no
    key can pass ``api_key="-"``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        **client_kwargs,
    ) -> None:
        from openai import AsyncOpenAI

        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise ValueError(
                f"No API key: pass api_key= or set the {api_key_env} environment variable"
            )
        self._client = AsyncOpenAI(api_key=key, base_url=base_url, **client_kwargs)
        self.usage = Usage()

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response.usage is not None:
            self.usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
        content = response.choices[0].message.content
        return content or ""
