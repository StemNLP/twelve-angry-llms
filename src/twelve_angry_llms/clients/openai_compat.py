"""Client for OpenAI and any OpenAI-compatible endpoint.

Because ``base_url`` is configurable, this one client covers OpenAI itself
plus OpenRouter, Together, Groq, Fireworks, local vLLM or Ollama servers,
Hugging Face inference endpoints, and anything else that speaks the
chat-completions API. Judges hosted behind different endpoints simply use
different client instances.
"""

from __future__ import annotations

import json
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

    ``extra_body`` is merged into every request body. Aggregators use it
    for routing controls - on OpenRouter, pinning a judge to one upstream
    provider and precision for reproducibility looks like::

        extra_body={"provider": {"order": ["together"], "allow_fallbacks": False,
                                 "quantizations": ["fp16"]}}

    Anything that changes what model actually serves the request belongs
    here, because ``extra_body`` (and ``base_url``) are folded into the
    response-cache key via ``cache_salt``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        extra_body: dict | None = None,
        **client_kwargs,
    ) -> None:
        from openai import AsyncOpenAI

        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise ValueError(
                f"No API key: pass api_key= or set the {api_key_env} environment variable"
            )
        self._client = AsyncOpenAI(api_key=key, base_url=base_url, **client_kwargs)
        self.extra_body = extra_body
        self.usage = Usage()

    @property
    def cache_salt(self) -> str:
        return json.dumps(
            {"base_url": str(self._client.base_url), "extra_body": self.extra_body},
            sort_keys=True,
        )

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
            extra_body=self.extra_body,
        )
        if response.usage is not None:
            self.usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
        content = response.choices[0].message.content
        return content or ""


class OpenRouterClient(OpenAICompatibleClient):
    """OpenAI-compatible client preconfigured for OpenRouter.

    The recommended way to run a cross-family panel behind a single API
    key (``OPENROUTER_API_KEY``). For reproducible runs, pin each judge's
    upstream provider and precision via ``extra_body`` - see
    ``OpenAICompatibleClient`` for the format.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        extra_body: dict | None = None,
        **client_kwargs,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            api_key_env=api_key_env,
            extra_body=extra_body,
            **client_kwargs,
        )
