"""Native Anthropic client (optional dependency).

Install with: pip install "twelve-angry-llms[anthropic]"
"""

from __future__ import annotations

import os
from typing import Sequence

from ..errors import MissingDependencyError
from .base import Message, Usage


class AnthropicClient:
    """Client for the Anthropic Messages API.

    ``api_key`` falls back to the environment variable named by
    ``api_key_env`` (default ``ANTHROPIC_API_KEY``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        **client_kwargs,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise MissingDependencyError("anthropic", "anthropic") from None

        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise ValueError(
                f"No API key: pass api_key= or set the {api_key_env} environment variable"
            )
        self._client = AsyncAnthropic(api_key=key, **client_kwargs)
        self.usage = Usage()

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        chat = [m for m in messages if m["role"] != "system"]
        extra = {"system": "\n\n".join(system_parts)} if system_parts else {}
        response = await self._client.messages.create(
            model=model,
            messages=chat,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        self.usage.add(response.usage.input_tokens, response.usage.output_tokens)
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
