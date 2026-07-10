"""Client interface every provider backend implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

Message = dict[str, str]


@dataclass
class Usage:
    """Cumulative token usage across all calls made through a client."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1


class LLMClient(Protocol):
    """Anything that can turn chat messages into a completion string.

    Implementations should expose a ``usage: Usage`` attribute so panels can
    report cost, but only ``complete`` is required.
    """

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str: ...
