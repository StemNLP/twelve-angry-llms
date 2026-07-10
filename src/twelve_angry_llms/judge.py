"""A judge: one model behind one client, with retry and parse-repair."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .clients.base import LLMClient
from .errors import JudgeError, ParseError
from .protocols import Protocol
from .types import JudgeAnnotation, PreferenceDatapoint


@dataclass
class Judge:
    """One panel member.

    ``name`` identifies the judge in results and defaults to the model id;
    give explicit names when the same model appears twice (e.g. in a
    self-consistency arm). Temperature defaults to 0 so that runs are as
    deterministic as the provider allows.
    """

    model: str
    client: LLMClient
    name: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    api_retries: int = 3
    parse_repairs: int = 1
    retry_base_delay: float = 2.0

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.model

    async def _complete(self, messages) -> str:
        last_error: Exception | None = None
        for attempt in range(self.api_retries + 1):
            try:
                return await self.client.complete(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:  # transport/provider errors: back off and retry
                last_error = e
                if attempt < self.api_retries:
                    await asyncio.sleep(self.retry_base_delay * 2**attempt)
        raise JudgeError(f"{self.name}: API call failed after retries: {last_error}")

    async def annotate(
        self, datapoint: PreferenceDatapoint, protocol: Protocol
    ) -> JudgeAnnotation:
        """Ask this judge to annotate one datapoint under a protocol.

        On a parse failure the judge is re-asked with its unparseable reply
        and a format reminder appended (up to ``parse_repairs`` times).
        Raises JudgeError when no usable annotation could be obtained.
        """
        raw = await self._complete(protocol.build_messages(datapoint))
        for repair in range(self.parse_repairs + 1):
            try:
                values = protocol.parse(raw, datapoint.k)
                return JudgeAnnotation(
                    judge=self.name,
                    model=self.model,
                    protocol=protocol.name,
                    values=values,
                    raw=raw,
                )
            except ParseError as e:
                if repair == self.parse_repairs:
                    raise JudgeError(f"{self.name}: unparseable after repairs: {e}") from e
                raw = await self._complete(protocol.repair_messages(datapoint, raw))
        raise AssertionError("unreachable")
