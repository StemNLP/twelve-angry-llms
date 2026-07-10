"""Elicitation protocols: how a judge is asked and how its answer is parsed.

Two protocols are provided, matching the two label styles found in raw
preference datasets:

- ``ScoringProtocol``: the judge scores each of the K responses on a 1-5
  scale against a shared rubric (UltraFeedback style).
- ``RankingProtocol``: the judge orders all K responses best-to-worst in a
  single pass (Nectar style).

Every judge in a panel must receive the identical prompt for a given
datapoint, so protocols are pure functions of (datapoint, rubric): they hold
no per-judge state. Both parse into a vector of per-response utilities,
higher = better (see ``types``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .errors import ParseError
from .types import PreferenceDatapoint

Message = dict[str, str]

DEFAULT_SCORING_RUBRIC = """\
Score each response on a scale of 1 to 5:
5 - Excellent: accurate, complete, directly follows the instruction, and is honest about any uncertainty.
4 - Good: accurate and follows the instruction, with minor omissions or minor issues of clarity.
3 - Adequate: mostly correct and on-task, but with noticeable gaps, imprecision, or partial instruction-following.
2 - Poor: significant errors, substantial missing content, or clear failure to follow parts of the instruction.
1 - Unacceptable: incorrect, off-topic, misleading, or ignores the instruction.

Judge only the quality of the content. Do not reward length or verbosity for its own sake."""

DEFAULT_RANKING_CRITERIA = """\
Rank the responses from best to worst by overall quality: accuracy, completeness, how well each follows the instruction, and honesty about uncertainty. Judge only the quality of the content. Do not reward length or verbosity for its own sake."""

_SYSTEM = (
    "You are an impartial judge evaluating candidate responses to an instruction. "
    "You follow the evaluation guideline exactly and answer only in the requested JSON format."
)


def _format_responses(datapoint: PreferenceDatapoint) -> str:
    blocks = []
    for i, response in enumerate(datapoint.responses, start=1):
        blocks.append(f"[Response {i}]\n{response}")
    return "\n\n".join(blocks)


def _extract_json_object(raw: str) -> dict:
    """Pull the first JSON object out of a possibly prose-wrapped reply."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for blob in candidates:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ParseError(f"No JSON object found in judge output: {raw[:200]!r}")


_REPAIR = (
    "Your previous reply could not be parsed. Answer again with ONLY the JSON "
    "object in the exact format requested, and nothing else."
)


@dataclass(frozen=True)
class ScoringProtocol:
    """Per-response 1-5 scoring against a shared rubric."""

    rubric: str = DEFAULT_SCORING_RUBRIC
    scale: tuple[int, int] = (1, 5)

    name: str = field(default="scoring", init=False)

    def build_messages(self, datapoint: PreferenceDatapoint) -> list[Message]:
        low, high = self.scale
        k = datapoint.k
        user = (
            f"Evaluation guideline:\n{self.rubric}\n\n"
            f"Instruction:\n{datapoint.prompt}\n\n"
            f"{_format_responses(datapoint)}\n\n"
            f"Score all {k} responses. Reply with ONLY a JSON object of the form "
            f'{{"scores": [s1, s2, ...]}} containing exactly {k} integers from {low} to {high}, '
            f"where the i-th integer is the score of Response i."
        )
        return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]

    def repair_messages(
        self, datapoint: PreferenceDatapoint, failed_raw: str
    ) -> list[Message]:
        return [
            *self.build_messages(datapoint),
            {"role": "assistant", "content": failed_raw},
            {"role": "user", "content": _REPAIR},
        ]

    def parse(self, raw: str, k: int) -> tuple[float, ...]:
        data = _extract_json_object(raw)
        scores = data.get("scores")
        if not isinstance(scores, list) or len(scores) != k:
            raise ParseError(f"Expected a 'scores' list of length {k}, got: {scores!r}")
        low, high = self.scale
        values: list[float] = []
        for s in scores:
            try:
                v = float(s)
            except (TypeError, ValueError):
                raise ParseError(f"Non-numeric score: {s!r}") from None
            if not (low <= v <= high):
                raise ParseError(f"Score {v} outside scale [{low}, {high}]")
            values.append(v)
        return tuple(values)


@dataclass(frozen=True)
class RankingProtocol:
    """K-wise ranking: the judge orders all responses best-to-worst.

    Parsed rankings are converted to utilities: the response ranked r-th of
    K gets utility K + 1 - r, so higher remains better and the downstream
    metrics are shared with the scoring protocol.
    """

    criteria: str = DEFAULT_RANKING_CRITERIA

    name: str = field(default="ranking", init=False)

    def build_messages(self, datapoint: PreferenceDatapoint) -> list[Message]:
        k = datapoint.k
        user = (
            f"Evaluation guideline:\n{self.criteria}\n\n"
            f"Instruction:\n{datapoint.prompt}\n\n"
            f"{_format_responses(datapoint)}\n\n"
            f"Rank all {k} responses from best to worst. Reply with ONLY a JSON object "
            f'of the form {{"ranking": [r1, r2, ...]}} where the list contains each '
            f"response number 1 to {k} exactly once, ordered best first."
        )
        return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]

    def repair_messages(
        self, datapoint: PreferenceDatapoint, failed_raw: str
    ) -> list[Message]:
        return [
            *self.build_messages(datapoint),
            {"role": "assistant", "content": failed_raw},
            {"role": "user", "content": _REPAIR},
        ]

    def parse(self, raw: str, k: int) -> tuple[float, ...]:
        data = _extract_json_object(raw)
        ranking = data.get("ranking")
        if not isinstance(ranking, list) or len(ranking) != k:
            raise ParseError(f"Expected a 'ranking' list of length {k}, got: {ranking!r}")
        try:
            order = [int(r) for r in ranking]
        except (TypeError, ValueError):
            raise ParseError(f"Non-integer entries in ranking: {ranking!r}") from None
        if sorted(order) != list(range(1, k + 1)):
            raise ParseError(f"Ranking is not a permutation of 1..{k}: {order!r}")
        values = [0.0] * k
        for position, response_number in enumerate(order):  # position 0 = best
            values[response_number - 1] = float(k - position)
        return tuple(values)


Protocol = ScoringProtocol | RankingProtocol
