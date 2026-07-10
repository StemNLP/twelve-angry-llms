"""Phase 1 controlled corruptions (RESEARCH_PLAN.md §6).

Each corruption takes a datapoint with a known-clean reference ordering and
returns a corrupted variant plus a tag recording exactly what was done.
Two corruptions act on the *label* (which pair is shipped as
chosen/rejected) and two act on the *responses* themselves:

- ``swap_adjacent``: chosen/rejected drawn from two adjacent-ranked
  responses, in the wrong order (a subtle label error).
- ``flip_extremes``: the clear winner and clear loser shipped reversed
  (a blatant label error).
- ``replace_offtopic``: one response replaced with an off-topic response
  taken from a different prompt.
- ``pad_weak``: the weakest response padded with verbose filler, testing
  whether the panel resists length bias.

The reference ordering is a list of response indices, best first.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from twelve_angry_llms import PreferenceDatapoint

CORRUPTION_TYPES = ("swap_adjacent", "flip_extremes", "replace_offtopic", "pad_weak")

_FILLER = (
    " To elaborate further and ensure this answer is as thorough and complete as "
    "possible, it is worth restating the key considerations in additional detail, "
    "since a comprehensive treatment of the question demands that every relevant "
    "aspect be covered from multiple angles and perspectives."
)


@dataclass(frozen=True)
class CorruptedDatapoint:
    """A datapoint with an injected, known corruption.

    ``chosen_index``/``rejected_index`` are the (possibly wrong) labels the
    corrupted datapoint ships with; for clean datapoints they follow the
    reference ordering.
    """

    datapoint: PreferenceDatapoint
    corruption: str | None  # None = clean control
    chosen_index: int
    rejected_index: int


def clean(dp: PreferenceDatapoint, ref_order: list[int]) -> CorruptedDatapoint:
    return CorruptedDatapoint(dp, None, ref_order[0], ref_order[-1])


def swap_adjacent(
    dp: PreferenceDatapoint, ref_order: list[int], rng: random.Random
) -> CorruptedDatapoint:
    """Label error: pair two adjacent-ranked responses, better one rejected."""
    i = rng.randrange(len(ref_order) - 1)
    better, worse = ref_order[i], ref_order[i + 1]
    return CorruptedDatapoint(dp, "swap_adjacent", chosen_index=worse, rejected_index=better)


def flip_extremes(dp: PreferenceDatapoint, ref_order: list[int]) -> CorruptedDatapoint:
    """Label error: the clear winner and clear loser shipped reversed."""
    return CorruptedDatapoint(
        dp, "flip_extremes", chosen_index=ref_order[-1], rejected_index=ref_order[0]
    )


def replace_offtopic(
    dp: PreferenceDatapoint, ref_order: list[int], distractor: str
) -> CorruptedDatapoint:
    """Response error: the shipped chosen response is replaced with an
    off-topic response (pass one taken from a different prompt)."""
    chosen = ref_order[0]
    responses = list(dp.responses)
    responses[chosen] = distractor
    corrupted = PreferenceDatapoint(
        prompt=dp.prompt,
        responses=tuple(responses),
        id=f"{dp.id}-offtopic",
        meta=dict(dp.meta),
    )
    return CorruptedDatapoint(
        corrupted, "replace_offtopic", chosen_index=chosen, rejected_index=ref_order[-1]
    )


def pad_weak(
    dp: PreferenceDatapoint, ref_order: list[int], repeats: int = 3
) -> CorruptedDatapoint:
    """Length bait: the weakest response is padded with verbose filler and
    shipped as chosen. A length-biased judge scores it up; the label is wrong."""
    weakest = ref_order[-1]
    responses = list(dp.responses)
    responses[weakest] = responses[weakest] + _FILLER * repeats
    corrupted = PreferenceDatapoint(
        prompt=dp.prompt,
        responses=tuple(responses),
        id=f"{dp.id}-padded",
        meta=dict(dp.meta),
    )
    return CorruptedDatapoint(
        corrupted, "pad_weak", chosen_index=weakest, rejected_index=ref_order[1]
    )


def corrupt_pool(
    pool: list[tuple[PreferenceDatapoint, list[int]]],
    corrupt_fraction: float = 0.5,
    seed: int = 0,
) -> list[CorruptedDatapoint]:
    """Corrupt a fraction of a pool, cycling through the four corruption
    types in equal shares; the rest are clean controls. Off-topic
    distractors are drawn from other prompts in the pool."""
    rng = random.Random(seed)
    pool = list(pool)
    rng.shuffle(pool)
    n_corrupt = int(len(pool) * corrupt_fraction)

    out: list[CorruptedDatapoint] = []
    for idx, (dp, ref_order) in enumerate(pool):
        if idx >= n_corrupt:
            out.append(clean(dp, ref_order))
            continue
        kind = CORRUPTION_TYPES[idx % len(CORRUPTION_TYPES)]
        if kind == "swap_adjacent":
            out.append(swap_adjacent(dp, ref_order, rng))
        elif kind == "flip_extremes":
            out.append(flip_extremes(dp, ref_order))
        elif kind == "replace_offtopic":
            other, other_order = pool[(idx + 1) % len(pool)]
            distractor = other.responses[other_order[0]]
            out.append(replace_offtopic(dp, ref_order, distractor))
        else:
            out.append(pad_weak(dp, ref_order))
    rng.shuffle(out)
    return out
