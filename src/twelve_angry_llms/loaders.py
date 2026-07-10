"""Loaders for the raw (non-binarized) preference datasets.

IJA needs the K candidate responses back, so these load the raw releases
that still contain them - not the binarized DPO exports. Requires the
'data' extra: pip install "twelve-angry-llms[data]".
"""

from __future__ import annotations

import random
from typing import Iterable

from .errors import MissingDependencyError
from .types import PreferenceDatapoint


def _load_dataset(name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError:
        raise MissingDependencyError("datasets", "data") from None
    return load_dataset(name, split=split)


def _sample(rows: Iterable[dict], limit: int | None, seed: int) -> list[dict]:
    rows = list(rows)
    if limit is None or limit >= len(rows):
        return rows
    return random.Random(seed).sample(rows, limit)


def load_ultrafeedback(
    split: str = "train", limit: int | None = None, seed: int = 0
) -> list[PreferenceDatapoint]:
    """Raw UltraFeedback (openbmb/UltraFeedback): 4 completions per prompt.

    Each datapoint keeps the source models and the original single-judge
    overall scores in ``meta`` so margin baselines can be computed without
    re-judging. Rows with fewer than 2 completions are dropped.
    """
    rows = _sample(_load_dataset("openbmb/UltraFeedback", split), limit, seed)
    datapoints = []
    for i, row in enumerate(rows):
        completions = row.get("completions") or []
        responses = [c["response"] for c in completions if c.get("response")]
        if len(responses) < 2:
            continue
        datapoints.append(
            PreferenceDatapoint(
                prompt=row["instruction"],
                responses=tuple(responses),
                id=f"ultrafeedback-{split}-{i}",
                meta={
                    "source": row.get("source"),
                    "models": [c.get("model") for c in completions],
                    "original_scores": [c.get("overall_score") for c in completions],
                },
            )
        )
    return datapoints


def load_nectar(
    split: str = "train", limit: int | None = None, seed: int = 0
) -> list[PreferenceDatapoint]:
    """Raw Nectar (berkeley-nest/Nectar): 7 ranked answers per prompt.

    The original GPT-4 K-wise ranks are kept in ``meta['original_ranks']``
    (parallel to ``responses``; 1 = best). Rows with fewer than 2 answers
    are dropped.
    """
    rows = _sample(_load_dataset("berkeley-nest/Nectar", split), limit, seed)
    datapoints = []
    for i, row in enumerate(rows):
        answers = row.get("answers") or []
        answers = [a for a in answers if a.get("answer")]
        if len(answers) < 2:
            continue
        datapoints.append(
            PreferenceDatapoint(
                prompt=row["prompt"],
                responses=tuple(a["answer"] for a in answers),
                id=f"nectar-{split}-{i}",
                meta={
                    "source": row.get("source"),
                    "models": [a.get("model") for a in answers],
                    "original_ranks": [a.get("rank") for a in answers],
                    "turns": row.get("turns"),
                },
            )
        )
    return datapoints
