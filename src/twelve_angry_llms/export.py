"""Export annotated results to the (prompt, chosen, rejected) interface.

Almost every DPO / reward-model trainer consumes TRL's
``(prompt, chosen, rejected)`` schema. The exporter emits exactly that,
with the reliability signals carried alongside as metadata columns:

- ``prompt_ija``: panel agreement over all K candidates,
- ``pair_agreement``: fraction of judges preferring chosen over rejected
  (ties count 0.5) - directly usable as a soft label,
- ``judge_values``: the raw per-judge utility vectors.

Practitioners filter or weight on those columns and run their existing
training setup unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .errors import MissingDependencyError
from .types import DatapointResult


def _consensus_pair(result: DatapointResult) -> tuple[int, int]:
    """Chosen/rejected via panel consensus: highest/lowest mean utility."""
    k = result.datapoint.k
    means = [
        sum(a.values[i] for a in result.annotations) / len(result.annotations)
        for i in range(k)
    ]
    chosen = max(range(k), key=lambda i: means[i])
    rejected = min(range(k), key=lambda i: means[i])
    return chosen, rejected


def _meta_pair(result: DatapointResult) -> tuple[int, int]:
    """Chosen/rejected indices supplied by the source dataset in meta."""
    meta = result.datapoint.meta
    try:
        return int(meta["chosen_index"]), int(meta["rejected_index"])
    except KeyError:
        raise KeyError(
            "strategy='meta' needs 'chosen_index' and 'rejected_index' in datapoint.meta"
        ) from None


def to_records(
    results: Sequence[DatapointResult],
    strategy: str = "consensus",
    skip_unannotated: bool = True,
) -> list[dict[str, Any]]:
    """Flatten results into TRL-schema records with IJA metadata.

    ``strategy`` picks how the chosen/rejected pair is selected:
    "consensus" uses the panel's mean utilities; "meta" trusts
    ``chosen_index`` / ``rejected_index`` provided on each datapoint (i.e.
    the source dataset's original labels). Datapoints where every judge
    failed are skipped unless ``skip_unannotated=False``, in which case
    they raise.
    """
    if strategy not in ("consensus", "meta"):
        raise ValueError(f"Unknown strategy {strategy!r}; choose 'consensus' or 'meta'")
    records: list[dict[str, Any]] = []
    for result in results:
        if not result.annotations:
            if skip_unannotated:
                continue
            raise ValueError(
                f"Datapoint {result.datapoint.id!r} has no annotations; "
                "cannot export (all judges failed)"
            )
        chosen_i, rejected_i = (
            _meta_pair(result) if strategy == "meta" else _consensus_pair(result)
        )
        dp = result.datapoint
        records.append(
            {
                "id": dp.id,
                "prompt": dp.prompt,
                "chosen": dp.responses[chosen_i],
                "rejected": dp.responses[rejected_i],
                "chosen_index": chosen_i,
                "rejected_index": rejected_i,
                "prompt_ija": result.ija,
                "ija_metric": result.metric,
                "pair_agreement": result.pair_agreement(chosen_i, rejected_i),
                "judge_values": {a.judge: list(a.values) for a in result.annotations},
                "judge_models": {a.judge: a.model for a in result.annotations},
                "n_judges": len(result.annotations),
                "n_judge_failures": len(result.failures),
            }
        )
    return records


def to_jsonl(records: Sequence[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def to_hf_dataset(records: Sequence[dict[str, Any]]):
    """Build a Hugging Face ``datasets.Dataset`` that drops into TRL's
    ``DPOTrainer`` (requires the 'data' extra)."""
    try:
        from datasets import Dataset
    except ImportError:
        raise MissingDependencyError("datasets", "data") from None
    return Dataset.from_list(list(records))
