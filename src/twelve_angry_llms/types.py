"""Core data types.

The package works on *preference datapoints*: a prompt together with K
candidate responses. A panel of judges annotates each datapoint, and every
annotation is normalized to a vector of per-response **utilities** where
higher means better, regardless of whether the judge scored (1-5) or ranked
(K-wise). All downstream metrics consume that single convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PreferenceDatapoint:
    """A prompt with K candidate responses (K >= 2).

    ``meta`` carries anything the caller wants to keep alongside the
    datapoint (source dataset, original ranks, chosen/rejected indices, ...).
    """

    prompt: str
    responses: tuple[str, ...]
    id: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.responses) < 2:
            raise ValueError("A preference datapoint needs at least 2 responses")

    @property
    def k(self) -> int:
        return len(self.responses)


@dataclass(frozen=True)
class JudgeAnnotation:
    """One judge's verdict on one datapoint.

    ``values`` holds one utility per response, higher is better. For the
    scoring protocol these are the raw 1-5 scores; for the ranking protocol
    a response ranked r-th out of K gets utility K + 1 - r.
    """

    judge: str
    model: str
    protocol: str
    values: tuple[float, ...]
    raw: str


@dataclass(frozen=True)
class JudgeFailure:
    """A judge that errored or could not be parsed for a datapoint."""

    judge: str
    model: str
    error: str


@dataclass
class DatapointResult:
    """Panel output for one datapoint.

    ``ija`` is the average pairwise rank correlation between judges
    (None when fewer than two judges produced usable annotations, or when
    no judge pair had a defined correlation). ``ija_pairwise`` maps
    "judgeA|judgeB" to that pair's correlation; undefined pairs (e.g. a
    judge that gave every response the same score) are omitted.
    """

    datapoint: PreferenceDatapoint
    annotations: list[JudgeAnnotation]
    failures: list[JudgeFailure] = field(default_factory=list)
    ija: float | None = None
    ija_pairwise: dict[str, float] = field(default_factory=dict)
    metric: str = "kendall"

    def pair_agreement(self, preferred_index: int, other_index: int) -> float | None:
        """Fraction of judges preferring one response over another.

        A judge that ties the two responses contributes 0.5. Returns None
        when there are no annotations.
        """
        if not self.annotations:
            return None
        total = 0.0
        for ann in self.annotations:
            a, b = ann.values[preferred_index], ann.values[other_index]
            total += 1.0 if a > b else 0.5 if a == b else 0.0
        return total / len(self.annotations)


@dataclass(frozen=True)
class PanelDiagnostics:
    """Dataset-level health indicators for a panel run."""

    krippendorff_alpha: float | None
    judge_correlation: dict[str, float]
    n_datapoints: int
    n_annotations: int
    n_failures: int
