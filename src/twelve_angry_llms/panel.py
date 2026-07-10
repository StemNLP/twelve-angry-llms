"""The panel: fan a set of judges out over datapoints and compute IJA."""

from __future__ import annotations

import asyncio
from typing import Callable, Iterable, Sequence

from . import metrics
from .judge import Judge
from .protocols import Protocol
from .types import (
    DatapointResult,
    JudgeAnnotation,
    JudgeFailure,
    PanelDiagnostics,
    PreferenceDatapoint,
)


class Panel:
    """A set of judges that annotate datapoints and measure their agreement.

    ``metric`` selects the pairwise agreement measure used for IJA:
    "kendall" (tau-b, the default), "spearman", or "winner"
    (pairwise-winner agreement). Judge names must be unique.
    """

    def __init__(self, judges: Sequence[Judge], metric: metrics.Metric = "kendall") -> None:
        if len(judges) < 2:
            raise ValueError("A panel needs at least 2 judges to measure agreement")
        names = [j.name for j in judges]
        if len(set(names)) != len(names):
            raise ValueError(f"Judge names must be unique, got {names}")
        metrics.pair_metric(metric)  # validate early
        self.judges = list(judges)
        self.metric = metric

    async def annotate_one(
        self,
        datapoint: PreferenceDatapoint,
        protocol: Protocol,
        semaphore: asyncio.Semaphore | None = None,
    ) -> DatapointResult:
        async def run(judge: Judge) -> JudgeAnnotation:
            if semaphore is None:
                return await judge.annotate(datapoint, protocol)
            async with semaphore:
                return await judge.annotate(datapoint, protocol)

        outcomes = await asyncio.gather(
            *(run(j) for j in self.judges), return_exceptions=True
        )
        annotations: list[JudgeAnnotation] = []
        failures: list[JudgeFailure] = []
        for judge, outcome in zip(self.judges, outcomes):
            if isinstance(outcome, BaseException):
                failures.append(
                    JudgeFailure(judge=judge.name, model=judge.model, error=str(outcome))
                )
            else:
                annotations.append(outcome)

        ija_value, pairs = metrics.ija(
            [a.values for a in annotations], metric=self.metric
        )
        pairwise = {
            f"{annotations[i].judge}|{annotations[j].judge}": value
            for i, j, value in pairs
        }
        return DatapointResult(
            datapoint=datapoint,
            annotations=annotations,
            failures=failures,
            ija=ija_value,
            ija_pairwise=pairwise,
            metric=self.metric,
        )

    async def annotate(
        self,
        datapoints: Iterable[PreferenceDatapoint],
        protocol: Protocol,
        concurrency: int = 8,
        on_result: Callable[[DatapointResult], None] | None = None,
    ) -> list[DatapointResult]:
        """Annotate many datapoints, at most ``concurrency`` requests in flight.

        Results are returned in input order. ``on_result`` (if given) is
        called as each datapoint completes - useful for progress reporting
        or incremental writes.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def run(dp: PreferenceDatapoint) -> DatapointResult:
            result = await self.annotate_one(dp, protocol, semaphore)
            if on_result is not None:
                on_result(result)
            return result

        return list(await asyncio.gather(*(run(dp) for dp in datapoints)))

    def annotate_sync(
        self,
        datapoints: Iterable[PreferenceDatapoint],
        protocol: Protocol,
        concurrency: int = 8,
        on_result: Callable[[DatapointResult], None] | None = None,
    ) -> list[DatapointResult]:
        """Blocking wrapper around :meth:`annotate` for scripts and notebooks."""
        return asyncio.run(
            self.annotate(datapoints, protocol, concurrency=concurrency, on_result=on_result)
        )

    def diagnostics(self, results: Sequence[DatapointResult]) -> PanelDiagnostics:
        """Dataset-level panel health: Krippendorff's alpha and the judge-judge
        correlation matrix (mean pairwise metric across datapoints)."""
        judge_names = [j.name for j in self.judges]

        # judges x (datapoints * responses) value matrix for alpha, with
        # missing entries where a judge failed.
        rows: dict[str, list[float | None]] = {name: [] for name in judge_names}
        for result in results:
            by_judge = {a.judge: a for a in result.annotations}
            k = result.datapoint.k
            for name in judge_names:
                ann = by_judge.get(name)
                rows[name].extend(ann.values if ann else [None] * k)
        alpha = metrics.krippendorff_alpha([rows[name] for name in judge_names])

        pair_values: dict[str, list[float]] = {}
        for result in results:
            for pair_key, value in result.ija_pairwise.items():
                pair_values.setdefault(pair_key, []).append(value)
        correlation = {
            pair_key: sum(vals) / len(vals) for pair_key, vals in sorted(pair_values.items())
        }

        return PanelDiagnostics(
            krippendorff_alpha=alpha,
            judge_correlation=correlation,
            n_datapoints=len(results),
            n_annotations=sum(len(r.annotations) for r in results),
            n_failures=sum(len(r.failures) for r in results),
        )
