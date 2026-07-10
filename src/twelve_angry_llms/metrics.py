"""Agreement metrics.

All functions operate on parallel vectors of per-response utilities
(higher = better). K is small in practice (2-10), so everything is plain
O(K^2) Python with no numeric dependencies.

``kendall_tau_b`` is the primary metric: with 1-5 scores ties are common,
and the tau-b denominator corrects for them; on tie-free rankings it
reduces to plain Kendall's tau. Pair correlations are undefined when either
vector is constant (a judge that gave every response the same score); such
functions return None and callers are expected to skip those pairs.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

Metric = str  # "kendall" | "spearman" | "winner"


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Tie-corrected Kendall rank correlation between two utility vectors.

    Returns None when undefined (fewer than 2 items, or either vector
    constant so that no pair discriminates).
    """
    n = len(x)
    if n != len(y):
        raise ValueError("Vectors must have the same length")
    if n < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i, j in combinations(range(n), 2):
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        if dx == 0 and dy == 0:
            continue  # tied in both: contributes to neither denominator term
        if dx == 0:
            ties_x += 1
        elif dy == 0:
            ties_y += 1
        elif (dx > 0) == (dy > 0):
            concordant += 1
        else:
            discordant += 1
    denom_x = concordant + discordant + ties_x
    denom_y = concordant + discordant + ties_y
    if denom_x == 0 or denom_y == 0:
        return None
    return (concordant - discordant) / math.sqrt(denom_x * denom_y)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks 1..n with tied values receiving the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman correlation with average ranks for ties.

    Computed as the Pearson correlation of the rank vectors, which is the
    correct generalization in the presence of ties. Returns None when
    undefined (constant vector or n < 2).
    """
    n = len(x)
    if n != len(y):
        raise ValueError("Vectors must have the same length")
    if n < 2:
        return None
    rx, ry = _average_ranks(x), _average_ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var_x = sum((a - mx) ** 2 for a in rx)
    var_y = sum((b - my) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def pairwise_winner_agreement(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Fraction of response pairs the two judges order the same way.

    A pair is counted as agreed when both judges express the same strict
    preference or both tie it. Unlike the correlations this lives in [0, 1]
    and is defined even for constant vectors.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("Vectors must have the same length")
    if n < 2:
        return None
    agreed = total = 0
    for i, j in combinations(range(n), 2):
        sx = (x[i] > x[j]) - (x[i] < x[j])
        sy = (y[i] > y[j]) - (y[i] < y[j])
        agreed += sx == sy
        total += 1
    return agreed / total


_METRICS = {
    "kendall": kendall_tau_b,
    "spearman": spearman_rho,
    "winner": pairwise_winner_agreement,
}


def pair_metric(name: Metric):
    try:
        return _METRICS[name]
    except KeyError:
        raise ValueError(f"Unknown metric {name!r}; choose from {sorted(_METRICS)}") from None


def ija(
    value_vectors: Sequence[Sequence[float]],
    metric: Metric = "kendall",
) -> tuple[float | None, list[tuple[int, int, float]]]:
    """Inter-judge agreement: mean pairwise correlation across judges.

    ``value_vectors`` is one utility vector per judge, all over the same K
    responses. Returns ``(ija, pairs)`` where ``pairs`` lists
    ``(judge_i, judge_j, value)`` for every judge pair with a defined
    correlation. ``ija`` is None when no pair is defined.
    """
    fn = pair_metric(metric)
    pairs: list[tuple[int, int, float]] = []
    for i, j in combinations(range(len(value_vectors)), 2):
        value = fn(value_vectors[i], value_vectors[j])
        if value is not None:
            pairs.append((i, j, value))
    if not pairs:
        return None, pairs
    return sum(v for _, _, v in pairs) / len(pairs), pairs


def krippendorff_alpha(
    values: Sequence[Sequence[float | None]],
) -> float | None:
    """Krippendorff's alpha with the interval difference function.

    ``values[judge][item]`` may be None for missing annotations. This is a
    dataset-level diagnostic of panel health, not a per-datapoint signal:
    the chance-correction term needs marginals estimated across many items.
    Returns None when undefined (fewer than two pairable values anywhere,
    or zero expected disagreement).
    """
    if not values:
        return None
    n_items = len(values[0])
    if any(len(row) != n_items for row in values):
        raise ValueError("All judges must cover the same items")

    columns: list[list[float]] = []
    for item in range(n_items):
        col = [row[item] for row in values if row[item] is not None]
        if len(col) >= 2:
            columns.append(col)  # type: ignore[arg-type]
    n_pairable = sum(len(col) for col in columns)
    if n_pairable < 2:
        return None

    observed = 0.0
    for col in columns:
        m = len(col)
        within = sum((a - b) ** 2 for a, b in combinations(col, 2))
        observed += 2 * within / (m - 1)
    observed /= n_pairable

    pooled = [v for col in columns for v in col]
    expected = (
        2
        * sum((a - b) ** 2 for a, b in combinations(pooled, 2))
        / (n_pairable * (n_pairable - 1))
    )
    if expected == 0:
        return None
    return 1 - observed / expected
