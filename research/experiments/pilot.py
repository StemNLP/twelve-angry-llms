"""The 200-prompt pilot (RESEARCH_PLAN.md §9, item 5).

A throwaway shakeout before Phase 1 proper: 200 raw UltraFeedback prompts,
half corrupted, run through a small panel. Its only job is to confirm that

1. judge outputs parse (failure rate reported),
2. IJA actually spreads across datapoints instead of clumping,
3. a blatant flip visibly tanks the corrupted pair's agreement.

Usage:
    export OPENAI_API_KEY=... OPENROUTER_API_KEY=... ANTHROPIC_API_KEY=...
    uv run python research/experiments/pilot.py --panel research/configs/panel.example.yaml
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from corruptions import corrupt_pool

from twelve_angry_llms import ScoringProtocol, to_jsonl, to_records
from twelve_angry_llms.cli import _load_panel
from twelve_angry_llms.loaders import load_ultrafeedback


def reference_order(meta: dict) -> list[int] | None:
    """Reference ordering from UltraFeedback's original single-judge scores.

    Good enough for the pilot; Phase 1 proper replaces this with
    strong-judge consensus plus manual spot-checks (plan §6).
    """
    scores = meta.get("original_scores")
    if not scores or any(s is None for s in scores):
        return None
    if len(set(scores)) < len(scores):
        return None  # ties make the reference ordering ambiguous; skip
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, help="panel yaml (see research/configs)")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default="research/results/pilot")
    args = parser.parse_args()

    print(f"Loading {args.n} raw UltraFeedback prompts...", file=sys.stderr)
    raw = load_ultrafeedback(limit=args.n * 2, seed=7)  # extra to survive filtering
    pool = []
    for dp in raw:
        order = reference_order(dict(dp.meta))
        if order is not None:
            pool.append((dp, order))
        if len(pool) == args.n:
            break
    corrupted = corrupt_pool(pool, corrupt_fraction=0.5, seed=7)

    panel, _ = _load_panel(Path(args.panel))
    print(f"Annotating {len(corrupted)} datapoints "
          f"with {len(panel.judges)} judges...", file=sys.stderr)
    results = panel.annotate_sync(
        [c.datapoint for c in corrupted], ScoringProtocol(), concurrency=args.concurrency
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = [(r, c) for r, c in zip(results, corrupted) if r.annotations]
    records = to_records([r for r, _ in kept])
    for record, (result, item) in zip(records, kept):
        record["corruption"] = item.corruption
        record["label_pair_agreement"] = result.pair_agreement(
            item.chosen_index, item.rejected_index
        )
    to_jsonl(records, out_dir / "pilot.jsonl")

    # --- The three shakeout checks ---
    diag = panel.diagnostics(results)
    print(f"\n1. Parse health: {diag.n_failures} judge failures "
          f"over {diag.n_annotations + diag.n_failures} calls")

    ija_values = [r.ija for r in results if r.ija is not None]
    print(f"2. IJA spread: n={len(ija_values)}, "
          f"mean={statistics.mean(ija_values):.3f}, "
          f"stdev={statistics.stdev(ija_values):.3f}, "
          f"min={min(ija_values):.3f}, max={max(ija_values):.3f}")

    by_kind: dict[str | None, list[float]] = {}
    for result, item in kept:
        agreement = result.pair_agreement(item.chosen_index, item.rejected_index)
        if agreement is not None:
            by_kind.setdefault(item.corruption, []).append(agreement)
    print("3. Label pair-agreement by corruption type "
          "(flip_extremes should be far below clean):")
    for kind in [None, "swap_adjacent", "flip_extremes", "replace_offtopic", "pad_weak"]:
        values = by_kind.get(kind, [])
        label = kind or "clean"
        if values:
            print(f"   {label:>16}: mean={statistics.mean(values):.3f} (n={len(values)})")

    alpha = diag.krippendorff_alpha
    alpha_text = f"{alpha:.3f}" if alpha is not None else "n/a"
    print(f"\nPanel health: Krippendorff's alpha = {alpha_text}")
    for pair_key, value in diag.judge_correlation.items():
        print(f"  {pair_key}: {value:.3f}")


if __name__ == "__main__":
    main()
