"""``tal`` - the twelve-angry-llms command line.

Typical flow:

    tal annotate --input data.jsonl --panel panel.yaml --output annotated.jsonl
    tal export --input annotated.jsonl --output pairs.jsonl

``data.jsonl`` rows need "prompt" and "responses" (a list of K strings);
"id" and "meta" are optional. ``--dataset ultrafeedback|nectar`` can be
used instead of ``--input`` (requires the 'data' extra). ``panel.yaml``
defines the judges:

    metric: kendall            # optional: kendall | spearman | winner
    cache: .tal/cache.sqlite   # optional: response cache path
    judges:
      - model: openai/gpt-4o-2024-08-06
        provider: openrouter              # the default
      - model: qwen/qwen-2.5-72b-instruct
        provider: openrouter
        extra_body:                       # optional: merged into requests,
          provider:                       # e.g. OpenRouter provider pinning
            order: [together]
            allow_fallbacks: false
      - model: claude-sonnet-5            # direct provider APIs also work
        provider: anthropic
      - model: llama3.1:70b               # any OpenAI-compatible endpoint
        provider: openai
        base_url: http://localhost:11434/v1

API keys are never written in the file - each judge names the environment
variable holding its key (``api_key_env``), defaulting to OPENAI_API_KEY
or ANTHROPIC_API_KEY by provider.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cache import CachedClient, ResponseCache
from .clients.openai_compat import OpenAICompatibleClient, OpenRouterClient
from .export import to_jsonl, to_records
from .judge import Judge
from .panel import Panel
from .protocols import RankingProtocol, ScoringProtocol
from .types import DatapointResult, JudgeAnnotation, JudgeFailure, PreferenceDatapoint


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_panel(path: Path) -> tuple[Panel, ResponseCache | None]:
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    cache = ResponseCache(config["cache"]) if config.get("cache") else None
    judges = []
    for spec in config["judges"]:
        provider = spec.get("provider", "openrouter")
        if provider == "anthropic":
            from .clients.anthropic_client import AnthropicClient

            client = AnthropicClient(
                api_key_env=spec.get("api_key_env", "ANTHROPIC_API_KEY")
            )
        elif provider == "openrouter":
            client = OpenRouterClient(
                api_key_env=spec.get("api_key_env", "OPENROUTER_API_KEY"),
                extra_body=spec.get("extra_body"),
            )
        elif provider in ("openai", "openai-compatible"):
            client = OpenAICompatibleClient(
                base_url=spec.get("base_url"),
                api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
                extra_body=spec.get("extra_body"),
            )
        else:
            raise SystemExit(
                f"Unknown provider {provider!r} "
                "(use 'openrouter', 'openai', 'openai-compatible', or 'anthropic')"
            )
        if cache is not None:
            client = CachedClient(client, cache)
        judges.append(
            Judge(
                model=spec["model"],
                client=client,
                name=spec.get("name", ""),
                temperature=float(spec.get("temperature", 0.0)),
                max_tokens=int(spec.get("max_tokens", 1024)),
            )
        )
    return Panel(judges, metric=config.get("metric", "kendall")), cache


def _load_datapoints(args) -> list[PreferenceDatapoint]:
    if args.input:
        rows = _read_jsonl(Path(args.input))
        if args.limit:
            rows = rows[: args.limit]
        return [
            PreferenceDatapoint(
                prompt=row["prompt"],
                responses=tuple(row["responses"]),
                id=row.get("id", f"row-{i}"),
                meta=row.get("meta", {}),
            )
            for i, row in enumerate(rows)
        ]
    from . import loaders

    loader = {"ultrafeedback": loaders.load_ultrafeedback, "nectar": loaders.load_nectar}[
        args.dataset
    ]
    return loader(limit=args.limit)


def _result_to_row(result: DatapointResult, protocol_name: str) -> dict:
    dp = result.datapoint
    return {
        "id": dp.id,
        "prompt": dp.prompt,
        "responses": list(dp.responses),
        "meta": dict(dp.meta),
        "protocol": protocol_name,
        "metric": result.metric,
        "ija": result.ija,
        "ija_pairwise": result.ija_pairwise,
        "judge_values": {a.judge: list(a.values) for a in result.annotations},
        "judge_models": {a.judge: a.model for a in result.annotations},
        "failures": [{"judge": f.judge, "model": f.model, "error": f.error} for f in result.failures],
    }


def _row_to_result(row: dict) -> DatapointResult:
    dp = PreferenceDatapoint(
        prompt=row["prompt"],
        responses=tuple(row["responses"]),
        id=row.get("id"),
        meta=row.get("meta", {}),
    )
    annotations = [
        JudgeAnnotation(
            judge=name,
            model=row.get("judge_models", {}).get(name, ""),
            protocol=row.get("protocol", ""),
            values=tuple(values),
            raw="",
        )
        for name, values in row.get("judge_values", {}).items()
    ]
    failures = [
        JudgeFailure(judge=f["judge"], model=f.get("model", ""), error=f.get("error", ""))
        for f in row.get("failures", [])
    ]
    return DatapointResult(
        datapoint=dp,
        annotations=annotations,
        failures=failures,
        ija=row.get("ija"),
        ija_pairwise=row.get("ija_pairwise", {}),
        metric=row.get("metric", "kendall"),
    )


def _cmd_annotate(args) -> None:
    panel, _ = _load_panel(Path(args.panel))
    protocol = ScoringProtocol() if args.protocol == "scoring" else RankingProtocol()
    datapoints = _load_datapoints(args)
    if not datapoints:
        raise SystemExit("No datapoints to annotate")

    done = 0

    def progress(result: DatapointResult) -> None:
        nonlocal done
        done += 1
        if done % 10 == 0 or done == len(datapoints):
            print(f"  annotated {done}/{len(datapoints)}", file=sys.stderr)

    results = panel.annotate_sync(
        datapoints, protocol, concurrency=args.concurrency, on_result=progress
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(_result_to_row(result, protocol.name), ensure_ascii=False) + "\n")

    diag = panel.diagnostics(results)
    alpha = f"{diag.krippendorff_alpha:.3f}" if diag.krippendorff_alpha is not None else "n/a"
    print(
        f"Wrote {len(results)} datapoints to {out}\n"
        f"  judge failures: {diag.n_failures}; Krippendorff's alpha: {alpha}",
        file=sys.stderr,
    )
    for pair_key, value in diag.judge_correlation.items():
        print(f"  {pair_key}: {value:.3f}", file=sys.stderr)


def _cmd_export(args) -> None:
    rows = _read_jsonl(Path(args.input))
    results = [_row_to_result(row) for row in rows]
    records = to_records(results, strategy=args.strategy)
    to_jsonl(records, args.output)
    print(f"Wrote {len(records)} (prompt, chosen, rejected) records to {args.output}",
          file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tal", description="Panel-based reliability annotation for preference data"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    annotate = sub.add_parser("annotate", help="Run a judge panel over datapoints")
    source = annotate.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="JSONL with prompt/responses rows")
    source.add_argument(
        "--dataset", choices=["ultrafeedback", "nectar"], help="Load a raw dataset instead"
    )
    annotate.add_argument("--panel", required=True, help="panel.yaml defining the judges")
    annotate.add_argument("--output", required=True, help="Annotated JSONL to write")
    annotate.add_argument(
        "--protocol", choices=["scoring", "ranking"], default="scoring",
        help="Elicitation protocol (default: scoring)",
    )
    annotate.add_argument("--concurrency", type=int, default=8)
    annotate.add_argument("--limit", type=int, default=None, help="Annotate at most N datapoints")
    annotate.set_defaults(func=_cmd_annotate)

    export = sub.add_parser(
        "export", help="Convert annotated JSONL to (prompt, chosen, rejected) records"
    )
    export.add_argument("--input", required=True, help="Annotated JSONL from `tal annotate`")
    export.add_argument("--output", required=True, help="TRL-schema JSONL to write")
    export.add_argument(
        "--strategy", choices=["consensus", "meta"], default="consensus",
        help="How chosen/rejected are picked (default: panel consensus)",
    )
    export.set_defaults(func=_cmd_export)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
