# Twelve Angry LLMs

**Panel-based reliability annotation for preference data.**

Modern preference datasets are built by asking a *single* strong judge to
score or rank candidate responses, then collapsing its verdict into
`(chosen, rejected)` pairs for DPO or reward-model training. That collapse
throws away any notion of how *contestable* each label was.

`twelve-angry-llms` recovers that signal: it runs a **panel of diverse LLM
judges** over each `(prompt, K responses)` datapoint and measures how much
the judges agree with each other — the **inter-judge agreement (IJA)**,
computed as average pairwise Kendall's τ-b over the judges' rankings. High
IJA means the label is trustworthy; low IJA means it is shaky and likely to
inject noise into training. The result exports straight into TRL's
`(prompt, chosen, rejected)` schema with the reliability signals attached,
so you can filter, down-weight, or soft-label your data and run your
existing training setup unchanged.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Installation

```bash
pip install twelve-angry-llms                # core (OpenAI-compatible providers)
pip install "twelve-angry-llms[anthropic]"   # + native Anthropic client
pip install "twelve-angry-llms[data]"        # + raw UltraFeedback / Nectar loaders
```

## Quickstart

Judges can live anywhere: any OpenAI-compatible endpoint (OpenAI,
OpenRouter, Together, Groq, a local vLLM or Ollama server, ...) or the
Anthropic API. Keys are read from environment variables — pass
`api_key_env` to point at a different variable per provider.

```python
from twelve_angry_llms import (
    Judge, Panel, PreferenceDatapoint, ScoringProtocol,
    OpenAICompatibleClient, to_records, to_jsonl,
)
from twelve_angry_llms.clients import AnthropicClient

openai = OpenAICompatibleClient()                       # OPENAI_API_KEY
openrouter = OpenAICompatibleClient(
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY",
)
panel = Panel([
    Judge(model="gpt-4o-2024-08-06", client=openai),
    Judge(model="claude-sonnet-5", client=AnthropicClient()),
    Judge(model="qwen/qwen-2.5-72b-instruct", client=openrouter),
])

datapoints = [
    PreferenceDatapoint(
        prompt="What causes tides?",
        responses=(
            "The gravitational pull of the moon and sun.",
            "Mostly wind patterns over the ocean.",
            "The moon's gravity, with a smaller solar contribution.",
        ),
    ),
]

results = panel.annotate_sync(datapoints, ScoringProtocol())
print(results[0].ija)              # panel agreement on this datapoint, in [-1, 1]
print(panel.diagnostics(results))  # Krippendorff's alpha, judge-judge correlations

to_jsonl(to_records(results), "annotated.jsonl")  # TRL schema + IJA columns
```

Each exported record carries:

| column | meaning |
|---|---|
| `prompt`, `chosen`, `rejected` | the standard TRL/DPO interface |
| `prompt_ija` | panel agreement over all K candidates (Kendall's τ-b) |
| `pair_agreement` | fraction of judges preferring chosen over rejected — usable directly as a soft label for conservative DPO / soft-label methods |
| `judge_values` | raw per-judge scores or rankings |

## Two elicitation protocols

- **`ScoringProtocol`** — every judge scores each response 1–5 against a
  shared rubric (the UltraFeedback style). Also yields per-judge score
  margins for free.
- **`RankingProtocol`** — every judge orders all K responses best-to-worst
  in one pass (the Nectar style).

Both parse into per-response utilities, so IJA and all exports work
identically. Judges receiving one shared guideline per protocol is the
point: any disagreement then reflects the judges, not prompt wording.

## Command line

```bash
tal annotate --input data.jsonl --panel panel.yaml --output annotated.jsonl
tal export   --input annotated.jsonl --output pairs.jsonl
```

`data.jsonl` rows need `prompt` and `responses`; `panel.yaml` declares the
judges and (optionally) a response cache:

```yaml
metric: kendall
cache: .tal/cache.sqlite
judges:
  - model: gpt-4o-2024-08-06
    provider: openai
  - model: claude-sonnet-5
    provider: anthropic
  - model: qwen/qwen-2.5-72b-instruct
    provider: openai
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
```

With the `data` extra you can annotate the raw (non-binarized) datasets
directly: `tal annotate --dataset ultrafeedback --limit 1000 ...`.

## Reproducibility

- Temperature defaults to 0 everywhere.
- `CachedClient` / the `cache:` key stores every raw response in SQLite,
  keyed on (model, sampling settings, messages) — re-runs never re-bill.
- Clients track token usage (`client.usage`) so runs can report cost.

## The research

This library is the instrument for an ongoing study of IJA as a
label-reliability signal — hypotheses, judging protocol, and experimental
phases live in [research/RESEARCH_PLAN.md](research/RESEARCH_PLAN.md).
Working title: *"Twelve Angry LLMs: Judge Agreement as a Label-Reliability
Signal for Preference Data."*

## License

MIT
