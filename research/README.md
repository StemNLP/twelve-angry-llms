# Research: Twelve Angry LLMs

Everything specific to the study lives here; none of it ships in the pip
package. The package (`src/twelve_angry_llms`) is the instrument, this
folder is the experiment.

- **[RESEARCH_PLAN.md](RESEARCH_PLAN.md)** — the plan: core idea, hypotheses,
  judging protocol, related work, and the four experimental phases.
- **configs/** — panel definitions (which judges, which providers). API keys
  are referenced by environment-variable name only, never stored.
- **experiments/** — one script per experimental step.
  - `corruptions.py` — the four Phase 1 corruption types.
  - `pilot.py` — the 200-prompt shakeout run that precedes Phase 1 (plan §9).
- **results/** — run outputs (gitignored except for summaries we choose to keep).

## Running the pilot

```bash
uv sync --extra data
export OPENROUTER_API_KEY=...
uv run python research/experiments/pilot.py --panel research/configs/panel.example.yaml
```

Every model response is cached in `.tal/cache.sqlite` (see the `cache` key
in the panel yaml), so re-runs are free.
