# Twelve Angry LLMs (work in progress)

*LLM-as-a-judge* is an emerging AI evaluation method where a large language model (LLM) assesses the quality and acts as an evaluator. LLM-as-a-judge has remarkable potential because it can replace the complex and effort-heavy methods, especially when it comes to open-ended evaluation. *Twelve Angry LLMs* library builds an LLM jury with multiple LLM judges to increases the stability and reliability of the final verdict. It's an experimental project to explore how agreement can be measured and a final verdict can be cast, from the decision of several LLMs.

 *Twelve Angry LLMs* supports the following tasks:

- Open-ended text generation
- Text/Image Classification
- Text/Image Ranking

Modular design:
- Tasks: Generation, Classification, Ranking
- Judge: prompts an LLM and normalizes outputs
- Jury: orchestrates judges and computes agreement
- LLMClient: plug any provider (OpenAI, HF, local), via a simple generate(prompt, ...) interface

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Installation

```bash
pip install twelve-angry-llms
```

## Quickstart

Below is a minimal, self-contained example using a tiny mock client. Replace the mock with your provider by implementing LLMClient.

```python
from twelve_angry_llms.tasks import GenerationTask, ClassificationTask, RankingTask
from twelve_angry_llms.judge import Judge
from twelve_angry_llms.jury import Jury
from twelve_angry_llms.clients.base import LLMClient

# Mock clients returning deterministic outputs for demonstration
class FixedClient:
    def __init__(self, text): self.text = text
    def generate(self, prompt: str, **kwargs) -> str: return self.text

# 1) Generation agreement (token Jaccard over pairwise outputs)
gen_task = GenerationTask(input_text="Summarize: LLMs are used for many NLP tasks.")
judges_gen = [
    Judge("gpt-A", FixedClient("LLMs are widely used in NLP tasks.")),
    Judge("gpt-B", FixedClient("Large language models power many NLP applications.")),
    Judge("gpt-C", FixedClient("LLMs are used for various NLP tasks.")),
]
jury = Jury(judges_gen)
gen_result = jury.evaluate(gen_task)
print("Generation agreement:", gen_result.agreement)

# 2) Classification agreement (single-label exact match)
cls_task = ClassificationTask(
    input_text="The service was quick and friendly.",
    labels=["positive", "neutral", "negative"],
    multi_label=False,
)
judges_cls = [
    Judge("cls-A", FixedClient("positive")),
    Judge("cls-B", FixedClient("positive")),
    Judge("cls-C", FixedClient("neutral")),
]
print("Classification agreement:", Jury(judges_cls).evaluate(cls_task).agreement)

# 3) Ranking agreement (pairwise Spearman rho)
rank_task = RankingTask(
    items=["Alpha", "Beta", "Gamma"],
    criteria="usefulness",
)
judges_rank = [
    Judge("rank-A", FixedClient("Alpha\nBeta\nGamma")),
    Judge("rank-B", FixedClient("Beta\nAlpha\nGamma")),
    Judge("rank-C", FixedClient("Alpha\nGamma\nBeta")),
]
print("Ranking agreement:", Jury(judges_rank).evaluate(rank_task).agreement)
```

## API Overview

- Tasks (twelve_angry_llms.tasks)
  - GenerationTask(input_text: str, guidance: Optional[str] = None)
  - ClassificationTask(input_text: str, labels: List[str], multi_label: bool = False)
  - RankingTask(items: List[str], criteria: Optional[str] = None)

- Judge (twelve_angry_llms.judge)
  - Judge(name: str, client: LLMClient, temperature: float = 0.0)
  - predict(task) -> JudgeOutput

- Jury (twelve_angry_llms.jury)
  - Jury(judges: List[Judge])
  - evaluate(task) -> JuryResult
    - JuryResult.agreement: float in [0, 1] (higher is better)
    - JuryResult.outputs: per-judge raw/normalized outputs
    - JuryResult.details: pairwise scores and extra info

- LLMClient protocol (twelve_angry_llms.clients.base)
  - generate(prompt: str, system: Optional[str] = None, **kwargs) -> str

## Agreement Metrics (default)
- Generation: average pairwise Jaccard similarity over token sets
- Classification (single): average pairwise exact match
- Classification (multi): average pairwise Jaccard over label sets
- Ranking: average pairwise Spearman rank correlation (no ties)

These are simple, dependency-free defaults. You can later swap in stronger metrics (e.g., Krippendorff’s alpha, Kendall’s tau-b, embedding similarity).

## Using Your Own Provider

Implement LLMClient and pass it to Judge.

```python
from twelve_angry_llms.clients.base import LLMClient

class MyProvider(LLMClient):
    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        # call your model here and return the text
        return "model output"

# Judge("my-judge", MyProvider())
```

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## License

MIT
