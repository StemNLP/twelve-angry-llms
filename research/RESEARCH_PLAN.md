# Research Plan: Inter-Judge Agreement as a Reliability Signal for Preference Data

*Working title: "Twelve Angry LLMs: Judge Agreement as a Label-Reliability Signal for Preference Data"*
*Versions*
v0.4 — 2026-08-14
v0.3 — 2026-07-09

## 1. The core idea

Modern preference datasets are built by showing a language model a prompt together with
several candidate responses and asking a single strong judge (most existing datasets have asked almost always GPT-4 or GPT-4o) to score or rank them. Those scores are then typically collapsed into a single `(chosen, rejected)` pair that gets used for DPO or reward-model training.

Our claim is that the collapsing step throws away a useful signal, and that we can recover it by replacing the single judge with a **panel** of diverse judges and measuring how much they agree with each other on each datapoint. That single-judge labels
are fragile is not hypothetical: UltraFeedback circulates in two binarizations
([HuggingFaceH4](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), and
[Argilla's cleaned re-binarization](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)
after corrupted overall scores were found) that disagree on which response is "chosen" for
many prompts.

This idea is borrowed from inter-annotator agreement (IAA), the standard way of assessing label reliability in human annotation; we carry it over to LLM judges and call it **inter-judge agreement (IJA)**. The intuition is: if five different models all rank four candidate responses in nearly the same order, the resulting preference label is trustworthy; if they disagree sharply, the label is shaky and is likely to inject noise into training. IJA is therefore a measure of *label reliability*, computed cheaply and automatically, before model training.

### What the target datapoint looks like

A preference datapoint is a prompt together with K candidate responses, where K is typically
4 or more. This is the shape data takes at the *judging stage* of the standard pipelines
(UltraFeedback, Nectar). Importantly, most of these datasets are published in a reduced binary form (i.e. just the chosen and rejected response) because the K candidates were scored and then
collapsed. To compute IJA we need the K candidates back, so we work from the **raw**,
non-binarized releases that still contain them. Raw UltraFeedback keeps all four completions
and is our primary source; Nectar keeps seven.

### How IJA is measured

Each of the **J** judges in the panel scores the K candidate responses, which induces a
ranking of those responses for that judge. For every pair of judges we compute the rank
correlation between their two rankings, and IJA for the datapoint is the average of that
correlation over all C(J,2) judge pairs.

We use **Kendall's τ-b** as the rank correlation rather than Spearman's ρ, for three reasons.
First, K is small (often just 4), and on so few items Spearman's ρ is jumpy and takes only a
handful of discrete values, whereas Kendall's τ counts concordant versus discordant response
pairs and behaves more stably. Second, under the scoring protocol judges emit 1–5 scores, so
ties among the K responses are common, and the "-b" variant of τ has a proper correction for
ties (under the ranking protocol judges produce total orders, where τ-b reduces to plain τ). Third, τ has a direct reading that matches what we care about: it is essentially the fraction of response-pairs that
two judges order the same way, which *is* preference agreement. We still report Spearman's ρ
in the ablations so that readers can see the result is not an artifact of the metric choice.

J (the number of judges) is itself a variable of the study; we sweep it in Phase 2 and use a
default panel of five judges elsewhere.

### Why this framing is justified

Agreement between judges is a direct measure of how *reliable* the training label is. We are not measuring whether the judges are *correct* — that would require an external ground truth about response quality, which we do not have, but we also do not need ath this stage and for the purpose of this research. If five independent judges order the responses the same way, the label is stable
and reproducible; if they largely disagree, the label is unreliable. 

This matters because label unreliability accounts for
many of the known failures of preference training: contradictory pairs pull the policy in
arbitrary directions, and DPO in particular is known to degrade under label noise.

## 2. Hypothesis

**H1 (core claim).** IJA measures preference-label reliability. Datapoints where the panel
disagrees produce noisy chosen/rejected pairs, and using IJA to filter, down-weight, or
soft-label those pairs improves DPO outcomes compared with (a) doing no filtering and (b) the
standard practice of margin-based filtering (keeping only pairs where a single judge assigned
a large score gap).

**A note on panel selection.** We draw the judges from different model families (OpenAI,
Anthropic, Qwen, Llama, Mistral) because judges sharing a base model tend to make correlated
errors and agree on their shared blind spots, which inflates agreement without adding
information. We treat this as a design principle rather than a hypothesis to test, and
monitor it through the judge–judge correlation matrix and Krippendorff's α that every run
reports.

## 3. How this integrates with real training pipelines

A central design goal is that IJA should slot into the way people actually build and consume
preference data, so that the downstream API we build is easy to adopt. The table below
summarizes the shapes that matter.

| Pipeline | Candidates per prompt | Label form | How it is trained on |
|---|---|---|---|
| InstructGPT | 4–9, human-ranked | full ranking | all C(K,2) pairs → Bradley-Terry RM |
| HH-RLHF / SHP / Chatbot Arena | 2 | chosen/rejected (Arena allows ties) | pairwise RM / DPO |
| Llama 2 | 2 | pair + 4-level margin | margin-aware Bradley-Terry loss |
| [UltraFeedback](https://arxiv.org/pdf/2310.01377) | 4 at judging; 2 as shipped | four 1–10 aspect scalars, then binarized | binarized → DPO (Zephyr recipe) |
| [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar) / Starling | 7, GPT-4 ranks K-wise | full ranking | K-wise reward loss |
| HelpSteer2 | 1 | scalar attribute ratings | regression RM |
| GRPO (DeepSeek) | K on-policy samples | reward per sample | group-normalized advantage |

Two things follow from this:

First, judging four or more candidates per prompt is already the
norm upstream, so IJA integrates well into how people generate data.

Second, almost everyone ultimately trains through the
`(prompt, chosen, rejected)` interface that the `DPOTrainer` of TRL (Hugging Face's post-training library) expects. Our exporter therefore
emits exactly that schema, with IJA carried alongside as extra metadata columns:
`prompt_ija` (the agreement over the K candidates), `pair_agreement` (the fraction
of judges that preferred the chosen response over the rejected one), and the raw per-judge
scores or rankings.

The `pair_agreement` column also serves as a **soft label**. Conservative DPO (cDPO) assumes
each pair is mislabeled with probability ε (label noise) and trains on:

> L = (1 − ε) · L_DPO(chosen ≻ rejected) + ε · L_DPO(rejected ≻ chosen),

which reduces to plain DPO at ε = 0 and cancels the pair entirely at ε = 0.5 — but ε is
normally a single global hyperparameter set for the whole dataset. 

The panel of judges, however, enables us to
estimate it per datapoint: if p = `pair_agreement` is the fraction of judges preferring the
chosen response, then ε̂ = 1 − p. A unanimous pair (p = 1) trains at full strength; a 3–2
split (ε̂ = 0.4) contributes close to zero. The same fraction can feed [geometric-averaged preference optimization](https://arxiv.org/pdf/2409.06691), which weights each pair's gradient by label confidence. 

Extending IJA to GRPO-style RL is a natural but separate line of work, we reserved as future direction (§8).

## 4. The judging protocol

Because the mentioned datasets used a single judge, we cannot calculate IJA off them. We have to run
the panel ourselves. We start with judging a thousand prompts with several models.

**One shared rubric across all judges.** Within a dataset, every judge receives the *same* guideline that includes:

- The definition of what a score of 1 through 5 means (scoring protocol) 

- The criteria by which responses are to be ordered (ranking protocol)
 
We do not give different judges different prompts. 

UltraFeedback releases its full annotation template 
(in `src/data_annotation/preference_templates.py`), so we adopt that as the canonical rubric for the scoring protocol;

Nectar publishes only an excerpt of its rubric and defers the position-bias handling to an
unreleased writeup, so it cannot serve as a reproducible template; for the ranking protocol we write a single explicit ranking instruction shared by all judges instead.

**Matching each dataset's native style.** Rather than forcing one protocol on
both datasets, the panel judges each dataset the way its original pipeline did:

**UltraFeedback** the judges score each response 1–5 against the shared rubric (per-response
scoring, the UltraFeedback style).
 
 **Nectar** the judges rank the seven responses
directly in one pass (K-wise ranking, the Nectar style). This keeps our panel labels
directly comparable to each dataset's published labels and means the downstream DPO
experiments consume data produced under the same elicitation the dataset was built with.

Both protocols induce a per-judge ranking of the K responses, so the IJA computation
(pairwise Kendall τ-b, averaged over judge pairs) is identical in both cases. Two
consequences of this choice are worth stating explicitly.

*First*, IJA values are compared
*within* a dataset, never pooled across datasets: scoring produces tie-heavy rankings while
K-wise ranking produces total orders, so the two IJA scales are not interchangeable, and
any cross-dataset IJA difference would partly reflect the protocol rather than the data.

*Second*, the single-judge margin baseline is protocol-specific: on UltraFeedback it is the
score gap between chosen and rejected, and on Nectar it is the rank-position gap in the
judge's ordering.

**Protocol ablation.** On a Nectar subset the same panel also runs the scoring protocol, so
we can check that the two protocols flag substantially the same low-agreement datapoints
i.e. that IJA measures a property of the data rather than of the elicitation format.

**Fixed candidate strings.** Every judge must score the exact identical K response texts for a given
prompt, so that any disagreement reflects the judges and not drift in the inputs.

## 5. Related work and positioning

**The problem is well-known.** Human annotators agree on preference judgments only about
60–75% of the time; the [data-centric RLHF metrics paper](https://arxiv.org/pdf/2409.09603)
floats annotator disagreement as a filter for low-quality preference data but does not
build one.

**Noise-robust losses.** [Provably robust DPO](https://arxiv.org/pdf/2403.00409) and
[soft preference labels](https://arxiv.org/pdf/2409.06691) change the loss so that noisy
labels hurt less — complementary to us: they cope with noise during training, we identify it
beforehand. Each has an uncertainty knob the method itself cannot measure: robust/conservative
DPO's flip probability ε is a global guessed hyperparameter, and geometric-averaged
preference optimization consumes a per-pair probability p that in their experiments comes
from *simulated* annotators. `pair_agreement` is that missing measurement — p directly, or
ε̂ = 1 − p (§3) — upgrading a global constant to a measured per-example quantity with no
change to their loss code, and giving Phase 3 its third arm (soft-labeling) alongside
filtering and down-weighting.

**Counterargument.** A recent
[preference-dataset curation study](https://arxiv.org/pdf/2511.10985) reports that
UltraFeedback and LMSYS are "fairly robust to label flipping" — DPO may simply shrug off the
noise we propose to filter. Phase 3 must show a real gain over no filtering and margin
filtering, or the thesis fails; this is the paper to argue against directly.

**Closest mechanisms.** [Reward-model ensembles](https://arxiv.org/html/2310.02743v2) use
disagreement for uncertainty, but during RL rather than dataset curation, and usually share
a base model — the correlated-judge situation §2 avoids. Margin filtering is the standard
cheap baseline, but a large margin from a *single* judge can still be contested across a
panel — the gap IJA catches.
[Cross-model disagreement](https://arxiv.org/pdf/2604.17112) applies our mechanism to
QA/hallucination detection instead, and BSDetector-style confidence filtering (surveyed in
[Data Tsunami](https://arxiv.org/html/2408.02085v3)) uses a single model's self-consistency —
reproduced as our baseline ablation.

**Panels and agreement measurement.** [PoLL](https://arxiv.org/html/2404.18796v1) showed
cross-family panels beat a single strong judge for *model* evaluation, but aggregates away
the disagreement we care about. [Nine Judges, Two Effective
Votes](https://arxiv.org/html/2605.29800) is the sharpest warning: correlated errors can
reduce nine judges to about two independent votes — the main threat to H1 and the reason for
cross-family selection (§2). Reading per-item agreement as signal is well precedented in
human annotation — Krippendorff's α, ChaosNLI's per-item label entropy (Nie et al. 2020),
CrowdTruth's per-unit ambiguity, surveyed by
[Uma et al. 2021](https://www.jair.org/index.php/jair/article/view/12752) — and
[Plank](https://arxiv.org/pdf/2211.02570) reminds us disagreement can be legitimate ambiguity
rather than error, exactly what Phase 4 asks about our low-IJA datapoints.

### Novelty statement (draft)

> Standard preference pipelines binarize a single judge's (or single judge's multi-aspect)
> scores into chosen/rejected pairs, discarding any notion of disagreement. We instead run a
> diverse panel and treat per-prompt cross-judge rank agreement (average pairwise Kendall's
> τ-b) as an explicit label-reliability signal. We show it flags corrupted and contested
> preference labels that margin-based filtering misses, and export it as metadata on the
> standard `(prompt, chosen, rejected)` schema so
> that it composes with any DPO or reward-model trainer.

## 6. Experimental phases

### Phase 0 — Novelty check *(ongoing)*

Before submission we re-run the targeted literature searches ("inter-judge agreement
preference data filtering," "rank correlation judges DPO data," "annotator agreement
preference label noise LLM"). The July 2026 sweep found nothing that uses per-datapoint panel
agreement to curate preference data, but the check needs repeating close to writing time.

### Phase 1 — Controlled corruption study (cheap, and a standalone blog post)

This gives IJA a ground truth to be measured against, with no training runs. From raw
UltraFeedback we establish a clean reference ordering on a subset (strong-judge consensus
plus manual spot-checks), then corrupt datapoints in exactly known ways, subtle to blatant:
swapping two adjacent-ranked responses, flipping a clear winner with a clear loser,
replacing a response with an off-topic one, and padding a weak response with verbosity —
the length bias a single judge is prone to.

We then measure how well IJA separates corrupted from clean datapoints (AUROC), against a
single judge's score margin, the panel's mean margin, and chance. The key result is the
**breakdown by corruption type**: we predict IJA wins on the subtle swaps and length-baiting,
where one confident judge is fooled but a diverse panel splits, while blatant flips are
caught just as well by margin alone.

Size: ~1,000 prompts covers the headline AUROC claim (half corrupted → ~500 positives and
negatives, standard error near ±0.015); going to ~2,000 only matters for keeping the
per-type AUROCs tight (~125 corrupted items per type otherwise). A throwaway 200-prompt
pilot (§9) comes first, purely to shake out parsing and confirm the metric moves in the
expected direction.

### Phase 2 — Ablation experiments

Phase 2 reruns the Phase 1 measurement — AUROC to separate corrupted from clean
datapoints — while changing one component of the setup at a time, so every run is directly
comparable to the Phase 1 with four variations:

1. **Panel vs. self-consistency.** Replace the J different models with one model sampled
   J times at temperature above zero, recompute IJA, recompute AUROC. If this matches the
   real panel, model diversity adds nothing and the panel apparatus is unnecessary — so we
   settle it up front. This arm doubles as the BSDetector-style baseline (§5) and as the
   extreme case of correlated judges (§2).
2. **Panel size.** Recompute IJA from judge subsets of size J = 3, 5, 7, 12 and plot AUROC
   against J — the "twelve angry" curve — to find where adding judges stops helping.
3. **Agreement metric.** Recompute IJA with Spearman's ρ and with simple pairwise-winner
   agreement in place of Kendall's τ-b. Similar AUROC across the three shows the result is
   not an artifact of the metric choice.
4. **Candidate count (maybe).** Repeat the experiment at K = 7 (Nectar) and K = 2 (a single pair)
   alongside the K = 4 UltraFeedback base case, to see whether the signal strengthens with
   longer rankings and survives the degenerate pairwise case.

Every run also reports the judge–judge correlation matrix and dataset-level Krippendorff's α
as panel-health diagnostics. 

From the matrix we additionally report each judge's **mean
agreement with the rest of the panel** (its row average): a judge whose mean falls well below
the others is an outlier — misreading the guidelines, or simply a weak judge — and we rerun the headline AUROC with it dropped to check whether it was adding signal or noise.

### Phase 3 — Downstream DPO validation (expensive; only if Phases 1–2 show promising results)

Here we test whether filtering on IJA actually produces better models.

From the full raw
UltraFeedback pool we compare six data-selection policies: keep everything, filter by
margin, filter by IJA, filter by their intersection, use IJA as a soft label, and drop at random — the random-drop control holds the data budget fixed, so any gain comes from *which*
datapoints were removed, not from training on less data. Per policy: fine-tune a small model
(Qwen2.5-1.5B or Llama-3.2-1B) with length-normalized DPO following the Zephyr recipe, three
seeds, confidence intervals rather than single-run deltas. We evaluate on length-controlled
AlpacaEval-2, MT-Bench, and IFEval, and additionally train a reward model per filtered set
and report its RewardBench-style accuracy. Two constraints frame the phase: evaluation
judges must be disjoint from the panel (else the comparison is circular), and the
improvement has to clear the "robust to label flipping" null result from §5.

### Phase 4 — Human validation (runs in parallel with Phase 3)

Finally we check the signal against people: 200–300 prompts stratified by IJA, ranked by two
or three human annotators blind to what the panel said. The question is what low IJA
actually means — genuine human disagreement (legitimate ambiguity in the Plank sense) or
judge failure on datapoints humans find easy. The answer decides whether a practitioner
should drop low-IJA data or keep it soft-labeled, and whether high IJA really marks labels
humans agree are correct.

## 7. Rigor checklist

- Pin exact judge model versions; score at temperature 0 (except the self-consistency arm);
  cache every API response keyed on (model, prompt hash); seed everything.
- Pre-register the hypotheses in this repo before Phase 1 results come in.
- Report dollar cost per policy — the economics of a cheap panel of small judges is part of the
  argument, following PoLL.
- Publication path: a blog post after Phase 1, then a workshop (DMLR at ICML) or ACL/EMNLP
  Findings submission, and NeurIPS Datasets & Benchmarks if Phase 3 lands. The library and the
  exported IJA-annotated datasets are artifacts in their own right.

## 8. Future directions

### Categorical agreement (κ) for binary preference datasets, including Tülu 3

The ranking-based IJA in this plan needs K ≥ 3 candidates to be meaningful, so it does not
apply to the large family of datasets that ship as a single binary comparison — Tülu 3
(as released), HH-RLHF, and Chatbot Arena among them. Once the ranking case is established, a
natural extension is to define IJA for binary data using a **chance-corrected categorical
agreement** measure: Fleiss' κ across the J judges (or Cohen's κ pairwise). Each judge simply
picks the winner of the two responses, and κ measures how much the panel agrees beyond what we
would expect by chance. This is the principled version of the "fraction of judges agreeing"
that τ-b collapses to at K = 2, and it unifies the framework — Kendall's τ-b for rankings,
κ for binary choices, both chance-corrected pairwise agreement.

One technical caveat drives the design: κ is inherently a *population* statistic, because its
chance-correction term needs label marginals estimated across many items, so an honest
per-datapoint κ does not exist from J votes alone. The clean resolution is to filter per
datapoint on **observed** agreement (the proportion of concordant judge pairs, or the vote
entropy) while reporting κ at the dataset or subset level as a panel-health diagnostic — the
same split we already use for Krippendorff's α. This phase is where Tülu 3 re-enters as an
actual data source rather than merely an integration target, and it brings HH-RLHF and Arena
into scope as well.

### GRPO and on-policy RL

Using IJA to gate or scale per-prompt rewards inside GRPO-style group-relative RL, so that
prompts with low panel agreement contribute less to the advantage estimate. This reuses the
same panel machinery at the RL stage rather than the data-curation stage.

## 9. Library status

The instrument is implemented as the `twelve-angry-llms` package (v0.2.0, `src/`), with the
study-specific material kept separate in `research/`. The plan's concepts map to the package
as follows: `ScoringProtocol` and `RankingProtocol` implement the two elicitation styles of
§4 with shared prompts and validated parsing; `Panel` fans judges out asynchronously and
computes per-datapoint IJA (tie-aware Kendall's τ-b by default, Spearman and pairwise-winner
selectable for the metric ablation) plus the dataset-level diagnostics (Krippendorff's α and
the judge–judge correlation matrix); the exporter emits TRL's `(prompt, chosen, rejected)`
schema with `prompt_ija`, `pair_agreement`, and the raw per-judge values; clients cover
OpenRouter (the default, with provider/quantization pinning for reproducibility) and any other
OpenAI-compatible endpoint, with a SQLite response cache and token-usage
tracking satisfying the §7 rigor items; and raw UltraFeedback/Nectar loaders preserve the
original scores and ranks in metadata for the margin baselines.

The remaining step before Phase 1 is the pilot (`research/experiments/pilot.py`): 200
UltraFeedback prompts, half corrupted via `research/experiments/corruptions.py`, run through
a small panel to confirm the outputs parse, that τ-b actually spreads across datapoints, and
that a blatant flip visibly tanks the corrupted pair's agreement. Then scale up to Phase 1.

## 10. Reading list (priority order)

1. [Data-centric RLHF metrics](https://arxiv.org/pdf/2409.09603) — the motivation, and the
   unbuilt "disagreement as a filter" suggestion closest to our thesis.
2. [Preference-dataset curation study](https://arxiv.org/pdf/2511.10985) — the "robust to label
   flipping" null result we must beat.
3. [PoLL](https://arxiv.org/html/2404.18796v1) and
   [Nine Judges, Two Effective Votes](https://arxiv.org/html/2605.29800) — the panel framing and
   its sharpest critique.
4. [UltraFeedback](https://arxiv.org/pdf/2310.01377) — our primary data source and the source of
   the shared rubric template.
5. [Robust DPO](https://arxiv.org/pdf/2403.00409) and
   [soft preference labels](https://arxiv.org/pdf/2409.06691) — complementary methods that
   consume our soft-label output.
6. [Reward-model ensembles](https://arxiv.org/html/2310.02743v2) — the closest mechanism, at a
   different pipeline stage.
7. [Plank 2022](https://arxiv.org/pdf/2211.02570) and Uma et al. 2021 — the theory that
   disagreement can be signal rather than noise.
8. [Cross-model disagreement for UQ](https://arxiv.org/pdf/2604.17112) — evidence that the
   mechanism works in a neighboring domain.
9. [Tülu 3](https://arxiv.org/pdf/2411.15124) — the binary-data integration target for the
   future κ phase (§8).

*Note: the 2025–2026 arXiv entries above were found via search (July 2026) and read only at
abstract level so far; read them in full before citing in a paper.*
