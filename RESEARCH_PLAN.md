# Research Plan: Inter-Judge Agreement as a Reliability Signal for Preference Data

*Working title: "Twelve Angry LLMs: Judge Agreement as a Label-Reliability Signal for Preference Data"*
*Status: draft v0.3 (preference-data framing, prose pass) — 2026-07-09*

## 1. The core idea

Modern preference datasets are built by showing a language model a prompt together with
several candidate responses and asking a single strong judge (almost always GPT-4 or GPT-4o)
to score or rank them. Those scores are then collapsed into a single `(chosen, rejected)`
pair that gets used for DPO or reward-model training. Our claim is that the collapsing step
throws away a useful signal, and that we can recover it by replacing the single judge with a
**panel** of diverse judges and measuring how much they agree with each other on each
datapoint.

We call that per-datapoint quantity the **inter-judge agreement (IJA)**. The intuition is: if five different models all rank four candidate responses in nearly the same order, the resulting preference label is trustworthy; if they disagree sharply, the label is shaky and is likely to inject noise into training. IJA is therefore a measure of *label
reliability*, computed cheaply and automatically, before any training happens.

### What the target datapoint looks like

A preference datapoint is a prompt together with K candidate responses, where K is typically
4 or more. This is the shape data takes at the *judging stage* of the standard pipelines
(UltraFeedback, Nectar). Importantly, most of these datasets ship in a reduced binary form (i.e. just the chosen and rejected response) because the K candidates were scored and then
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
ties (under the ranking protocol judges produce total orders, where τ-b reduces to plain τ). Third, τ has a direct
reading that matches what we care about: it is essentially the fraction of response-pairs that
two judges order the same way, which *is* preference agreement. We still report Spearman's ρ
in the ablations so that readers can see the result is not an artifact of the metric choice.

J (the number of judges) is itself a variable of the study; we sweep it in Phase 2 and use a
default panel of five judges elsewhere.

### Why this framing is the right one

Agreement between judges on the ranking is a direct measure
of how reliable the training label is, with no separate "goodness" axis to worry about. A practical example is the use of this method to filter out training datapoints based on their reliability level. A strict filtering can be applied for example in critical domains to ensure the the training data was highly reliable.

## 2. Hypotheses

**H1 (core claim).** IJA measures preference-label reliability. Datapoints where the panel
disagrees produce noisy chosen/rejected pairs, and using IJA to filter, down-weight, or
soft-label those pairs improves DPO outcomes compared with (a) doing no filtering and (b) the
standard practice of margin-based filtering (keeping only pairs where a single judge assigned
a large score gap).

**H2 (diversity claim).** A panel of judges drawn from *different* model families produces
more informative IJA than a same-family panel of the same size, or than a single model sampled
K times. The reasoning is that judges sharing a base model tend to make correlated errors and
agree on their shared blind spots, which inflates agreement without adding information.

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

Two things follow from this. First, judging four or more candidates per prompt is already the
norm upstream, so IJA does not ask anyone to change how they generate data — it inserts a
panel at a stage that already exists. Second, almost everyone ultimately trains through the
`(prompt, chosen, rejected)` interface that TRL's `DPOTrainer` expects. Our exporter therefore
emits exactly that schema, with IJA carried alongside as extra metadata columns:
`prompt_ija` (the Kendall τ-b agreement over the K candidates), `pair_agreement` (the fraction
of judges that preferred the chosen response over the rejected one), and the raw per-judge
scores or rankings. A practitioner can then filter or weight on those columns and run their
existing DPO
setup unchanged.

The `pair_agreement` column is also directly usable as a **soft label**: instead of treating
every pair as a hard 1/0 preference, methods like conservative DPO or
[geometric-averaged preference optimization](https://arxiv.org/pdf/2409.06691) can consume the
fraction directly. Extending IJA to gate or scale per-prompt rewards inside GRPO-style RL is a
natural but separate line of work, noted here as future direction (§8).

## 4. The judging protocol

Because the mentioned datasets used a single judge, we cannot calculate IJA off them. We have to run
the panel ourselves. We start with judging a thousand prompts with several models.

**One shared rubric across all judges.** Within a dataset, every judge receives the *same* guideline — the definition of what a score of 1 through 5 means (scoring protocol) or the criteria by which responses are to be ordered (ranking protocol) — so that disagreement reflects the judges, not prompt wording. We do not give different judges different prompts. UltraFeedback releases its full annotation template 
(in `src/data_annotation/preference_templates.py`), so we adopt that as the canonical rubric for the scoring protocol;

Nectar publishes only an excerpt of its rubric and defers the position-bias handling to an
unreleased writeup, so it cannot serve as a reproducible template; for the ranking protocol
we write a single explicit ranking instruction shared by all judges instead.

**Elicitation matches each dataset's native style.** Rather than forcing one protocol on
both datasets, the panel judges each dataset the way its original pipeline did: on
UltraFeedback the judges score each response 1–5 against the shared rubric (per-response
scoring, the UltraFeedback style), and on Nectar the judges rank the seven responses
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

**Fixed candidate strings.** Every judge must score the identical K response texts for a given
prompt, so that any disagreement reflects the judges and not drift in the inputs.

**A reproduction baseline.** Because UltraFeedback's original template is public, we include a
run in which one panel member uses that exact template, letting us confirm our pipeline
reproduces the published GPT-4 labels before we introduce the rest of the panel.

## 5. Related work and positioning

**The problem is well-known and quantified.** Human annotators only agree on preference judgments
about 60–75% of the time, and disagree on 30–50% of the subtle comparisons; the
[data-centric RLHF metrics paper](https://arxiv.org/pdf/2409.09603) explicitly floats
annotator disagreement as a filter for low-quality preference data but does not build one.
That unbuilt suggestion is close to our thesis and worth citing prominently.

**Others make training tolerate noise rather than remove it.** Work such as
[provably robust DPO](https://arxiv.org/pdf/2403.00409) and
[soft preference labels](https://arxiv.org/pdf/2409.06691) changes the loss so that noisy
labels hurt less. This is complementary to us: they cope with noise during training, we
identify it beforehand, and in fact our soft-label output feeds directly into their methods.

**The null result we must beat.** A recent
[preference-dataset curation study](https://arxiv.org/pdf/2511.10985) reports that
UltraFeedback and LMSYS are "fairly robust to label flipping," meaning DPO may simply shrug
off the very noise we propose to filter. Phase 3 has to show a real gain over both no
filtering and margin filtering, or the whole thesis is in doubt — so this is the paper to
argue against directly.

**Closest mechanisms.** [Reward-model ensembles](https://arxiv.org/html/2310.02743v2) use
disagreement among reward models to estimate uncertainty, but during RL rather than for
dataset curation, and their ensembles usually share a base model — which is exactly the
correlated-judge situation H2 warns about. Margin-based filtering is the standard, cheap
baseline; note that a large margin from a *single* judge can still be contested across a
panel, which is precisely the gap IJA is meant to catch.
[Cross-model disagreement for uncertainty quantification](https://arxiv.org/pdf/2604.17112)
uses the same mechanism as us but for QA/hallucination detection, not training-data curation,
and BSDetector-style confidence filtering (surveyed in
[Data Tsunami](https://arxiv.org/html/2408.02085v3)) relies on a single model's
self-consistency — which our self-consistency ablation reproduces as a baseline.

**Panels and agreement measurement.** [PoLL](https://arxiv.org/html/2404.18796v1) established
that cross-family panels beat a single strong judge for *model* evaluation, but it aggregates
the panel and discards the disagreement we care about. [Nine Judges, Two Effective
Votes](https://arxiv.org/html/2605.29800) is the sharpest warning to us: judges make
correlated errors, so a nine-judge panel can carry the information of only about two
independent votes — this is the main threat to H1 and the motivation for H2. The idea of
reading per-item agreement as signal is well precedented in the human-annotation literature:
Krippendorff's α is built from per-unit pairwise disagreement, ChaosNLI (Nie et al. 2020) uses
per-item label entropy, CrowdTruth defines per-unit ambiguity, and
[Uma et al. 2021](https://www.jair.org/index.php/jair/article/view/12752) survey the area.
Plank's [work on human label variation](https://arxiv.org/pdf/2211.02570) reminds us that
disagreement is sometimes legitimate ambiguity rather than error, which is exactly the
question Phase 4 asks about our low-IJA datapoints.

### Novelty statement (draft)

> Standard preference pipelines binarize a single judge's (or single judge's multi-aspect)
> scores into chosen/rejected pairs, discarding any notion of disagreement. We instead run a
> diverse panel and treat per-prompt cross-judge rank agreement (average pairwise Kendall's
> τ-b) as an explicit label-reliability signal. We show it flags corrupted and contested
> preference labels that margin-based filtering misses, quantify how panel diversity governs
> the signal, and export it as metadata on the standard `(prompt, chosen, rejected)` schema so
> that it composes with any DPO or reward-model trainer.

## 6. Experimental phases

### Phase 0 — Novelty check *(ongoing)*

Re-run the targeted searches before submission ("inter-judge agreement preference data
filtering," "rank correlation judges DPO data," "annotator agreement preference label noise
LLM"). The July 2026 sweep found nothing using per-datapoint panel agreement to curate
preference data, but this needs a final check close to writing.

### Phase 1 — Controlled corruption study (cheap, and a standalone blog post)

The first real experiment gives IJA a ground truth to be measured against, without any
training runs. We take a set of prompts with their K candidate responses from raw
UltraFeedback, establish a clean reference ordering on a subset (strong-judge consensus plus
manual spot-checks), and then inject controlled corruptions whose nature we know exactly:

- swapping two adjacent-ranked responses (a subtle error),
- flipping a clear winner with a clear loser (a blatant error),
- replacing one response with an off-topic or degenerate one,
- padding a weak response to test whether IJA resists the length/verbosity bias that a single
  judge is prone to.

We then run the panel and ask how well IJA detects the corrupted datapoints, measured by AUROC,
against three comparisons: a single judge's score margin, the panel's mean margin, and chance.
The most informative result is the **breakdown by corruption type** — our prediction is that
IJA wins specifically on subtle swaps and length-baiting, where a single confident judge is
fooled but a diverse panel splits, while blatant flips are caught by margin alone.

On size: roughly 1,000 prompts is enough for the headline AUROC claim (with about half
corrupted, that is ~500 positives and ~500 negatives, giving a standard error near ±0.015),
and it is cheap. The one reason to go to ~2,000 is the per-corruption-type breakdown, where
each type otherwise gets only ~125 corrupted items; a larger pool keeps those per-type
AUROCs tight. A throwaway 200-prompt pilot (see §9) comes first, purely to shake out parsing
and confirm the metric moves in the expected direction.

### Phase 2 — Ablations that turn the result into a paper

- **Panel diversity (tests H2):** a cross-family panel versus a same-family panel of equal
  size. This is the direct answer to the correlated-errors critique.
- **Panel versus self-consistency:** J different models versus one model sampled J times at
  temperature > 0. If self-consistency does nearly as well, reviewers will ask, so we answer
  it up front; this arm also serves as the BSDetector baseline.
- **Panel size (the "twelve angry" curve):** J = 3, 5, 7, 12, to see where the agreement signal
  saturates.
- **Agreement metric:** Kendall's τ-b versus Spearman's ρ versus a simple pairwise-winner
  agreement, to show robustness to the choice.
- **Candidate count K:** K = 4 (UltraFeedback) versus K = 7 (Nectar), and the degenerate K = 2
  case, to characterize how IJA behaves as the ranking gets longer or collapses to a single
  comparison.
- Throughout, we report the judge–judge correlation matrix and dataset-level Krippendorff's α
  as diagnostics of the panel's health.

### Phase 3 — Downstream DPO validation (expensive; only if Phases 1–2 hold)

Here we test whether filtering on IJA actually produces better models. We start from the full
raw UltraFeedback pool and compare selection policies: no filter, margin filter, IJA filter,
the intersection of margin and IJA, IJA used as a soft label, and a random-drop control that
holds the data budget fixed so we are not just measuring the effect of training on less data.
For each policy we fine-tune a small model (Qwen2.5-1.5B or Llama-3.2-1B) with
length-normalized DPO following the Zephyr/UltraFeedback recipe, across three seeds, and report
confidence intervals rather than single-run deltas. Evaluation uses AlpacaEval-2 (length-
controlled), MT-Bench, and IFEval, plus the RewardBench-style accuracy of a reward model
trained on each filtered set. The evaluation judges must be disjoint from the panel to avoid
circularity, and the bar to clear is the "robust to label flipping" null result from §5.

### Phase 4 — Human validation (runs in parallel with Phase 3)

We take ~200–300 prompts stratified by their IJA value and have 2–3 people, blind to the panel,
rank the same K responses. This answers whether low IJA really tracks genuine human
disagreement (i.e., legitimate ambiguity in the Plank sense) versus judge failure, and whether
high IJA corresponds to labels humans agree are correct. The distinction matters for how a
practitioner should treat low-IJA data — drop it, or keep it with a soft label.

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

## 9. Library and API starting points

1. Add a `ScoringTask` that asks each judge to score the K responses in a group against the
   shared 1–5 rubric, with per-judge score extraction; keep the existing `RankingTask` for
   judges that rank directly.
2. Extend `JuryResult` with an `ija` field (average pairwise Kendall's τ-b, tie-aware) next to
   the existing `agreement`, plus dataset-level diagnostics (Krippendorff's α and the
   judge–judge correlation matrix).
3. Build the **exporter**: a Hugging Face dataset in TRL's `(prompt, chosen, rejected)` schema
   with `prompt_ija`, `pair_agreement`, and `judge_scores` columns, so it drops straight into
   `DPOTrainer`.
4. Implement real `LLMClient` backends for three providers, with a response cache.
5. Run the pilot: 200 UltraFeedback prompts, 4 responses each, 3 judges, clean versus
   corrupted, to confirm the scores parse, that τ-b actually spreads across datapoints, and
   that a blatant flip visibly tanks IJA. Then scale up to Phase 1.

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
