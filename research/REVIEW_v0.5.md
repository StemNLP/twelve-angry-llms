# Review of RESEARCH_PLAN.md (v0.5) — threats to validity

This is a design review of the v0.5 plan. Each item names the **section / lines** it targets,
states the **reason** it is a problem, and gives a concrete **proposal** to fix it. Items are
ordered by how much they threaten the central thesis (H1). Nothing here is a copy-edit; these are
the points that, left unaddressed, could sink Phase 3 or a reviewer.

The two highest-leverage items — **#1** and **#2** — are cheap enough to run in the pilot (§9) and
should gate the expensive Phase 3 DPO runs.

---

## 1. Reliability ≠ validity, and shared judge bias contradicts the Phase 1 length-bait prediction

**Section / lines:** §1 "Why this framing is justified" (L56–58) vs. §6 Phase 1 (L249–250).

**Quote (§1, L57):** "We are not measuring whether the judges are *correct* … but we also do not
need it at this stage."
**Quote (Phase 1, L249):** "we predict IJA wins on the subtle swaps and length-baiting, where one
confident judge is fooled but a diverse panel splits."

**Reason.** The plan disclaims correctness, but H1 and Phase 3 require that low IJA correlates with
*wrong* labels, not merely *unstable* ones — reliability (reproducibility) and validity
(correctness) are different axes, and a panel can be reliably wrong. LLM judges share systematic
biases (length/verbosity, position, formatting, sycophancy), and these are *correlated across
families* because open judges (Qwen, Llama, Mistral) are heavily trained on GPT-4-distilled data.
On a length-baited corruption the panel is therefore likely to be fooled *together* → high
agreement → **high IJA on exactly the corruption Phase 1 predicts IJA catches.** The length-bait
case is the one most likely to falsify H1, not confirm it. This is an internal contradiction
between §1 and Phase 1.

**Proposal.**
- On the Phase 1 clean-reference subset (where truth is known), directly measure the
  **correlation of judge errors** — do judges fail on the same items?
- Split corruption types into **shared-bias** (length, formatting) vs **idiosyncratic-noise**
  (adjacent swaps) and only claim IJA catches the latter.
- Soften the §1 claim: IJA measures reliability *conditional on judges not sharing the relevant
  bias*; report where that condition fails. If IJA is high on length-baits, report it as a finding
  and reframe the contribution as an *idiosyncratic-noise* detector.

---

## 2. Synthetic corruptions don't establish that IJA catches the noise that actually hurts DPO

**Section / lines:** §6 Phase 1 (L237–256); the ready-made resource is already cited in §1 (L19–23).

**Quote (Phase 1, L240–244):** "corrupt datapoints in exactly known ways … swapping two
adjacent-ranked responses, flipping a clear winner … padding a weak response with verbosity."

**Reason.** Phase 1's AUROC shows IJA separates *artificial* perturbations, not the *naturally
occurring* label noise that degrades training. The §5 counterargument (2511.10985, "robust to label
flipping") most plausibly holds because natural noise is near-ties — legitimate ambiguity (Plank),
which is what Phase 4 goes looking for. If low-IJA items are mostly "both responses are fine,"
filtering removes *harmless* pairs and Phase 3 shows no gain — even with a beautiful synthetic
AUROC. Synthetic corruption cannot distinguish these cases because it manufactures a real quality
gap.

**Proposal.** Add a **natural-noise arm** to Phase 1 using a resource the plan already cites: the
two UltraFeedback binarizations (HuggingFaceH4 vs Argilla-cleaned, L20–23) disagree on the chosen
response for many prompts *after corrupted scores were found*. That disagreement set is a
naturally-occurring ground truth of real mislabels with a known corrected answer. Test whether IJA
flags the Argilla-corrected flips. This is far stronger evidence than synthetic AUROC and costs
almost nothing.

---

## 3. Cross-family independence is load-bearing but marked "not a hypothesis to test"

**Section / lines:** §2 "A note on panel selection" (L72–77); diagnostics in §6 Phase 2 (L278–283).

**Quote (L75–76):** "We treat this as a design principle rather than a hypothesis to test, and
monitor it through the judge–judge correlation matrix and Krippendorff's α."

**Reason.** Panel independence is the assumption the entire method rests on; declaring it a design
principle is backwards. Different families are *not* independent — Qwen/Llama/Mistral instruction
data is substantially GPT-4-generated, so their preferences are partly inherited from the very
GPT-4 that built UltraFeedback. The plan itself cites "Nine Judges, Two Effective Votes" (L208–210)
as the main threat: correlated errors can collapse the panel to ~2 effective votes. Relegating the
correlation matrix and α to "diagnostics reported alongside runs" hides the test that could sink
the method.

**Proposal.** Promote **effective number of votes** to a *primary* reported result, computed
properly (participation ratio / eigenvalue spread of the judge score-correlation matrix, or the
α-implied information reduction). If ≥4, the panel is justified; if ~2, honestly reframe as a
"two-judge disagreement flag" (still publishable) rather than a J-judge panel.

---

## 4. Phase 3's IJA filter changes the data *distribution*, not just its *quality*

**Section / lines:** §6 Phase 3 (L285–298), specifically the random-drop control (L291–293).

**Quote (L291–293):** "the random-drop control holds the data budget fixed, so any gain comes from
*which* datapoints were removed, not from training on less data."

**Reason.** Filtering out low-IJA prompts removes the harder / more ambiguous / more diverse
prompts, so the IJA-filtered set is systematically easier and narrower. The random-drop control
holds *budget* fixed but samples uniformly, preserving the difficulty distribution that IJA-filter
destroys. A gain on AlpacaEval/MT-Bench could therefore be **distribution shift** (trained on an
easier slice), not noise removal — and the current control cannot separate the two.

**Proposal.**
- Add a **difficulty-stratified control**: drop random pairs *within* IJA strata, matching
  difficulty while not targeting specific low-quality items. A gain that survives this is real.
- Break evaluation down by prompt ambiguity/difficulty (you already have per-prompt IJA) so a
  narrowing effect is visible rather than hidden in the aggregate.

---

## 5. The soft-label estimator ε̂ = 1 − p is fragile at J = 5 and cannot represent p < 0.5

**Section / lines:** §3 soft-label mapping (L107–118).

**Quote (L116–118):** "if p = `pair_agreement` is the fraction of judges preferring the chosen
response, then ε̂ = 1 − p. A unanimous pair (p = 1) trains at full strength; a 3–2 split
(ε̂ = 0.4) contributes close to zero."

**Reason.** With J = 5, p ∈ {0, 0.2, 0.4, …} — an extremely coarse, high-variance estimate fed
into cDPO as if it were a Bernoulli mislabel *probability*. If judges are correlated (see #3), p is
not an unbiased estimate of a flip probability at all. Worse, a 2–3 split *against* the dataset's
chosen response gives ε̂ = 0.6 > 0.5, which cDPO structurally cannot represent — that case means
"relabel," not "noisy," and the plan does not handle it.

**Proposal.**
- Treat **p < 0.5 as a relabel candidate**, reported explicitly — arguably the most interesting
  output of the panel.
- Shrink p toward 0.5 with a Beta prior to reflect the tiny sample; report ε̂ variance.
- Use a larger panel for the soft-label arm specifically, and lead with the geometric-averaging
  path (2409.06691), which is more forgiving of a noisy p than cDPO.

---

## 6. Phase 1's clean base pool is likely selected for high agreement, inflating AUROC

**Section / lines:** §6 Phase 1 (L240–241).

**Quote (L240–241):** "we establish a clean reference ordering on a subset (strong-judge consensus
plus manual spot-checks), then corrupt datapoints in exactly known ways."

**Reason.** To obtain stable orderings to corrupt, the clean base will be drawn from high-consensus
datapoints. The "clean" negatives are then artificially high-IJA, so IJA separates clean vs
corrupted partly *because the clean set was pre-filtered to be high-agreement* — a base-rate
artifact, not a property of real data. Real preference data spans the full IJA spectrum, so the
reported AUROC will be optimistic relative to deployment. There is also mild circularity: the
ground truth is defined by judge consensus and the metric is judge consensus.

**Proposal.** Sample the clean base pool to **match the natural IJA distribution** of raw
UltraFeedback, or at minimum report AUROC conditioned on the base item's pre-corruption agreement.
State the circularity explicitly and rely on #2's natural-noise arm as the un-circular check.

---

## 7. The economic argument has no decision threshold, and the baseline it must beat is nearly free

**Section / lines:** §7 rigor checklist (L314–315); baseline framing in §5 (L198–200) and Phase 3.

**Quote (L314–315):** "Report dollar cost per policy — the economics of a cheap panel of small
judges is part of the argument, following PoLL."

**Reason.** Margin filtering — the main baseline — costs *zero extra*: the margin was already
computed by the single judge that built the dataset. IJA costs J× the judging of the full raw pool.
"Beats margin on AUROC" is not "worth 5× the money." If margin captures most of the benefit for
free, IJA loses in practice even while winning the plot, and the plan never states the bar.

**Proposal.** Pre-register a **decision criterion**: minimum gain-per-dollar over margin and a
minimum absolute DPO delta. Report a **cost-matched frontier** — what a single strong judge scoring
K responses buys vs the panel at equal dollar spend.

---

## 8. Phase 4 shares the reliability ceiling of the thing it measures

**Section / lines:** §6 Phase 4 (L300–307).

**Quote (L301–302):** "200–300 prompts stratified by IJA, ranked by two or three human annotators
blind to what the panel said."

**Reason.** Phase 4 must decide whether low IJA is legitimate human ambiguity or judge failure, but
human preference IAA is only 60–75% (the plan's own §5, L174–175) — and it is *lowest* precisely on
the low-IJA items that matter most. Two or three annotators cannot establish reliable per-item human
ground truth on the hardest items; the instrument has the same reliability ceiling as the target.

**Proposal.** Either scale annotators-per-item on the low-IJA stratum (5+), or reframe the
deliverable away from per-item ground truth toward the **correlation between human IJA and LLM
IJA**. Agreement on *which* items are contested supports "legitimate ambiguity" without needing a
gold label.

---

## 9. Minor: metric coarseness is the real issue, not τ-b vs ρ

**Section / lines:** §1 "How IJA is measured" (L43–50).

**Quote (L43–44):** "We use Kendall's τ-b … First, K is small (often just 4), and on so few items
Spearman's ρ is jumpy … whereas Kendall's τ … behaves more stably."

**Reason.** τ-b at K = 4 has only ~6 discrete levels (C(4,2) pairs) — nearly as coarse as the
Spearman it rejects. The honest concern is not the metric choice but that per-datapoint IJA from
K = 4 × J = 5 is a **low-information estimate**; thresholding on it may be thresholding on noise.

**Proposal.** Add a stability check to the Phase 2 metric ablation: show the datapoint *ranking* by
IJA is stable under judge resampling (bootstrap the panel). Reframe the §1 justification around
estimator variance rather than τ-vs-ρ.

---

## 10. Minor: novelty is positional, not mechanistic

**Section / lines:** §5 "Novelty statement" (L218–226); reward-model ensembles (L196–197).

**Reason.** Disagreement-as-uncertainty is exactly what reward-model ensembles already do; the moat
is "nobody did it at the curation stage," which rests entirely on Phase 0 staying empty — a thin
claim for NeurIPS D&B. The durable contribution is the empirical characterization (Phases 1–4).

**Proposal.** Lead positioning with the empirical characterization and the exported artifact, not
"new signal." Keep the mechanistic-novelty claim modest to avoid an easy reviewer rejection.

---

### Suggested gate before Phase 3

Run **#2** (natural-noise arm on the UltraFeedback binarization-disagreement set) and **#1**
(judge-error correlation on the clean subset) inside the pilot. Both are cheap, reuse resources the
plan already cites, and together either de-risk the thesis or kill it before the expensive DPO runs.
