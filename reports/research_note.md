# Can an agent program Swedish grammar into Gemma's weights? A tiered, controlled test

**Result in one line.** Agent-designed MLP edits shifted Gemma 4 12B IT's likelihood margins toward grammatical Swedish on held-out
pairs, by +0.04 to +0.16 nats/token. This is far beyond matched random edits. But held-out Swe-BLiMP accuracy rose only +0.6 to +1.3 pp,
with every interval crossing zero, and never clearly beat the random control. Other languages were preserved. Large dev gains
(+11.5 to +23 pp) did not survive to test. **Verdict on H1: not supported at the pre-registered threshold (≥ +2 pp, CI > 0).**

## Question and why it matters
Swedish models need precise grammar, and SE.LLMA must decide where to spend adaptation effort. *Weight-space programming* (Takehi et al., 2026)
suggests an LLM can read and write network weights directly. I tested the narrowest useful version of that claim. Can a coding agent,
working without gradient training and from a small labelled budget, edit a capable model's MLP weights so that it prefers grammatical Swedish
on unseen minimal pairs? The edit must beat a matched random edit and must not lose ground in other languages.

## The article, read critically
The clean "reading" results concern tiny networks: identifying a game, a DDPM's shape, or a char-LM's language and induction heads. The
language-model results lean on behaviour and activations. The sycophant model could *not* be identified from weights alone, only once attention patterns were
allowed. The flagship Qwen3.5-0.8B result is a **repair**: one MLP direction behind reasoning loops was removed (131k weights), with
budgets of 50 batched, long-rollout forward passes. "No gradients" therefore still means iterative optimisation on task feedback. The one clean
*capability* attempt (Gemma 3 270M, which rarely loops) produced no established gain. The authors themselves conclude that edits remove
obstacles rather than add ability. The posts include no matched random or untargeted controls, so they establish feasibility, not attribution.

## Setup
- **Model and precision.** `google/gemma-4-12B-it` @ `43c9d01`, float32 weights and compute.
- **Pair scoring.** Each sentence is scored by mean token log-probability as the model's reply to "Skriv en mening på svenska.". The brief's
  raw BOS-only scoring is degenerate for this IT model, giving 52% on Swe-BLiMP. It is still reported as a secondary protocol.
- **Benchmarks and splits.** Swe-BLiMP (2,458 test pairs; primary), Swedish Idioms (430), ScaLA sv/da/nb/en (256 each; 12-shot greedy, strict
  parser), and Belebele in 7 languages (201 aligned questions). Each benchmark reserves nested dev tiers: 26/52/104, 12/24/48, 8/16/32 and
  10/20/40 items (seed 42).
- **Editing procedure.** Each tier restarts from the original weights. The agent (Claude Opus 5.5) inspected activations and weights, evaluated at most
  4 rounds × 4 candidates on that tier's full dev suite, and locked by a pre-registered rule.
- **Controls and sealing.** A matched random control has the same matrices, rank and singular values, with random singular vectors (seed 43).
  Edited-model test scores were sealed in code until all tiers were locked.
- **Statistics.** 95% paired cluster bootstrap (10,000 draws).

**What the agent found and edited.** Language-direction and grammar-contrast projection edits did nothing useful. The agent did find a
consistent signal in direct effects: a first-order readout of how scaling each late-layer MLP neuron would move the good-minus-bad margin.
The readout uses no backpropagation through layers. Every locked edit is a neuron-scaling edit in `down_proj`. Per layer, the 32 neurons that
most favour grammatical sentences are scaled up and the 32 that most favour ungrammatical ones are scaled down:

| tier | layers | scaling | changed entries | relative norm per matrix |
|---|---|---|---|---|
| 1 | 44–47 | ×1.5 / ×0.5 | 0.98M | 2.5–2.9% |
| 2 | 40–47 | ×1.5 / ×0.5 | 1.97M | 2.4–3.0% |
| 3 | 40–47 | ×1.25 / ×0.75 | 1.97M | 1.2–1.5% |

Within dev, these neurons beat random neurons in every check. Cross-validation showed little transfer to *unseen phenomena* once tier 3 had
enough pairs. Transfer to unseen *examples* within seen phenomena was modest.

## Results (held-out test; Δ vs unchanged model; pp unless stated)
| | tier 1 | tier 2 | tier 3 (primary) |
|---|---|---|---|
| dev Swe-BLiMP Δ (in sample, at lock) | +23.1 | +17.3 | +11.5 |
| **Swe-BLiMP macro acc. Δ** (base 69.08%) | +1.30 [−0.04, +2.58] | +1.29 [−0.15, +2.73] | **+0.61 [−0.42, +1.65]** |
| Swe-BLiMP margin Δ (nats/token) | +0.04 [+0.02, +0.06] | +0.16 [+0.13, +0.18] | +0.11 [+0.09, +0.12] |
| edit − matched random, accuracy | +0.62 [−0.69, +1.89] | +0.97 [−0.48, +2.37] | +0.32 [−0.70, +1.37] |
| edit − matched random, margin | +0.04 [+0.02, +0.06] | +0.16 [+0.14, +0.19] | +0.11 [+0.10, +0.12] |
| random control − unchanged, accuracy | +0.68 [+0.27, +1.13] | +0.32 [−0.13, +0.81] | +0.29 [−0.07, +0.67] |
| Idioms acc. Δ (base 73.5%) | −0.47 [−3.49, +2.56] | −1.16 [−4.42, +2.09] | +0.23 [−2.09, +2.56] |
| ScaLA-sv acc. / MCC×100 Δ (base 82.0 / 64.3) | +0.39 / +0.68 | −0.39 / −0.74 | −0.39 / −0.74 |
| ScaLA da, nb, en acc. Δ | 0, +0.39, +0.39 | 0, +0.78, 0 | +0.39, +0.39, +0.78 |
| Belebele, worst language Δ (7 langs) | −0.50 (swe, zho) | −0.50 (nob) | 0.00 |
| raw-BOS Swe-BLiMP (secondary) Δ | −0.35 | +0.17 | −0.04 |

All 21 Belebele language × condition intervals lie above the −3 pp tolerance, so preservation is supported for each language (exploratory,
no joint claim). The sensitivity analyses agree with the primary metric. Tier-3 total-log-prob accuracy is +0.81 pp, and the equal-token-length
subset (n = 1,633) is +0.61 pp. The edits **reshuffle** outcomes rather than uniformly improving them. Tier 3 corrected 91 pairs and broke 76;
tier 2 corrected 182 and broke 150. Gains cluster in the word-order phenomena where the baseline is weakest (RESUMPTION 1 +9 to +22 pp,
EMBEDDED V3 OBJ. +8 to +9, V3 PP +5 to +17). Losses fall in V3, V2 REL. CL. and ARG. STR. (`reports/phenomenon_changes.png`).

## What this supports, and what it does not
- **Supported.** A no-backprop, activation-plus-weights procedure can find a handful of late-layer neurons whose rescaling moves held-out
  likelihood margins toward grammatical Swedish. The move is reliably larger than a spectrum-matched random edit, and costs essentially nothing
  in seven languages. The direction-finding adds value on margins.
- **Not supported.** A useful accuracy gain, or an accuracy advantage over random. The margin shift is spread across pairs; in tiers 2–3 it is larger
  on pairs already far from the decision boundary (+0.11 to +0.16) than near it (|margin| < 0.1: +0.04 to +0.10). Even random edits move accuracy by up to +0.7 pp, which sets the noise floor for "improvement"
  claims at this scale. There was no transfer to Idioms, ScaLA or Belebele-sv.
- **Selection bias is the dominant effect.** The pre-registered lock rule ranks candidates by in-sample dev gain. Those gains were 10 to 20
  times the test gains. Within-dev cross-validation predicted the margin well but still overstated accuracy: tier 3's within-phenomenon CV forecast
  +0.07 margin and +5.8 pp, against +0.11 margin and +0.6 pp on test. **More dev data did not buy a larger test gain** (26 → 104 pairs: +1.3 → +0.6 pp).
  The later tiers also chose a gentler strength.
- **Limits.**
  - Swe-BLiMP probes likelihood preference, not writing quality, and the primary score depends on a frozen chat prompt.
  - Templates may leak across splits, and pretraining contamination is unknown.
  - Intervals are conditional on the chosen edits and split.
  - Agent knowledge carried from tier to tier (disclosed). The agent had seen baseline test results and an earlier run's negative results before tier 1.
  - Direct-effect attribution uses an analytic readout derivative; there is no backpropagation through any layer, but it is not "weights only".

## Recommendation for SE.LLMA and next experiment
Do not use agent-driven weight edits as a route to Swedish grammatical competence in a model of this class: the measured gains are within
random-perturbation noise. Spend the budget on data and tuning. Keep the method as an **interpretability probe**, because it cheaply isolates
late-layer neurons whose effects concentrate on word-order phenomena (V3, resumption).

**Most informative next experiment:** an **equal-budget LoRA/fine-tuning baseline** on the same 104 pairs, the same sealed splits and the
same random-control logic. That directly answers whether the small effect reflects the edit family or the 104-pair budget itself.

*Artifacts:* `results/tables.md`, `results/analysis.json`, `reports/baseline/BASELINE.md`, the per-session reasoning and token reports in
`reports/adjustments/`, `results/session_summary.md`, `manifests/experiment_lock.json`, and `AI_USAGE.md`.
