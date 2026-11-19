# Weight-space programming for Swedish: a tiered, controlled test

Vadym Kulish · Candidate assignment for SE.LLMA · Full report: [`report.pdf`](report.pdf)

**Result.** Agent-designed MLP edits to Gemma 4 12B IT shifted held-out likelihood margins toward grammatical Swedish, but held-out Swe-BLiMP accuracy rose only +0.6 to +1.3 pp. Every interval crossed zero and no edit clearly beat a matched random edit. Other languages were unaffected. The hypothesis is not supported. Changing the scoring prompt of the unchanged model moved the same benchmark by +16.9 pp.

## The article

*Sparks of Weight-Space Programming* (Takehi et al., 2026) shows that a frontier LLM can read small networks from their weights and edit models to a measurable benefit. In its strongest result, removing one looping direction from 64 output columns in two MLP layers of Qwen3.5-0.8B raised MATH-500 from 7.9 to 48.1. The article does not establish a methodology, as the authors acknowledge. For language models the evidence relies on behavioural feedback as well as weights: the sycophant was found only with attention patterns, and the Qwen edit took up to 50 batches of long rollouts. The largest edited model has 0.8B parameters. The one attempt to raise a capability without an obvious obstacle (Gemma 3 270M) gave no gain, and there are no random-edit controls. The idea still deserves wider testing: other patient models, other editing agents, purpose-built harnesses, evaluation-driven improvement, inference optimisation, and diagnostics during training. This study tests evaluation-driven improvement for Swedish.

## Question and hypothesis

A web survey of Swedish and English discussions and of Nordic research found that models often write understandable Swedish that sounds translated, with unnatural word choices. Other recurring problems were register, unreliable grammar explanations, language mixing and weak local knowledge. Grammatical, idiomatic and language-adherence quality behave as separate outcomes. Benchmarks had to be scorable without manual assessment or paid LLM judges, which favours minimal pairs.

**H1.** An activation-informed weight edit, built and selected without gradient training from a small set of Swedish grammatical contrasts, improves held-out Swe-BLiMP accuracy over the unchanged model and by more than a matched random edit.

The pre-registered support threshold is at least +2 pp macro accuracy with a 95% interval above zero, plus a positive edit-minus-random difference. A gain on dev only, or in margin only, does not count.

## Setup

- **Model.** `google/gemma-4-12B-it` @ `43c9d014`, float32 weights and computation. A pilot with Codex used bfloat16-rounded edits and raw-text scoring; both proved unreliable.
- **Benchmarks.**
  - Swe-BLiMP: 26 phenomena of Swedish grammatical minimal pairs; primary.
  - Swedish Idioms: idiom minimal pairs.
  - ScaLA sv/da/nb/en: acceptability judgement, 12-shot.
  - Belebele: reading comprehension in sv, da, nb, fi, is, en, zh; regression guard with a −3 pp tolerance.
- **Scoring.** A pair is correct when the grammatical sentence has the higher mean token log-probability. Each sentence is scored as the model's reply to *"Skriv en mening på svenska."* with thinking disabled. Scored as raw text, the instruction-tuned model is near chance (52.2%); with the prompt it reaches 69.1%.
- **Budget tiers.** Every benchmark reserves nested dev sets; Swe-BLiMP has 26, 52 and 104 pairs. Each tier restarts from the original weights, mirroring fine-tuning studies with growing data. Test scores of edited models stayed sealed in code until all three tiers were locked.
- **Editing agent.** Claude Opus 5.5 (high effort) in Claude Code read activations, weights and dev results. Per tier it could evaluate at most 4 rounds of 4 candidates, using low-rank MLP edits and no gradients. A pre-registered rule locked the candidate with the largest dev Swe-BLiMP gain, provided pooled Swedish and non-Swedish dev changes stayed above −2 pp.
- **Edits found.** Every locked edit rescales `down_proj` columns in layers 40–47 (44–47 in tier 1). Per layer, the 32 neurons whose direct effect on the output most favours grammatical sentences are scaled up and the 32 that most favour ungrammatical ones are scaled down. The neurons are selected with a first-order readout estimate on labelled dev pairs.
- **Controls and statistics.** Each locked edit has a random edit with the same matrices, rank and singular values (seed 43). Intervals are 95% paired cluster-bootstrap intervals with 10,000 draws.

## Results (held-out test; change vs unchanged model, pp unless stated)

| | Tier 1 (26 pairs) | Tier 2 (52) | Tier 3 (104, primary) |
|---|---|---|---|
| Dev Swe-BLiMP gain at lock | +23.1 | +17.3 | +11.5 |
| **Swe-BLiMP accuracy** (base 69.08%) | +1.30 [−0.04, +2.58] | +1.29 [−0.15, +2.73] | **+0.61 [−0.42, +1.65]** |
| Edit − random, accuracy | +0.62 [−0.69, +1.89] | +0.97 [−0.48, +2.37] | +0.32 [−0.70, +1.37] |
| Random − unchanged, accuracy | +0.68 [+0.27, +1.13] | +0.32 [−0.13, +0.81] | +0.29 [−0.07, +0.67] |
| Swe-BLiMP margin (nats/token) | +0.04 [+0.02, +0.06] | +0.16 [+0.13, +0.18] | +0.11 [+0.09, +0.12] |
| Idioms accuracy (base 73.5%) | −0.47 | −1.16 | +0.23 |
| ScaLA sv / da / nb / en accuracy | +0.4 / 0 / +0.4 / +0.4 | −0.4 / 0 / +0.8 / 0 | −0.4 / +0.4 / +0.4 / +0.8 |
| Belebele, worst of 7 languages | −0.5 | −0.5 | 0.0 |

- The edits move margins reliably while random edits do not, but few pairs cross the decision boundary.
- The tier-3 edit corrected 91 test pairs and broke 76. Gains concentrate in word-order phenomena with weak baselines (resumption, verb placement after a fronted constituent); losses appear in V3, V2 in relative clauses and argument structure.
- Dev gains shrank 10–20× on test. More labelled data did not help: 26 pairs gave +1.3 pp and 104 pairs +0.6 pp.
- All 21 Belebele language-by-edit intervals lie above −3 pp.
- Agent output tokens were 31.6k, 11.8k and 8.7k per tier. All tiers ran in one session, so later tiers reused tools and knowledge from tier 1.

## Interpretation

- **What is supported.** A gradient-free procedure using activations, weights and labels found late-layer neurons that move held-out preferences toward grammatical Swedish more than random edits, at no measurable cost elsewhere.
- **What is not.** A useful accuracy gain, or an accuracy advantage over random edits. Selection on small dev sets dominated: the lock rule picks the largest in-sample gain, and the agent's own cross-validation forecast +5.8 pp where the test gave +0.6 pp.
- **The agent's round reports.** They show correct reasoning about the architecture: normalised MLP outputs, a near-zero residual scale at layer 11, and much larger direct effects in the last layers. After three discarded ideas, the agent recognised its own overfitting and predicted a small test effect. Its search stayed narrow: two of four rounds per tier and one edit family.
- **The prompt as an alternative.** For an instruction-tuned model, how the model is asked mattered far more than these weight edits.

## Recommendation

With a general coding agent and a small budget, agent-driven weight editing is not a practical route to better Swedish grammar in a model of this class. The method is cheap, controllable and inspectable, so further work is justified in three directions:
1. Apply the same sealed, tiered and random-controlled method to models of other sizes, types and architecture families, to learn which models are more susceptible to weight-space programming.
2. Build and evaluate an agent harness for weight editing, with attribution, cross-validated selection and controls, so that budgets and edit extent are better informed.
3. Study weight-space diagnostics during pre- and post-training.

The most informative next experiment in this setup is an equal-budget LoRA baseline on the same 104 pairs and sealed splits.

## Limitations

- **Time and tooling.** About four hours of the author's time, mostly on design and evaluation. The switch from Codex to Claude was forced by subscription limits.
- **Shared session.** One agent session ran all tiers, so the tiers are not independent replications.
- **What the benchmark measures.** Swe-BLiMP measures likelihood preference on templated sentences, and the primary score depends on one prompt.
- **Exposure.** The agent had seen the pilot's and the baseline's test scores before editing.
- **Uncertainty.** Intervals are conditional on the chosen edits, split and one random seed.

Details: [`RESULTS.md`](RESULTS.md), [`reports/adjustments/`](reports/adjustments/) (the agent's round reports), [`AI_USAGE.md`](AI_USAGE.md).
