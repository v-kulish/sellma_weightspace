# Adjustment-session protocol (pre-registered before tier 1, 2026-11-19)

This fixes the rules for the agent-driven edits (DESIGN_DECISIONS.md #5–#10) before any edit is tried.

## Information the agent may use
- Unchanged-model activations, weights and dev-tier outputs of **all** benchmarks for the current tier only
  (tier N may use the tier-N dev items, which include the tier-(N−1) items).
- Knowledge from earlier tiers (findings, failed ideas) carries over and is disclosed. Weights do not: every tier
  restarts from the original checkpoint.
- Never test items or test scores of any edited condition. Baseline test scores were seen before tier 1
  (user request #4a/#9); the agent had also read the earlier `~/sellma` run's README, including its test results
  (raw-BOS Swe-BLiMP ≈52% unchanged; its rank-one contrast edits at layers 11/23/35 did not transfer to test).

## Allowed edits
- Locations: `model.language_model.layers.<i>.mlp.{gate_proj, up_proj, down_proj}` (text decoder MLP only;
  no attention, embeddings, LM head, norms, vision/audio).
- Form: W ← W + U Vᵀ in float32, rank ≤ 64 per matrix, relative Frobenius norm ≤ 10% per matrix; may combine
  several matrices. Constructed without gradients from dev activations, dev labels and weights.
- Each candidate starts from the original weights and is evaluated on the **full tier-N dev suite**
  (Swe-BLiMP, Idioms, ScaLA sv/da/nb/en, Belebele ×7).

## Budget per tier
- At most **4 rounds** per tier, at most **4 candidates** evaluated per round (≤16 candidates per tier).
- One report per round: `reports/adjustments/tier{N}_round{R}.md` with observations, targeted weights and why, the
  exact change, dev results, decision, and token accounting (agent tokens from the session transcript; exact model
  forward tokens from the scorer).

## Lock rule (fixed)
Metrics are dev changes vs the unchanged model on the same tier:
- Primary: Swe-BLiMP macro accuracy (chat-context protocol, DESIGN_DECISIONS #11).
- Swedish secondary (item-weighted pooled accuracy): Idioms, ScaLA-sv, Belebele-sv.
- Non-Swedish guard (item-weighted pooled accuracy): ScaLA da/nb/en, Belebele dan/nob/fin/isl/eng/zho.

A candidate is **eligible** iff Swe-BLiMP Δ > 0 pp (strict), Swedish secondary Δ ≥ −2 pp and non-Swedish Δ ≥ −2 pp.
Among all candidates evaluated in the tier, lock the eligible one with the largest Swe-BLiMP Δ; ties → larger
Swedish-secondary Δ → larger Swe-BLiMP margin gain → smaller total relative norm. If none is eligible, lock
**no edit** (a valid outcome). The agent may stop before 4 rounds.

## Controls
- Each locked edit gets a matched random control: same matrices, same rank, identical singular values (hence
  identical intended Frobenius norm), output- and input-side singular vectors replaced by Haar-random orthonormal
  bases, seed 43 (`src/edits.py:matched_random`). The control is evaluated on dev for reporting only.

## Sealing
- No edited test evaluation until all three tiers are locked and `manifests/experiment_lock.json` is written
  (`src/evaluate.py:check_unsealed`).
