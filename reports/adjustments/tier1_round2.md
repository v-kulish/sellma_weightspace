# Tier 1 · Round 2 (final round of tier 1; lock)

## Observations carried in
Round 1 showed that direct-effect neuron scaling in layers 44–47 gives +23 pp **in sample** on the 26 tier-1 Swe-BLiMP pairs.
Individual neuron scores are not significant at this sample size (round 1 §3). The open question: does the *selection* generalise
beyond the pairs it was fitted on, or does it only fit noise?

## Within-dev checks (inspection only; tier-1 dev pairs; no full-suite evaluation)
1. **2-fold cross-validation over phenomena** (`sessions/dla_cv.py`). The 26 phenomena were split at random into 13 + 13 (seed 7)
   and the 12 idiom pairs alternately. Neurons were selected on one fold and the margin change measured on the other fold's pairs,
   for 27 configurations (layer set × K ∈ {8, 32, 64} × s ∈ {0.25, 0.5, 1.0}). This tests transfer to **unseen phenomena**, harder
   than the test split's unseen examples of seen phenomena. Held-out mean margin rose in 25/27 configurations (+0.002 to +0.130
   nats/token). Held-out Swe-BLiMP accuracy rose in 26/27, by +3.8 to +19.2 pp on 13 pairs per fold. Idioms held-out halves
   were mostly unchanged (6 pairs each, ±8.3 pp = one pair).
2. **Random-neuron comparison within the same folds** (`sessions/dla_cv_random.py`). The same layers, K, s and up/down split were
   applied to random neurons, 4 seeds × 3 configurations. Held-out margin changes were −0.010 to +0.010, against +0.066 to +0.130
   for the selected neurons at the same configurations. So the selection carries real signal across phenomena **on dev**.
   It is still possibly a broad likelihood-calibration shift (e.g. down-weighting rare or ill-formed continuations), not a grammar-specific mechanism.

## Targeted weights and why
The configurations best supported by CV were rebuilt from **all** tier-1 pairs (`sessions/tier1_round2.py`):

| candidate | change (down_proj columns) | CV held-out Δmargin |
|---|---|---|
| `t1r2_dla40-47_K32_s0.5` | layers 40–47; per layer 16 top neurons ×1.5, 16 bottom ×0.5 | +0.115 |
| `t1r2_dla40-47_K32_s1.0` | same, ×2.0 / ×0 (ablate) | +0.123 |
| `t1r2_dla40-47_K8_s1.0` | layers 40–47; 4 top ×2.0, 4 bottom ×0 | +0.080 |
| `t1r2_dla36-46e_K8_s1.0` | layers 36, 38, …, 46; 4 top ×2.0, 4 bottom ×0 | +0.130 |

Each matrix changes K columns (K×3,840 entries), rank K. Relative Frobenius norm is 1.7–4.3% per matrix.

## Dev results (tier 1; Δ vs unchanged)
| candidate | Swe-BLiMP Δpp | Swe-BLiMP Δmargin | Swedish secondary Δpp | non-Swedish Δpp | eligible |
|---|---|---|---|---|---|
| t1r2_dla40-47_K32_s0.5 | +15.38 | +0.3784 | +0.00 | −1.19 | yes |
| t1r2_dla40-47_K32_s1.0 | +15.38 | +0.4962 | −3.33 | −1.19 | no |
| t1r2_dla40-47_K8_s1.0 | +19.23 | +0.2211 | −3.33 | −1.19 | no |
| t1r2_dla36-46e_K8_s1.0 | +7.69 | +0.0752 | +0.00 | −2.38 | no |

Per-task losses: ablation variants lost ScaLA-sv and ScaLA-nb rows or an idiom pair. The 40–47 and 36–46 variants each lost one
Belebele-nob question; 36–46 also lost one Belebele-zho question. Ablating neurons (×0) costs more off-target than halving them.

## Decision: lock tier 1
Under the pre-registered rule, the eligible candidates are t1r1_dla44-47_k32_s0.5 (+23.08), t1r2_dla40-47_K32_s0.5 (+15.38)
and t1r1_gram24_a0.5 (+3.85). **Locked: `t1r1_dla44-47_k32_s0.5`**: layers 44–47 down_proj; per layer, 32 neurons ×1.5 and
32 neurons ×0.5; rank 64 per matrix; 983,040 changed entries in total.

Stopping after 2 of 4 allowed rounds: more full-suite rounds would mostly add selection on 26 pairs' noise. The lock rule ranks by
in-sample dev gain, which is optimistic by construction. The CV and random checks above are the within-dev evidence that the
family transfers across phenomena. The matched random control is built with `src/edits.py:matched_random` (seed 43) and evaluated on
dev for reporting.

**Knowledge carried to tier 2:** direct-effect neuron scaling in the top layers works on dev; random neurons do not; ablation (×0)
costs off-target accuracy; language-direction projection scaling did nothing at α ≤ 0.5.

**Matched random control on tier-1 dev (reporting only):** `artifacts/tier1/t1r1_dla44-47_k32_s0.5__random43.pt`. It has the same 4 matrices, rank 64 each and identical singular values. Intended Frobenius norms are 6.1057, 5.5727, 5.4952, 5.3001 vs edit 6.1057, 5.5727, 5.4952, 5.3001; realised 6.1057, 5.5727, 5.4952, 5.3001. Dev: Swe-BLiMP +7.69 pp, margin +0.0112; Swedish secondary +0.00 pp, non-Swedish +0.00 pp.

## Token accounting
Window 2026-11-19T12:16:57.418166Z → 2026-11-19T12:33:35.182983+00:00 (UTC). Includes the call that opened the session (it writes the start marker); a boundary call may be counted in both adjacent sessions; calls after the end mark (writing this block) are excluded.

**Agent (Claude Opus 5.5) tokens, reasoning + executing the edit,** summed over the session transcript's per-call usage records (`scripts/agent_tokens.py`):

| API calls | output tokens | of which thinking | output non-thinking | input uncached | cache-write input | cache-read input |
|---|---|---|---|---|---|---|
| 6 | 11,086 | 2,215 | 8,871 | 12 | 16,633 | 1,858,092 |

**Gemma model-forward tokens** (exact counts from the scorer; non-padding input tokens plus generated tokens):

| stage | forward calls | sequences | input tokens | generated tokens | GPU forward s |
|---|---|---|---|---|---|
| candidate construction (build) | 0 | 0 | 0 | 0 | 0 |
| dev evaluation of candidates | 580 | 1016 | 97,248 | 260 | 359 |
| inspection - 2-fold CV over phenomena | 112 | 2128 | 74,648 | 0 | 261 |
| inspection - random-neuron CV comparison | 52 | 988 | 34,658 | 0 | 121 |
| lock - random control dev evaluation | 144 | 254 | 24,312 | 64 | 91 |
| **total** | | | | 231,190 (in+gen) | |
