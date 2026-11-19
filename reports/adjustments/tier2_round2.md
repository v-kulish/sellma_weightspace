# Tier 2 · Round 2 (final round of tier 2; lock)

## Observations carried in
Round 1's leader was `t2r1_dla40-47_K64_s0.5` (+17.3 pp Swe-BLiMP dev, margin +0.54, one Belebele-nob question lost).
Tier-2 CV had shown that Idioms halves tend to drop out of sample.

## Within-dev check
**Random-neuron CV comparison on tier-2 folds** (`sessions/dla_cv_random.py 2`; 4 seeds × 3 configurations): held-out margin
changes were −0.005 to +0.007. The selected neurons at the same configurations gave +0.039 to +0.131 (tier-2 CV). This reconfirms on twice
the data that the direct-effect selection, not the perturbation itself, drives the dev gain.

## Targeted weights and why
| candidate | change | why |
|---|---|---|
| `t2r2_dla40-47_K64_s0.75` | L40–47; 32 top ×1.75, 32 bottom ×0.25 | stronger, still non-ablating |
| `t2r2_dla36-47_K64_s0.5` | L36–47 (12 matrices); 32 + 32 ×1.5/×0.5 | wider range |
| `t2r2_dla40-47_K64_s0.5_blimp` | as the leader, but neurons selected on the 52 Swe-BLiMP pairs only | tests whether Idioms pairs in the selection matter |
| `t2r2_dla32-47_K64_s0.5` | L32–47 (16 matrices) | widest range |

## Dev results (tier 2; Δ vs unchanged)
| candidate | Swe-BLiMP Δpp | Swe-BLiMP Δmargin | Swedish secondary Δpp | non-Swedish Δpp | eligible |
|---|---|---|---|---|---|
| t2r2_dla40-47_K64_s0.75 | +15.38 | +0.7501 | +1.67 | −0.60 | yes |
| t2r2_dla36-47_K64_s0.5 | +13.46 | +0.6659 | −1.67 | −0.60 | yes |
| t2r2_dla40-47_K64_s0.5_blimp | +15.38 | +0.5233 | −3.33 | −1.19 | no |
| t2r2_dla32-47_K64_s0.5 | +11.54 | +0.5723 | +0.00 | −1.19 | yes |

- A stronger or wider edit raises the margin further but does not raise accuracy beyond round 1's leader. Grammar accuracy is not the bottleneck
  of the margin: many pairs are far from the decision boundary.
- Dropping the Idioms pairs from selection cost 2 of 24 idiom pairs (in sample). Idioms pairs do shape the selected neurons.
- Off-target changes remain single Belebele questions (dan/nob, 1 of 20 each).

## Decision: lock tier 2
Under the pre-registered rule, eligible candidates ranked by Swe-BLiMP Δ: t2r1_dla40-47_K64_s0.5 (+17.31), then t2r2_dla40-47_K64_s0.75 and
t2r1/2 others (+15.38 or less). **Locked: `t2r1_dla40-47_K64_s0.5`**: layers 40–47 down_proj; per layer 32 neurons ×1.5 and
32 ×0.5, selected on all 76 tier-2 pairs; rank 64 per matrix; 1,966,080 changed entries (8 × 64 columns × 3,840).
Stopped after 2 of 4 rounds, for the same reason as tier 1.

**Knowledge carried to tier 3:** L40–47 K=64 s=0.5 is the best accuracy/off-target trade-off at tier 2. Stronger or wider variants
add margin, not accuracy. Keep the Idioms pairs in the selection.

**Matched random control on tier-2 dev (reporting only):** `artifacts/tier2/t2r1_dla40-47_K64_s0.5__random43.pt`, with the same 8 matrices, rank 64 and identical singular values. Realised Frobenius norms are 6.035, 5.928, 2.202, 2.134, 6.149, 5.614, 5.449, 5.276, against edit intended 6.035, 5.928, 2.202, 2.134, 6.149, 5.614, 5.449, 5.276. Dev: Swe-BLiMP +5.77 pp, margin +0.0075; Swedish secondary -1.67 pp, non-Swedish +0.00 pp.

## Token accounting
Window 2026-11-19T12:59:46.821777Z → 2026-11-19T13:19:34.371373+00:00 (UTC). Includes the call that opened the session (it writes the start marker); a boundary call may be counted in both adjacent sessions; calls after the end mark (writing this block) are excluded.

**Agent (Claude Opus 5.5) tokens, reasoning + executing the edit,** summed over the session transcript's per-call usage records (`scripts/agent_tokens.py`):

| API calls | output tokens | of which thinking | output non-thinking | input uncached | cache-write input | cache-read input |
|---|---|---|---|---|---|---|
| 7 | 4,301 | 539 | 3,762 | 14 | 8,536 | 2,380,394 |

**Gemma model-forward tokens** (exact counts from the scorer; non-padding input tokens plus generated tokens):

| stage | forward calls | sequences | input tokens | generated tokens | GPU forward s |
|---|---|---|---|---|---|
| candidate construction (build) | 0 | 0 | 0 | 0 | 0 |
| dev evaluation of candidates | 1136 | 2032 | 190,460 | 512 | 717 |
| inspection - random-neuron CV comparison | 78 | 1976 | 69,199 | 0 | 206 |
| lock - random control dev evaluation | 284 | 508 | 47,615 | 128 | 171 |
| **total** | | | | 307,914 (in+gen) | |
