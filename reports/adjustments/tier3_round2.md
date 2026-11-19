# Tier 3 · Round 2 (final round of tier 3; lock)

## Observations carried in
Round 1's leader was `t3r1_dla40-47_K64_s0.25` (+11.5 pp Swe-BLiMP dev). Within-phenomenon CV supports L40–47 K=64 as the most transferable
configuration, while held-out Idioms tend to fall. Ranking neurons by mean effect is sensitive to a few pairs with large effects.

## Within-dev check
**Within-phenomenon CV with t-statistic ranking** (`sessions/dla_cv.py 3 within t`, 8 configurations). Ranking neurons by the per-neuron t-statistic
(mean / standard error over pairs), rather than the mean, gives more **consistent** but **smaller** held-out effects. 46–54 of 76 held-out pairs
improve their margin (38–45 with mean ranking). Held-out Idioms are unchanged in all 8 configurations (mean ranking: mostly negative). Held-out
margin +0.007 to +0.022 (mean ranking: up to +0.103) and Swe-BLiMP accuracy 0 to +2.9 pp.

## Targeted weights and why
| candidate | change | why |
|---|---|---|
| `t3r2_dla40-47_K64_s0.5_t` | L40–47; 32 highest-t ×1.5, 32 lowest-t ×0.5 | conservative selection, best t-ranked CV configuration |
| `t3r2_dla44-47_K64_s0.5_t` | L44–47; same | smaller footprint |

## Dev results (tier 3; Δ vs unchanged)
| candidate | Swe-BLiMP Δpp | Swe-BLiMP Δmargin | Swedish secondary Δpp | non-Swedish Δpp | eligible |
|---|---|---|---|---|---|
| t3r2_dla40-47_K64_s0.5_t | +3.85 | +0.1189 | +0.83 | −0.30 | yes |
| t3r2_dla44-47_K64_s0.5_t | +3.85 | +0.1042 | +0.83 | −0.30 | yes |

Both gain 1 idiom pair and lose one off-target item (Belebele-nob, or one ScaLA-nb row). The t-ranked edits also have larger norms
(3.2% per matrix vs 1.4% for the leader), because high-t neurons have larger write vectors.

## Decision: lock tier 3
Under the pre-registered rule, the largest eligible Swe-BLiMP dev gain over all 6 tier-3 candidates is **`t3r1_dla40-47_K64_s0.25` (+11.54 pp)**.
Locked: layers 40–47 down_proj; per layer, the 32 neurons with the highest mean direct-effect score ×1.25 and the 32 lowest ×0.75, selected on all
152 tier-3 pairs; rank 64 per matrix; 1,966,080 changed entries; relative Frobenius norm 1.2–1.5% per matrix.
Within-dev evidence predicts a **small** test effect for this edit. Within-phenomenon CV suggests +0.07 margin and about +5 pp Swe-BLiMP on
held-out pairs, with Idioms possibly slightly negative. The in-sample +11.5 pp should be read as optimistic.

**Matched random control on tier-3 dev (reporting only):** `artifacts/tier3/t3r1_dla40-47_K64_s0.25__random43.pt`, with the same 8 matrices, rank 64 and identical singular values. Realised Frobenius norms are 3.030, 2.960, 1.097, 1.086, 3.131, 2.762, 2.723, 2.513, against edit intended 3.030, 2.960, 1.097, 1.086, 3.131, 2.762, 2.723, 2.513. Dev: Swe-BLiMP +2.88 pp, margin -0.0008; Swedish secondary +0.00 pp, non-Swedish +0.00 pp.

## Token accounting
Window 2026-11-19T14:30:11.029957Z → 2026-11-19T14:53:47.270076+00:00 (UTC). Includes the call that opened the session (it writes the start marker); a boundary call may be counted in both adjacent sessions; calls after the end mark (writing this block) are excluded.

**Agent (Claude Opus 5.5) tokens, reasoning + executing the edit,** summed over the session transcript's per-call usage records (`scripts/agent_tokens.py`):

| API calls | output tokens | of which thinking | output non-thinking | input uncached | cache-write input | cache-read input |
|---|---|---|---|---|---|---|
| 5 | 3,470 | 1,017 | 2,453 | 10 | 7,472 | 1,815,441 |

**Gemma model-forward tokens** (exact counts from the scorer; non-padding input tokens plus generated tokens):

| stage | forward calls | sequences | input tokens | generated tokens | GPU forward s |
|---|---|---|---|---|---|
| candidate construction (build) | 0 | 0 | 0 | 0 | 0 |
| dev evaluation of candidates | 1120 | 2032 | 188,164 | 512 | 687 |
| inspection - CV within phenomena, t-ranking | 90 | 2736 | 96,183 | 0 | 283 |
| lock - random control dev evaluation | 560 | 1016 | 94,082 | 256 | 351 |
| **total** | | | | 379,197 (in+gen) | |
