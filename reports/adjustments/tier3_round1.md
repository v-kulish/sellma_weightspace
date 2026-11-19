# Tier 3 · Round 1

**Dev budget (tier 3):** 104 Swe-BLiMP pairs (4 per phenomenon), 48 Idioms pairs, 32 ScaLA rows per language, 40 Belebele questions
(20 passages) per language. Restarted from the original checkpoint.
**Carried from tiers 1–2 (disclosed):** direct-effect neuron scaling in layers 40–47, K=64, s=0.5 was the tier-2 lock. Random neurons do
not move margins. Ablation costs off-target accuracy. Stronger or wider edits add margin, not accuracy.

## Observations
1. **Attribution shrinks with more data.** Mean direct-effect magnitudes per layer fall as pairs are added: summed positive effect at L47 is
   2.86 (tier 1) → 2.08 (tier 2) → 1.60 (tier 3). That is the √n signature of selection noise: part of the per-neuron mean was noise.
   Single-neuron |t| > 3 counts (≈15–40 per layer) remain at chance level.
2. **Cross-phenomenon 2-fold CV** (`sessions/dla_cv.py 3`; 76 held-out pairs per fold; same phenomenon split as before). Unlike tiers 1–2,
   held-out gains nearly vanish: margin change −0.069 to +0.040, Swe-BLiMP accuracy −4.8 to +3.8 pp, and 6 of 27 configurations are negative.
   The large cross-phenomenon transfer in tiers 1–2 (13 or 26 held-out pairs) was largely small-sample luck. **Direct-effect neurons
   do not transfer to unseen phenomena** to any useful degree.
3. **Random-neuron comparison on the same folds** (`sessions/dla_cv_random.py 3`): −0.001 to +0.004. Selected neurons are still slightly above
   random, but the margin of advantage is small.
4. **Within-phenomenon 2-fold CV** (`sessions/dla_cv.py 3 within`; pairs alternate inside each phenomenon, Idioms alternate). This mirrors the test
   split (unseen examples of represented phenomena) and is new in this tier. 24 of 27 configurations raise the held-out margin. The best is
   **L40–47 K=64 s=0.5: +0.103 margin, +4.8 pp Swe-BLiMP**, then L40–47 K=64 s=0.25 (+0.072, +5.8 pp). Held-out Idioms halves drop in most
   configurations (typically −2 to −10 pp = 1–5 of 48 pairs pooled over folds).

## Targeted weights and why
Neurons selected on all 152 tier-3 pairs (`sessions/dla_round.py 3 1 …`):

| candidate | change (down_proj columns) | within-CV support |
|---|---|---|
| `t3r1_dla40-47_K64_s0.5` | L40–47; 32 top ×1.5, 32 bottom ×0.5 | best margin (+0.103) |
| `t3r1_dla40-47_K64_s0.25` | L40–47; 32 top ×1.25, 32 bottom ×0.75 | best accuracy (+5.8 pp) |
| `t3r1_dla44-47_K64_s0.25` | L44–47; ×1.25 / ×0.75 | Idioms unchanged in CV |
| `t3r1_dla36-46e_K32_s0.5` | L36, 38, …, 46; 16 top ×1.5, 16 bottom ×0.5 | Idioms unchanged in CV |

## Dev results (tier 3; Δ vs unchanged)
| candidate | Swe-BLiMP Δpp | Swe-BLiMP Δmargin | Swedish secondary Δpp | non-Swedish Δpp | eligible |
|---|---|---|---|---|---|
| t3r1_dla40-47_K64_s0.5 | +10.58 | +0.3786 | +1.67 | −0.89 | yes |
| t3r1_dla40-47_K64_s0.25 | **+11.54** | +0.2102 | +0.83 | −0.89 | yes |
| t3r1_dla44-47_K64_s0.25 | +6.73 | +0.1595 | +0.00 | −0.30 | yes |
| t3r1_dla36-46e_K32_s0.5 | +2.88 | +0.0769 | +0.83 | −1.19 | yes |

Unchanged tier-3 dev: Swe-BLiMP 63.5, Idioms 70.8, ScaLA sv/da/nb/en 87.5/87.5/81.2/81.2, Belebele swe/dan/nob/fin/isl/eng/zho
87.5/87.5/85.0/95.0/87.5/90.0/90.0. Off-target changes: ScaLA-nb loses 1–2 of 32 rows in every candidate; Belebele-nob loses 1 of 40 in three; Belebele-zho loses 1 in one.
In sample, Idioms gain 1–2 of 48 pairs.

## Decision
Leader `t3r1_dla40-47_K64_s0.25`. The in-sample gain is about half of tiers 1–2, as expected if earlier in-sample gains were partly overfitting.
Round 2 will test a more conservative neuron-selection statistic (per-neuron t-statistic instead of mean) under within-phenomenon CV, and one
off-target guard. No lock yet.

## Token accounting
Window 2026-11-19T13:19:42.132005Z → 2026-11-19T14:30:01.173889+00:00 (UTC). Includes the call that opened the session (it writes the start marker); a boundary call may be counted in both adjacent sessions; calls after the end mark (writing this block) are excluded.

**Agent (Claude Opus 5.5) tokens, reasoning + executing the edit,** summed over the session transcript's per-call usage records (`scripts/agent_tokens.py`):

| API calls | output tokens | of which thinking | output non-thinking | input uncached | cache-write input | cache-read input |
|---|---|---|---|---|---|---|
| 10 | 5,206 | 1,502 | 3,704 | 20 | 13,209 | 3,511,182 |

**Gemma model-forward tokens** (exact counts from the scorer; non-padding input tokens plus generated tokens):

| stage | forward calls | sequences | input tokens | generated tokens | GPU forward s |
|---|---|---|---|---|---|
| candidate construction (build) | 0 | 0 | 0 | 0 | 0 |
| dev evaluation of candidates | 2240 | 4064 | 376,328 | 1,024 | 1408 |
| inspection - grammar+language activations | 45 | 444 | 33,106 | 0 | 90 |
| inspection - direct-effect attribution | 304 | 304 | 10,687 | 0 | 0 |
| inspection - CV across phenomena | 280 | 8512 | 299,236 | 0 | 861 |
| inspection - random-neuron CV | 130 | 3952 | 138,931 | 0 | 403 |
| inspection - CV within phenomena | 280 | 8512 | 299,236 | 0 | 865 |
| **total** | | | | 1,158,548 (in+gen) | |
