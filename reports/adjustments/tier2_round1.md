# Tier 2 · Round 1

**Dev budget (tier 2):** 52 Swe-BLiMP pairs (2 per phenomenon), 24 Idioms pairs, 16 ScaLA rows per language, 20 Belebele questions
(10 passages) per language. Restarted from the original checkpoint.
**Carried from tier 1 (disclosed):** direct-effect neuron scaling in the top layers transfers across phenomena within dev and beats
random neurons. Neuron ablation (×0) costs off-target accuracy. Language-direction projection scaling had no effect.

## Observations
1. **Re-inspection on tier-2 pairs** (`scripts/inspect_tier.py 2`, `sessions/tier1_inspect_dla.py 2`). The picture matches tier 1.
   Direct effects concentrate in layers 40–47 (summed positive effect 0.6 at L43, 2.1 at L47). Single-neuron |t| > 3 counts are ≈20–50 per
   layer, still near chance for 76 pairs × 15,360 neurons.
2. **2-fold CV over phenomena on tier-2 pairs** (`sessions/dla_cv.py 2`; 38 held-out pairs per fold). All 27 configurations raise the held-out
   margin (+0.025 to +0.175 nats/token) and Swe-BLiMP accuracy (+0 to +13.5 pp). But held-out **Idioms halves (12 pairs) fall in
   22/27 configurations** (−4.2 to −16.7 pp, i.e. 1–4 pairs). Neurons selected mostly on grammar pairs do not help, and may hurt,
   idiomatic lexical preference.
   Strongest margins: L40–47 with K=64/s=1.0 (+0.175, but ablation) and L40–47 K=32/s=1.0 (+0.131). The best non-ablating option was L40–47 K=64/s=0.5 (+0.094).

## Targeted weights and why
| candidate | change (down_proj columns, neurons selected on all 76 tier-2 pairs) | why |
|---|---|---|
| `t2r1_dla44-47_K64_s0.5` | L44–47; per layer 32 top ×1.5, 32 bottom ×0.5 | the tier-1 lock, re-selected on tier-2 data |
| `t2r1_dla40-47_K64_s0.5` | L40–47; same | best non-ablating CV configuration |
| `t2r1_dla40-47_K32_s0.5` | L40–47; 16 top ×1.5, 16 bottom ×0.5 | smaller edit |
| `t2r1_dla36-46e_K64_s0.5` | L36, 38, …, 46; 32 + 32 | reaches into lower layers |

Construction: `sessions/dla_round.py 2 1 …`. Artifacts: `artifacts/tier2/`. Relative Frobenius norm 1.7–3.0% per matrix.

## Dev results (tier 2; Δ vs unchanged)
| candidate | Swe-BLiMP Δpp | Swe-BLiMP Δmargin | Swedish secondary Δpp | non-Swedish Δpp | eligible |
|---|---|---|---|---|---|
| t2r1_dla44-47_K64_s0.5 | +7.69 | +0.3974 | +0.00 | +0.00 | yes |
| t2r1_dla40-47_K64_s0.5 | **+17.31** | **+0.5436** | +0.00 | −0.60 | yes |
| t2r1_dla40-47_K32_s0.5 | +7.69 | +0.3100 | +0.00 | −1.19 | yes |
| t2r1_dla36-46e_K64_s0.5 | +11.54 | +0.2059 | −3.33 | −0.60 | no |

Unchanged tier-2 dev: Swe-BLiMP 61.5, Idioms 79.2, ScaLA sv/da/nb/en 93.8/87.5/81.2/75.0, Belebele swe/dan/nob/fin/isl/eng/zho
90/90/90/95/90/90/90. Every off-target change is a single item: Belebele-nob/dan −5 pp = 1 of 20; Idioms −8.3 pp = 2 of 24.

## Decision
Leading candidate: `t2r1_dla40-47_K64_s0.5`. Round 2 will test (i) strength 0.75 on the same set, (ii) a wider layer range,
and (iii) selecting neurons on Swe-BLiMP pairs only, given the CV evidence that Idioms pairs contribute little to transferable selection.
No lock yet.

## Token accounting
Window 2026-11-19T12:33:58.759802Z → 2026-11-19T12:59:46.803846+00:00 (UTC). Includes the call that opened the session (it writes the start marker); a boundary call may be counted in both adjacent sessions; calls after the end mark (writing this block) are excluded.

**Agent (Claude Opus 5.5) tokens, reasoning + executing the edit,** summed over the session transcript's per-call usage records (`scripts/agent_tokens.py`):

| API calls | output tokens | of which thinking | output non-thinking | input uncached | cache-write input | cache-read input |
|---|---|---|---|---|---|---|
| 8 | 7,532 | 2,048 | 5,484 | 16 | 12,349 | 2,641,214 |

**Gemma model-forward tokens** (exact counts from the scorer; non-padding input tokens plus generated tokens):

| stage | forward calls | sequences | input tokens | generated tokens | GPU forward s |
|---|---|---|---|---|---|
| candidate construction (build) | 0 | 0 | 0 | 0 | 0 |
| dev evaluation of candidates | 1136 | 2032 | 190,460 | 512 | 677 |
| inspection - grammar+language activations | 26 | 222 | 16,180 | 0 | 46 |
| inspection - direct-effect attribution | 152 | 152 | 5,323 | 0 | 0 |
| inspection - 2-fold CV over phenomena | 168 | 4256 | 149,044 | 0 | 447 |
| **total** | | | | 361,519 (in+gen) | |
