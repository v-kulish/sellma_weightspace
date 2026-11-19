# Tier 1 · Round 1

**Dev budget (tier 1):** 26 Swe-BLiMP pairs (1 per phenomenon), 12 Idioms pairs, 8 ScaLA rows per language, 10 Belebele questions
(5 passages) per language. The model starts from the original checkpoint. Inputs: unchanged-model activations and weights, plus dev labels.
Prior knowledge (disclosed): baseline test results; the earlier `~/sellma` run's failed fixed-grid contrast edits.

## Observations
1. **Architecture (verified in code and weights).** Each decoder MLP is `down_proj(gelu(gate_proj x) * up_proj x)`, followed by
   `post_feedforward_layernorm` (RMS-normalised) before the residual add. Every layer then multiplies the whole residual by a fixed
   `layer_scalar`: 0.053 at layer 0, **0.005 at layer 11**, 0.6–0.9 in layers 12–46, 0.048 at layer 47. Consequences:
   uniformly scaling a down_proj does nothing, and only changes in the *direction mix* of the MLP output survive the norm.
   Contributions written before layer 12 are strongly down-weighted relative to later writes.
2. **Grammar contrasts (inspect_tier 1).** The mean good-minus-bad down_proj output contrast is 1.5–3.3% of the output norm in layers 12–30.
   Per-pair contrasts agree only weakly with their mean (mean cosine 0.15–0.25; 82–97% of pairs have positive cosine). The most
   consistent layers are 19–27.
3. **Direct-effect neuron attribution.** For each neuron in layers 12–47, I computed the first-order direct effect of scaling it on
   each pair's mean-log-prob margin. The calculation linearises post-FFN norm → layer_scalar products → final norm → tied unembedding → softcap → softmax
   at observed activations. It uses one forward pass per sentence and **no backpropagation through later layers**, so it covers the direct
   path only. It is an analytic readout derivative, disclosed as such.
   Direct effects grow sharply toward the top: summed positive effect ≈0.0003 at layer 12, 0.03 at layer 30, 0.5–2.9 at layers 40–47.
   Individual neurons are **not** distinguishable from noise with 38 pairs: |t|>3 counts (≈20–35 per layer) are at or below the ≈75
   per 15,360 expected by chance.
4. **Language direction.** I took the Swedish-minus-Danish/Norwegian difference of mean down_proj outputs on the 5 parallel Belebele dev
   passages. It is ≈4–9% of the Swedish output norm in layers 12–39, jumping to 24–54% in layers 40–44 (peak at layer 42). Logit-lens
   readouts of mid-layer directions are uninterpretable token soup.

## Targeted weights and why
| candidate | matrices | change | rationale |
|---|---|---|---|
| `t1r1_lang42_a0.5` | L42 down_proj | W ← W + 0.5 d dᵀW, d = sv−(da,nb) direction | Late layers carry the output-language signal. Amplifying it might reduce Scandinavian interference (the model's own continuation of "Hunden" leaned Norwegian, " lø"). |
| `t1r1_lang40-42-44_a0.3` | L40, L42, L44 down_proj | same, α=0.3 each | spread version of the above |
| `t1r1_dla44-47_k32_s0.5` | L44–47 down_proj | per layer, the 32 neurons with the highest mean direct-effect score have their write column ×1.5; the 32 lowest ×0.5 (rank 64 per matrix) | The direct path is largest in these layers. This is an aggregate of many individually noisy neurons. |
| `t1r1_gram24_a0.5` | L24 down_proj | W ← W + 0.5 g gᵀW, g = mean good−bad output contrast | the most consistent contrast layer (a fixed-grid-style reference edit, chat context) |

Exact construction: `sessions/tier1_inspect_dla.py`, `sessions/tier1_round1.py`. Factors: `artifacts/tier1/*.pt`. All matrices are
float32. Realised relative Frobenius norms: 0.8%, 0.5–0.6% each, 2.5–2.9% each, 2.0%. Projection edits change ≈100% of entries of
their matrix; neuron edits change 64 columns (245,760 entries) per matrix.

## Dev results (tier 1; Δ vs unchanged, pp)
| candidate | Swe-BLiMP Δpp | Swe-BLiMP Δmargin | Swedish secondary Δpp | non-Swedish Δpp | eligible |
|---|---|---|---|---|---|
| t1r1_lang42_a0.5 | +0.00 | +0.0098 | +0.00 | +0.00 | no |
| t1r1_lang40-42-44_a0.3 | +0.00 | −0.0017 | +0.00 | +0.00 | no |
| t1r1_dla44-47_k32_s0.5 | **+23.08** | **+0.4219** | +0.00 | +0.00 | yes |
| t1r1_gram24_a0.5 | +3.85 | −0.0106 | +0.00 | −1.19 | yes |

Unchanged tier-1 dev: Swe-BLiMP 53.8, Idioms 91.7, ScaLA sv/da/nb/en 100/87.5/87.5/62.5, Belebele swe/dan/nob/fin/isl/eng/zho
90/90/100/90/80/90/90. The only other change was gram24 losing one Belebele-nob question (−10 pp on n=10).

## Decision
- The language-direction edits have no measurable effect at these strengths; drop them.
- The +23 pp for `dla44-47` is **in-sample**: its neurons were chosen on these same 26+12 pairs. It is the leading candidate,
  but very likely overfit. Its idiom pairs (also used for selection) were already at 11/12 and did not move.
- Next round: estimate out-of-sample behaviour **within dev**. Choose neurons on half of the phenomena, measure the margin on the other
  half (2-fold cross-validation over phenomena). Use that to pick K/strength/layers, then build final candidates on all tier-1 pairs.
- No lock yet.

## Token accounting
Window 2026-11-19T12:04:05.786844Z → 2026-11-19T12:16:35.521629+00:00 (UTC). Includes the call that opened the session (it writes the start marker); a boundary call may be counted in both adjacent sessions; calls after the end mark (writing this block) are excluded.

**Agent (Claude Opus 5.5) tokens, reasoning + executing the edit,** summed over the session transcript's per-call usage records (`scripts/agent_tokens.py`):

| API calls | output tokens | of which thinking | output non-thinking | input uncached | cache-write input | cache-read input |
|---|---|---|---|---|---|---|
| 11 | 20,466 | 9,353 | 11,113 | 22 | 35,517 | 3,130,975 |

**Gemma model-forward tokens** (exact counts from the scorer; non-padding input tokens plus generated tokens):

| stage | forward calls | sequences | input tokens | generated tokens | GPU forward s |
|---|---|---|---|---|---|
| candidate construction (build) | 0 | 0 | 0 | 0 | 0 |
| dev evaluation of candidates | 576 | 1016 | 97,248 | 256 | 343 |
| inspection - grammar+language activations (inspect_tier.py) | 17 | 111 | 7,573 | 0 | 23 |
| inspection - direct-effect attribution | 76 | 76 | 2,666 | 0 | 0 |
| **total** | | | | 107,743 (in+gen) | |
