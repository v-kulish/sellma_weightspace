# AI use disclosure

## Tools and roles
- **Planning assistant (earlier conversation, not this repository):** drafted `Swedish_Weight_Editing_Codex_Brief.md`.
  The user and the planning assistant then agreed `DESIGN_DECISIONS.md` (items 1–9).
- **Claude Code (Claude Opus 5.5, model id `claude-opus-5-5`), this repository:** implemented all code, ran every GPU job on the
  user's DGX Spark, made the edits in the agent-driven adjustment sessions, ran the analysis and wrote the reports,
  research note and this disclosure. In the adjustment sessions the *agent itself* is the editing procedure (DESIGN_DECISIONS #7).
  Its reasoning per session is in `reports/adjustments/tier{N}_round{R}.md`. Its API token usage per session is measured
  from the session transcript (`scripts/agent_tokens.py`; DESIGN_DECISIONS #10).
- **Earlier Codex run (`~/sellma`, not part of this repository):** an independent implementation of the original brief.
  This project reused its downloaded files (model weights and data, verified against pinned SHA-256 checksums) and its Python
  environment. It reused its design for some loaders and prompts (Belebele instructions, EuroEval prompt extraction, the strict
  ScaLA parser). All code here was rewritten.
- No AI judge evaluates any output; all metrics are likelihood or exact-match based.

## Human decisions (the user)
- Model, dtype, run order, tier sizes, tier restart rule, agent-driven edits, per-session reports (DESIGN_DECISIONS #1–#8).
- Test sealing until all tiers are locked, with the baseline test seen first (#9). Token accounting in reports (#10).
- Primary minimal-pair context `Skriv en mening på svenska.` (#11). The user chose this after seeing dev-only diagnostics
  for raw BOS, `Skriv en mening.` and the Swedish-cue variant.

## Exposure disclosures (affecting how "held-out" should be read)
- Before tier 1, the agent read the earlier `~/sellma` README/STATUS. These report that run's *test* results under the raw-BOS protocol:
  Swe-BLiMP ≈52% unchanged, and its fixed-grid rank-one contrast edits at layers 11/23/35 did not transfer.
  That run used the same split seed, and its Swe-BLiMP test set has the same size (2,458).
- Baseline test results on all benchmarks were computed and viewed before any adjustment (user request).
- Knowledge carries over across tiers (small → large): the tier-2 and tier-3 sessions were run by the same agent with memory of
  earlier tiers' dev findings. Tier-1 results are therefore not "clean" small-budget discoveries for later tiers, and later tiers
  are not independent replications.
- No edited-model test score was computed before `manifests/experiment_lock.json` was written (enforced in code).

## Measured agent effort in the adjustment sessions
From `results/session_summary.md`, via the transcript's per-call usage:

| tier | rounds | agent API calls | agent output tokens (thinking) |
|---|---|---|---|
| 1 | 2 | 17 | 31,552 (11,568) |
| 2 | 2 | 15 | 11,833 (2,587) |
| 3 | 2 | 15 | 8,676 (2,519) |

Cache-read input is about 5M tokens per tier: each call re-reads the whole conversation context. Tier 1 includes writing the reusable
inspection tools (direct-effect attribution, CV scripts), so later tiers mostly reused them. Implementation, the baseline and the sealed test
evaluation took place outside these windows. Unattended GPU time: baseline ≈ 48 min; adjustment sessions ≈ 2.3 h of forward passes; sealed test ≈ 4.2 h.
