# AI usage

## Contributions

- **Author (Vadym Kulish).** Article analysis, hypothesis, research questions and experiment design. Key decisions:
  - the patient model and numerics;
  - the budget tiers and the restart-per-tier rule;
  - sealing of edited test scores until all tiers were locked;
  - token accounting in the agent's reports;
  - the primary scoring prompt, chosen from dev-only diagnostics.
- **ChatGPT (GPT-6 Astra) and Claude (Opus 5.5).** Helped find and summarise information on the web: discussions and research on how LLMs handle Swedish, candidate benchmarks, and the article.
- **Codex (GPT-6 Astra).** Implemented and ran a pilot experiment in a separate repository. It used fixed-grid rank-one edits, bfloat16-rounded weights and raw-text scoring, and its result was negative. Its downloaded model, data and environment were reused here after checksum verification, together with some prompt and loader designs. All code in this repository was rewritten. The pilot was stopped when the Codex subscription quota ran out.
- **Claude Opus 5.5 (`claude-opus-5-5`, high effort) in Claude Code.** Main run, after the author defined the repository structure and targets:
  - wrote all code and ran every GPU job;
  - acted as the editing agent in the adjustment rounds (the agent itself is the editing procedure);
  - ran the analysis and assisted with writing the reports.
  
  Its reasoning for every round is in `reports/adjustments/tier{N}_round{R}.md`.
- **No AI judge evaluated any output.** All metrics are likelihood- or exact-match-based.

## Exposure affecting how "held out" should be read

- Before tier 1 the agent had read the pilot's README, including its test results. The pilot used the same split seed and test size.
- Baseline test results were computed and viewed before any edit, at the author's request.
- All three tiers ran in one continuous agent session. Knowledge and tools from smaller tiers carried to larger ones, so the tiers are not independent replications.
- No edited-model test score was computed before `manifests/experiment_lock.json` was written; this is enforced in code.

## Measured agent effort in the adjustment rounds

From `results/session_summary.md`, using the per-call usage in the session transcript:

| tier | rounds | agent API calls | agent output tokens (thinking) |
|---|---|---|---|
| 1 | 2 | 17 | 31,552 (11,568) |
| 2 | 2 | 15 | 11,833 (2,587) |
| 3 | 2 | 15 | 8,676 (2,519) |

- **Cached context.** Each call re-read the accumulated context, about 5M cached input tokens per tier.
- **Why tier 1 cost more.** Tier 1 included writing the inspection tools, which later tiers reused.
- **Outside these counts.** Implementation, the baseline and the sealed test evaluation are not included.
- **Unattended GPU time.** Baseline about 48 minutes, adjustment rounds about 2.3 hours, sealed test about 4.2 hours.
