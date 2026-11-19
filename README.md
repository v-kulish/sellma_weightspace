# Agent-driven weight edits for Swedish grammatical preferences (Gemma 4 12B IT)

Can a coding agent that reads activations and weights (no gradient training) find small edits to Gemma 4 12B IT's MLP weights
that improve Swedish grammatical preferences? The constraints: limited labelled feedback (three nested dev budgets), no regression
on other languages, and the edit must beat a matched random edit on held-out data.

**Result.** Agent-designed late-layer MLP neuron edits (a fixed rule picked one per tier) raise held-out Swe-BLiMP likelihood margins
by +0.04 to +0.16 nats/token, far beyond spectrum-matched random edits. But macro accuracy moves only +0.61 pp [−0.42, +1.65] for the
primary tier-3 edit (unchanged 69.08%), and +1.3 pp for tiers 1–2; every interval crosses zero, and none clearly beats its random control.
There is no transfer to Idioms or ScaLA, and all seven Belebele languages are preserved within −3 pp. In-sample dev gains (+11.5 to +23 pp) were
mostly selection noise. H1 is not supported at the pre-registered threshold.

**Details:** see `reports/research_note.md` (PDF: `reports/research_note.pdf`), `results/tables.md` and `reports/baseline/BASELINE.md`.
The agent's reasoning for every adjustment session, including token accounting, is in `reports/adjustments/`.

## Design in one paragraph
Model `google/gemma-4-12B-it` @ `43c9d014b231474fa7212dbb48a762566c2b255d`, float32 weights and compute, SDPA attention, no TF32,
eager PyTorch (`src/model.py`). Four benchmarks, each with nested **adjustment tiers** held out from its test set (split seed 42,
immutable manifests in `manifests/splits/`):

| benchmark | tiers 1/2/3 (dev) | test | metric |
|---|---|---|---|
| Swe-BLiMP v1 | 26/52/104 pairs (1/2/4 per phenomenon) | 2,458 pairs (38 related pairs quarantined) | macro pair accuracy |
| Swedish Idioms | 12/24/48 pairs | 430 pairs | pair accuracy |
| ScaLA sv/da/nb/en | 8/16/32 rows per language (label-balanced) | 256 rows per language | accuracy, MCC |
| Belebele ×7 languages | 10/20/40 questions (whole passages, aligned IDs) | 201 questions / 113 passages | accuracy |

Minimal pairs are scored as mean token log-probability of each sentence, as the model's reply to the fixed user turn
`Skriv en mening på svenska.` (thinking disabled). Raw BOS-only scoring is reported as a secondary protocol: this IT checkpoint's raw-text
likelihoods are degenerate (DESIGN_DECISIONS #11). Each tier restarts from the original weights. The agent (Claude Opus 5.5) inspects
dev behaviour, activations and weights, evaluates at most 4 rounds × 4 candidates on that tier's full dev suite, and locks by a fixed rule
(`reports/adjustments/PROTOCOL.md`). Each locked edit gets a matched random control: same matrices and rank, identical singular values,
random singular vectors (seed 43). Test scores of edited models were sealed in code until all three tiers were locked
(`manifests/experiment_lock.json`).

## Reproduce
Hardware used: NVIDIA DGX Spark (GB10, ARM64, 121 GiB unified memory), driver 580.173.02, CUDA 13.0. Peak GPU allocation is about 56 GB.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.lock   # torch 2.14.0+cu130, transformers 5.23.0
bash scripts/fetch_data.sh                                            # pinned data; then download the model revision shown
.venv/bin/python scripts/record_sources.py && git diff manifests/sources.json   # checksums must be unchanged
.venv/bin/python -m src.data                                          # refuses to overwrite the immutable split manifests if they differ
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/validate_gpu.py                              # batch/single, zero edit, hook equivalence, restoration, prefix cache
.venv/bin/python -m src.evaluate baseline                             # ~48 min
```

Adjustment sessions (dev only; their order and arguments are recorded in each `reports/adjustments/tier*_round*.md`):

```bash
.venv/bin/python scripts/inspect_tier.py 1 && .venv/bin/python sessions/tier1_inspect_dla.py 1
.venv/bin/python sessions/tier1_round1.py
.venv/bin/python sessions/dla_cv.py 1 && .venv/bin/python sessions/dla_cv_random.py 1 && .venv/bin/python sessions/tier1_round2.py
.venv/bin/python scripts/lock_tier.py 1 t1r1_dla44-47_k32_s0.5
# tiers 2 and 3: sessions/dla_round.py TIER ROUND name=layers:K:s[:subset[:stat]] ...; scripts/lock_tier.py TIER NAME
.venv/bin/python scripts/lock.py                                      # experiment lock: unseals test evaluation
```

The locked edits are stored as compact specs in `artifacts/specs/` (neuron indices and scales; the random-control seed). Rebuild the low-rank
factors from the original checkpoint, checking each identity against the lock:

```bash
.venv/bin/python scripts/edit_specs.py rebuild
.venv/bin/python -m src.evaluate test          # ~5 h for 6 edited conditions
.venv/bin/python -m src.analyze && .venv/bin/python -m src.figures && .venv/bin/python -m src.report
```

`src.analyze`, `src.figures` and `src.report` need only the saved item records in `results/` (CPU). Benchmark text is not committed:
records keep ids, hashes and scores.

## Layout
`src/` library (data, model, scoring, edits, evaluate, session tools, analysis, figures, PDF). `sessions/` the agent's per-round scripts.
`scripts/` setup, validation, locking, token accounting. `manifests/` sources, splits, environment, experiment lock. `results/` dev and test
item records, rounds, tier locks, analysis. `reports/` baseline report, adjustment reports, research note. See `AI_USAGE.md`, `THIRD_PARTY.md`,
`DESIGN_DECISIONS.md` and the original brief `Swedish_Weight_Editing_Codex_Brief.md`.

## Publishing
No remote is configured. After review, create an empty GitHub repository and run:

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
```

```bash
git push -u origin main
```
