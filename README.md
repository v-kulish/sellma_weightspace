# Weight-space programming for Swedish: agent-driven edits to Gemma 4 12B IT

Candidate assignment for SE.LLMA, by Vadym Kulish.

Can a coding agent that reads activations and weights edit Gemma 4 12B IT's MLP weights so that the model prefers grammatical Swedish? The agent may not use gradient training and gets a small labelled budget. The edit must also beat a matched random edit on held-out data and leave other languages intact.

**Result.** The agent's late-layer neuron edits raised held-out Swe-BLiMP likelihood margins by 0.04 to 0.16 nats per token, well beyond random edits of identical spectrum. Accuracy moved only +0.61 pp [−0.42, +1.65] for the primary tier (unchanged 69.08%) and +1.3 pp for the smaller tiers. Every interval crosses zero, and no edit clearly beats its random control. There was no transfer to idioms or acceptability judgement, and all seven Belebele languages stayed within the −3 pp tolerance. Dev gains of 11.5 to 23 pp were mostly selection noise. The hypothesis is not supported. Changing the scoring prompt of the unchanged model moved the same benchmark by +16.9 pp.

## Where to find things

| file | contents |
|---|---|
| [`REPORT.pdf`](REPORT.pdf) | full report (ACL format) |
| [`RESEARCH_NOTE.md`](RESEARCH_NOTE.md), [`RESEARCH_NOTE.pdf`](RESEARCH_NOTE.pdf) | two-page research note (PDF via `scripts/research_note_pdf.py`, headless Chrome) |
| [`RESULTS.md`](RESULTS.md) | configuration, model and data revisions, seeds, outcomes, examples, compute |
| [`reports/baseline/BASELINE.md`](reports/baseline/BASELINE.md) | unchanged-model results on all benchmarks |
| [`reports/adjustments/`](reports/adjustments/) | pre-registered editing protocol and the agent's reasoning report for each round, with token accounting |
| [`results/`](results/) | item-level dev and test records, candidate evaluations, tier locks, statistics (`tables.md`, `analysis.json`) |
| [`manifests/`](manifests/) | pinned sources and checksums, split manifests, environment, experiment lock |
| [`artifacts/specs/`](artifacts/specs/) | compact specifications of the locked edits and random controls |
| [`AI_USAGE.md`](AI_USAGE.md) | AI tools and their contributions |

## Method in brief

- **Model.** `google/gemma-4-12B-it` @ `43c9d014b231474fa7212dbb48a762566c2b255d`, float32 weights and computation, SDPA attention, TF32 off.
- **Benchmarks.** Swe-BLiMP (primary), Swedish Idioms, ScaLA sv/da/nb/en, and Belebele in seven languages. Each reserves nested dev tiers held out from its test set (split seed 42):

| benchmark | dev tiers 1/2/3 | test | metric |
|---|---|---|---|
| Swe-BLiMP v1 | 26/52/104 pairs (1/2/4 per phenomenon) | 2,458 pairs (38 related pairs quarantined) | macro pair accuracy |
| Swedish Idioms | 12/24/48 pairs | 430 pairs | pair accuracy |
| ScaLA sv/da/nb/en | 8/16/32 rows per language | 256 rows per language | accuracy, MCC |
| Belebele ×7 languages | 10/20/40 questions (whole passages) | 201 questions / 113 passages | accuracy |

- **Scoring.** Minimal pairs are scored by the mean token log-probability of each sentence, as the model's reply to the user turn `Skriv en mening på svenska.` with thinking disabled. Raw scoring after `<bos>` alone is reported as a secondary protocol.
- **Editing.** Each tier restarts from the original weights. Claude Opus 5.5 inspects dev behaviour, activations and weights, then evaluates at most 4 rounds × 4 candidate edits on the tier's full dev suite. It locks one edit by a fixed rule (`reports/adjustments/PROTOCOL.md`).
- **Controls and sealing.** Each locked edit gets a matched random edit with the same matrices, rank and singular values but random singular vectors (seed 43). Edited test scores were sealed in code until all three tiers were locked (`manifests/experiment_lock.json`).
- **Statistics.** 95% paired cluster-bootstrap intervals, 10,000 draws, seed 2026.

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

Adjustment sessions are dev only. Their order and arguments are recorded in each `reports/adjustments/tier*_round*.md`:

```bash
.venv/bin/python scripts/inspect_tier.py 1 && .venv/bin/python sessions/tier1_inspect_dla.py 1
.venv/bin/python sessions/tier1_round1.py
.venv/bin/python sessions/dla_cv.py 1 && .venv/bin/python sessions/dla_cv_random.py 1 && .venv/bin/python sessions/tier1_round2.py
.venv/bin/python scripts/lock_tier.py 1 t1r1_dla44-47_k32_s0.5
# tiers 2 and 3: sessions/dla_round.py TIER ROUND name=layers:K:s[:subset[:stat]] ...; scripts/lock_tier.py TIER NAME
.venv/bin/python scripts/lock.py                                      # experiment lock: unseals test evaluation
```

The locked edits are rebuilt from the original checkpoint and their compact specs, checking each identity against the lock. Then the sealed test and the analysis run:

```bash
.venv/bin/python scripts/edit_specs.py rebuild
.venv/bin/python -m src.evaluate test          # ~5 h for 6 edited conditions
.venv/bin/python -m src.analyze && .venv/bin/python -m src.figures && .venv/bin/python -m src.paper
```

`src.analyze`, `src.figures` and `src.paper` need only the saved item records in `results/` and run on CPU. `src.paper` regenerates the report's figures and LaTeX tables into `overleaf/`; the LaTeX source of `report.pdf` itself is not part of the repository. Benchmark text is not committed; records keep ids, hashes and scores.

## Layout

| folder | contents |
|---|---|
| `src/` | library: data, model, scoring, edits, evaluate, session tools, analysis, figures, report figures and tables |
| `sessions/` | the agent's per-round scripts |
| `scripts/` | setup, validation, locking, token accounting |

See `THIRD_PARTY.md` for licenses.

## AI usage

The article analysis, hypothesis, research and experiment design are the author's own. Other contributions:
- **ChatGPT (GPT-6 Astra) and Claude (Opus 5.5):** helped find and summarise information on the web.
- **Codex (GPT-6 Astra):** implemented a pilot run.
- **Claude Opus 5.5 (high effort) in Claude Code:** in the main run, wrote the code after the author defined the repository structure and targets, and acted as the editing agent. Its round reports are experimental artifacts. It also assisted with writing.

No AI judge evaluated any output. Details are in [`AI_USAGE.md`](AI_USAGE.md).

## Publishing

No remote is configured. After review, create an empty GitHub repository and run:

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
```

```bash
git push -u origin main
```
