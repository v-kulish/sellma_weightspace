# Results summary

Configuration, outcomes and examples for the main experiment. Interpretation is in [`RESEARCH_NOTE.md`](RESEARCH_NOTE.md) and the full report [`REPORT.pdf`](REPORT.pdf). All numbers can be regenerated on CPU from the saved item records (`python -m src.analyze`).

## Configuration

| item | value |
|---|---|
| Model | `google/gemma-4-12B-it`, revision `43c9d014b231474fa7212dbb48a762566c2b255d` (Apache 2.0) |
| Numerics | float32 weights and computation, SDPA attention, TF32 off, eager PyTorch, batch scoring validated against single scoring |
| Software | torch 2.14.0+cu130, transformers 5.23.0 (`requirements.lock`, `manifests/environment.json`) |
| Hardware | NVIDIA DGX Spark (GB10, ARM64, 121 GiB unified memory), driver 580.173.02, CUDA 13.0; peak 56 GB |
| Data revisions | Swe-BLiMP `4770e273`, Swedish Idioms `3d33f55d`, ScaLA `890966a7`, Belebele `7899cdfa`, EuroEval prompts `efb93af3`; SHA-256 checksums in `manifests/sources.json` |
| Primary pair scoring | mean log-probability of the sentence tokens, scored as the model's reply to the user turn `Skriv en mening på svenska.` (Gemma chat template, thinking disabled); pair correct if the grammatical sentence scores strictly higher |
| Secondary pair scoring | raw sentence after `<bos>` only |
| ScaLA protocol | EuroEval wording, 12 fixed few-shot examples from the train split, greedy decoding, strict `ja`/`nej` parser; invalid outputs count as errors and as a third class in MCC |
| Belebele protocol | native-language instruction in the chat template; prediction = highest log-probability of the letters A–D |
| Seeds | split 42, random-control edits 43, bootstrap 2026 (10,000 paired cluster-bootstrap draws) |
| Splits (dev tiers / test) | Swe-BLiMP 26/52/104 / 2,458 pairs (38 related pairs quarantined); Idioms 12/24/48 / 430; ScaLA 8/16/32 / 256 per language; Belebele 10/20/40 / 201 questions (113 passages) per language. Manifests: `manifests/splits/` |
| Edit protocol | `reports/adjustments/PROTOCOL.md`: low-rank MLP edits (rank ≤ 64, ≤ 10% relative norm per matrix), ≤ 4 rounds × 4 candidates per tier, fixed lock rule |
| Sealing | edited test scores were computed only after `manifests/experiment_lock.json` was written (enforced in `src/evaluate.py`) |

## Locked edits

| tier | edit | change | entries changed | random control |
|---|---|---|---|---|
| 1 | `t1r1_dla44-47_k32_s0.5` | `down_proj` layers 44–47: per layer 32 neurons ×1.5, 32 neurons ×0.5 | 983,040 | same matrices, rank and singular values; random singular vectors (seed 43) |
| 2 | `t2r1_dla40-47_K64_s0.5` | layers 40–47: 32 neurons ×1.5, 32 ×0.5 | 1,966,080 | as above |
| 3 (primary) | `t3r1_dla40-47_K64_s0.25` | layers 40–47: 32 neurons ×1.25, 32 ×0.75 | 1,966,080 | as above |

The edits are stored as compact specifications in `artifacts/specs/` and are rebuilt exactly with `scripts/edit_specs.py rebuild`. All 22 dev candidates and their dev scores are listed in `results/rounds/` and in Table 5 of the report.

## Test outcomes

Swe-BLiMP (2,458 pairs). Differences in pp with 95% intervals; margins in nats per token.

| | Tier 1 | Tier 2 | Tier 3 (primary) |
|---|---|---|---|
| Edited accuracy (unchanged 69.08%) | 70.38 | 70.37 | 69.69 |
| Edit − unchanged | +1.30 [−0.04, +2.58] | +1.29 [−0.15, +2.73] | +0.61 [−0.42, +1.65] |
| Edit − random | +0.62 [−0.69, +1.89] | +0.97 [−0.48, +2.37] | +0.32 [−0.70, +1.37] |
| Random − unchanged | +0.68 [+0.27, +1.13] | +0.32 [−0.13, +0.81] | +0.29 [−0.07, +0.67] |
| Margin, edit − unchanged | +0.04 [+0.02, +0.06] | +0.16 [+0.13, +0.18] | +0.11 [+0.09, +0.12] |
| Raw-scored accuracy, edit − unchanged (unchanged 52.22%) | −0.35 [−1.74, +1.02] | +0.17 [−1.49, +1.87] | −0.04 [−1.21, +1.10] |
| Verdict (≥ +2 pp with interval above zero) | inconclusive | inconclusive | inconclusive |

Other benchmarks: test accuracy (%; ScaLA MCC ×100 in brackets).

| task | n | unchanged | T1 edit | T1 random | T2 edit | T2 random | T3 edit | T3 random |
|---|---|---|---|---|---|---|---|---|
| Swedish Idioms | 430 | 73.49 | 73.02 | 73.72 | 72.33 | 74.19 | 73.72 | 74.19 |
| ScaLA sv | 256 | 82.03 (64.3) | 82.42 (64.9) | 81.64 (63.5) | 81.64 (63.5) | 82.03 (64.3) | 81.64 (63.5) | 82.03 (64.3) |
| ScaLA da | 256 | 75.78 (51.6) | 75.78 (51.7) | 75.78 (51.6) | 75.78 (51.6) | 75.78 (51.6) | 76.17 (52.4) | 75.78 (51.6) |
| ScaLA nb | 256 | 78.91 (58.7) | 79.30 (59.3) | 78.91 (58.7) | 79.69 (60.1) | 78.91 (58.7) | 79.30 (59.6) | 78.91 (58.7) |
| ScaLA en | 256 | 74.22 (50.0) | 74.61 (50.7) | 74.22 (50.0) | 74.22 (50.0) | 74.22 (49.8) | 75.00 (51.9) | 74.61 (50.7) |
| Belebele swe | 201 | 88.56 | 88.06 | 88.06 | 88.56 | 88.56 | 88.56 | 88.56 |
| Belebele dan | 201 | 89.55 | 89.55 | 89.55 | 89.55 | 89.55 | 89.55 | 89.55 |
| Belebele nob | 201 | 92.54 | 92.54 | 92.54 | 92.04 | 92.54 | 92.54 | 92.54 |
| Belebele fin | 201 | 91.04 | 91.04 | 91.04 | 91.54 | 91.04 | 91.04 | 91.04 |
| Belebele isl | 201 | 86.07 | 86.07 | 86.07 | 86.57 | 86.07 | 86.07 | 86.07 |
| Belebele eng | 201 | 94.53 | 94.53 | 94.53 | 95.02 | 94.53 | 95.02 | 94.53 |
| Belebele zho | 201 | 90.05 | 89.55 | 90.05 | 90.05 | 90.05 | 90.05 | 90.05 |

All 21 Belebele edit-versus-unchanged intervals lie above the −3 pp tolerance. Paired intervals for every comparison are in [`results/tables.md`](results/tables.md), and the per-phenomenon breakdown is in [`reports/baseline/BASELINE.md`](reports/baseline/BASELINE.md) and `reports/phenomenon_changes.png`.

## Examples

Outcome changes for the tier-3 edit on test, relative to the unchanged model:

| | Swe-BLiMP | Swedish Idioms |
|---|---|---|
| corrected | 91 | 13 |
| newly broken | 76 | 12 |
| unchanged correct | 1,618 | 304 |
| unchanged wrong | 673 | 101 |

Examples chosen by a fixed rule: within each outcome class, the first item ids by SHA-256 (`src/paper.py`; ids and margins for all classes in `results/examples.json`). The sentences are short excerpts from Swe-BLiMP, quoted for illustration. The dataset is not redistributed here. `m0` and `m3` are the margins (nats per token) of the unchanged model and of the tier-3 edit.

| class | phenomenon | grammatical | ungrammatical | m0 | m3 |
|---|---|---|---|---|---|
| corrected | ADJ. NOUN AGR. DEF. | Mamman var ung | Mamman var ungt | −0.30 | +0.33 |
| corrected | V3 CL. | När advokaten sprang sålde mästaren skäran | När advokaten sprang mästaren sålde skäran | −0.18 | +0.18 |
| newly broken | NO RESUMPTION | Niklas litar på en bläckfisk som han inte vet om den lever | Niklas litar på en bläckfisk som han inte vet om lever | +0.06 | −0.02 |
| newly broken | ART. NOUN AGR. INDEF. | En låt tillkom under nästet | En låten tillkom under nästet | +0.26 | −0.26 |
| still wrong | ART. NOUN AGR. GEN. | Det rullade en boll under palatset | Det rullade ett boll under palatset | −1.51 | −1.51 |
| still wrong | EMBEDDED V2 AdvP. | Anders medgav att Kristina oftast tappar humöret i juni | Anders medgav att Kristina tappar oftast humöret i juni | −0.50 | −0.74 |

Effect of the scoring prompt on the unchanged model: for *På tisdagar matar Hanna hunden* / \**På tisdagar Hanna matar hunden*, raw scoring gives −14.95 vs −14.93 (wrong) and the chat prompt gives −10.72 vs −11.11 (right). Over the test set the prompt corrects 709 pairs and breaks 298 (52.2% → 69.1%).

## Compute and agent effort

| tier | rounds | candidates | labelled dev items | Gemma forward tokens | agent output tokens (thinking) |
|---|---|---|---|---|---|
| 1 | 2 | 8 | 140 | 338,933 | 31,552 (11,568) |
| 2 | 2 | 8 | 280 | 669,433 | 11,833 (2,587) |
| 3 | 2 | 6 | 560 | 1,537,745 | 8,676 (2,519) |

All three tiers ran in one agent session, so tiers 2 and 3 reused tools and knowledge from tier 1. GPU time: baseline about 48 minutes, adjustment rounds 2.3 hours, sealed test 4.2 hours. Per-round accounting is in each `reports/adjustments/tier*_round*.md` and in `results/session_summary.md`.

## Where the files are

| path | contents |
|---|---|
| `results/test/<condition>/` | item-level test records per benchmark: ids, hashes, scores, margins, predictions |
| `results/dev/tier{1,2,3}/` | dev records for the unchanged model and every candidate |
| `results/rounds/`, `results/tiers/` | candidate evaluations per round; tier locks with random-control dev results |
| `results/analysis.json`, `results/tables.md` | bootstrap statistics and verdicts |
| `manifests/` | sources, split manifests, environment, experiment lock |
| `artifacts/specs/` | compact specifications of the locked edits and random controls |
| `reports/adjustments/` | the agent's reasoning report for every round, with token accounting |
