# Baseline: unchanged google/gemma-4-12B-it (float32)

Model revision `43c9d014b231474fa7212dbb48a762566c2b255d`; FP32 weights and compute, SDPA, no TF32. Primary minimal-pair protocol: sentence as the reply to the user turn "Skriv en mening på svenska." (thinking disabled; sentence tokens only; mean log-prob). Secondary: raw BOS-only (brief §7 default). Intervals: 95% cluster bootstrap (10,000 draws, seed 2026), single condition.

**Exposure note:** these baseline *test* numbers were computed and viewed before any adjustment session, at the user's request (DESIGN_DECISIONS #4a, #9).

## Headline (test)

| benchmark | n (groups) | metric | value | 95% CI |
|---|---|---|---|---|
| belebele_dan_Latn | 201 (113) | accuracy | 89.55 | [84.7, 93.9] |
| belebele_eng_Latn | 201 (113) | accuracy | 94.53 | [91.0, 97.5] |
| belebele_fin_Latn | 201 (113) | accuracy | 91.04 | [86.9, 94.9] |
| belebele_isl_Latn | 201 (113) | accuracy | 86.07 | [81.2, 90.7] |
| belebele_nob_Latn | 201 (113) | accuracy | 92.54 | [88.6, 96.1] |
| belebele_swe_Latn | 201 (113) | accuracy | 88.56 | [84.1, 92.8] |
| belebele_zho_Hans | 201 (113) | accuracy | 90.05 | [85.7, 94.0] |
| scala_da | 256 (256) | accuracy | 75.78 | [70.7, 80.9] |
| scala_da | 256 (256) | mcc | 51.62 | [41.0, 61.9] |
| scala_en | 256 (256) | accuracy | 74.22 | [68.8, 79.3] |
| scala_en | 256 (256) | mcc | 50.03 | [39.6, 60.1] |
| scala_nb | 256 (256) | accuracy | 78.91 | [73.8, 83.6] |
| scala_nb | 256 (256) | mcc | 58.69 | [48.7, 67.9] |
| scala_sv | 256 (256) | accuracy | 82.03 | [77.3, 86.7] |
| scala_sv | 256 (256) | mcc | 64.26 | [54.7, 73.5] |
| swe_blimp_sv | 2458 (2137) | accuracy | 69.08 | [67.3, 70.8] |
| swe_blimp_sv | 2458 (2137) | raw_accuracy | 52.22 | [50.2, 54.2] |
| swedish_idioms_sv | 430 (430) | accuracy | 73.49 | [69.3, 77.7] |
| swedish_idioms_sv | 430 (430) | raw_accuracy | 49.53 | [44.9, 54.2] |

## Swe-BLiMP (test, 2,458 pairs; 38 quarantined related pairs excluded)

- Macro accuracy (primary, chat-sv): **69.08%**; micro 68.92%; mean margin +0.4400 nats/token; ties 0.
- Sensitivity: total-log-prob accuracy 70.14%; equal-token-length subset (n=1633) 71.71%.
- Secondary raw-BOS protocol: macro 52.22%, mean margin +0.0688.

| phenomenon | n | chat-sv acc | raw-BOS acc | mean margin (chat) |
|---|---|---|---|---|
| ADJ. NOUN AGR. DEF. | 96 | 74.0 | 52.1 | +0.694 |
| ADJ. NOUN AGR. GEN. | 91 | 74.7 | 50.5 | +0.489 |
| ADJ. NOUN AGR. INDEF. | 87 | 79.3 | 50.6 | +0.572 |
| ADJ. NOUN AGR. NUM. PL. | 96 | 89.6 | 49.0 | +0.778 |
| ADJ. NOUN AGR. NUM. SG. | 87 | 89.7 | 48.3 | +1.065 |
| ARG. STR. | 96 | 66.7 | 51.0 | +0.493 |
| ART. NOUN AGR. GEN. | 93 | 90.3 | 82.8 | +0.997 |
| ART. NOUN AGR. INDEF. | 96 | 63.5 | 40.6 | +0.493 |
| ART. NOUN AGR. PL. | 96 | 91.7 | 60.4 | +1.143 |
| ART. NOUN AGR. SG. | 92 | 68.5 | 56.5 | +0.630 |
| DOUBLE DEF. | 95 | 96.8 | 58.9 | +0.988 |
| ELLIPSIS1 | 92 | 78.3 | 43.5 | +0.587 |
| ELLIPSIS2 | 93 | 60.2 | 62.4 | +0.080 |
| EMBEDDED V2 AdvP. | 96 | 80.2 | 45.8 | +0.351 |
| EMBEDDED V3 CL. | 96 | 52.1 | 51.0 | +0.022 |
| EMBEDDED V3 OBJ. | 96 | 15.6 | 39.6 | -0.416 |
| NO RESUMPTION | 96 | 95.8 | 96.9 | +0.546 |
| PARTICLE OBJ. ORDER | 96 | 65.6 | 74.0 | +0.341 |
| RESUMPTION 1 | 96 | 45.8 | 17.7 | +0.046 |
| RESUMPTION 2 | 96 | 2.1 | 26.0 | -0.521 |
| V2 REL. CL. | 96 | 74.0 | 43.8 | +0.232 |
| V3 | 96 | 84.4 | 61.5 | +0.884 |
| V3 CL. | 96 | 60.4 | 42.7 | +0.101 |
| V3 COMPLEX NP | 96 | 69.8 | 55.2 | +0.279 |
| V3 OBJ. | 96 | 60.4 | 46.9 | +0.196 |
| V3 PP | 96 | 66.7 | 50.0 | +0.471 |

## Swedish Idioms (test, 430 pairs)

- Pair accuracy (primary, chat-sv): **73.49%**; raw-BOS 49.53%; mean margin +0.4167; total-log-prob accuracy 71.86%.

| subset | n | acc |
|---|---|---|
| challenge | 92 | 64.1 |
| standard | 338 | 76.0 |

## ScaLA (test, 256 rows per language; 12 fixed official-train few-shots; greedy, strict parser)

Custom protocol `scala_euroeval_wording_12shot_greedy_strict_v1` (EuroEval wording, public alexandrainst/scala data); not an official EuroEval score. MCC treats invalid outputs as a third predicted class.

| lang | acc | MCC×100 | macro F1 | invalid | acc on correct | acc on incorrect | acc delete | acc flip_neighbours |
|---|---|---|---|---|---|---|---|---|
| sv | 82.0 | 64.3 | 82.0 | 0.0% | 78.1 (n=128) | 85.9 (n=128) | 84.4 (n=64) | 87.5 (n=64) |
| da | 75.8 | 51.6 | 75.8 | 0.0% | 78.1 (n=128) | 73.4 (n=128) | 72.1 (n=61) | 74.6 (n=67) |
| nb | 78.9 | 58.7 | 78.7 | 0.0% | 70.3 (n=128) | 87.5 (n=128) | 86.9 (n=61) | 88.1 (n=67) |
| en | 74.2 | 50.0 | 73.8 | 0.0% | 61.7 (n=128) | 86.7 (n=128) | 80.6 (n=67) | 93.4 (n=61) |

Distinct raw outputs (all languages): {'ja': 358, 'nej': 260, 'no': 160, 'nei': 150, 'yes': 96}

## Belebele (test, 201 aligned questions in 113 passages; same IDs across languages)

Native-language instruction, chat template with thinking disabled, answer-label log-likelihood over A/B/C/D.

| language | acc | 95% CI | mean gold margin (nats) |
|---|---|---|---|
| swe_Latn | 88.6 | [84.1, 92.8] | 11.78 |
| dan_Latn | 89.6 | [84.7, 93.9] | 12.04 |
| nob_Latn | 92.5 | [88.6, 96.1] | 12.05 |
| fin_Latn | 91.0 | [86.9, 94.9] | 12.45 |
| isl_Latn | 86.1 | [81.2, 90.7] | 9.87 |
| eng_Latn | 94.5 | [91.0, 97.5] | 13.47 |
| zho_Hans | 90.0 | [85.7, 94.0] | 12.20 |

## Dev tiers (unchanged model; these are the only sets adjustment sessions may use)

| task | tier 1 | tier 2 | tier 3 |
|---|---|---|---|
| belebele_dan_Latn | 90.0 (n=10) | 90.0 (n=20) | 87.5 (n=40) |
| belebele_eng_Latn | 90.0 (n=10) | 90.0 (n=20) | 90.0 (n=40) |
| belebele_fin_Latn | 90.0 (n=10) | 95.0 (n=20) | 95.0 (n=40) |
| belebele_isl_Latn | 80.0 (n=10) | 90.0 (n=20) | 87.5 (n=40) |
| belebele_nob_Latn | 100.0 (n=10) | 90.0 (n=20) | 85.0 (n=40) |
| belebele_swe_Latn | 90.0 (n=10) | 90.0 (n=20) | 87.5 (n=40) |
| belebele_zho_Hans | 90.0 (n=10) | 90.0 (n=20) | 90.0 (n=40) |
| scala_da | 87.5 (n=8) | 87.5 (n=16) | 87.5 (n=32) |
| scala_en | 62.5 (n=8) | 75.0 (n=16) | 81.2 (n=32) |
| scala_nb | 87.5 (n=8) | 81.2 (n=16) | 81.2 (n=32) |
| scala_sv | 100.0 (n=8) | 93.8 (n=16) | 87.5 (n=32) |
| swe_blimp_sv | 53.8 (n=26) | 61.5 (n=52) | 63.5 (n=104) |
| swedish_idioms_sv | 91.7 (n=12) | 79.2 (n=24) | 70.8 (n=48) |

Tier views here are filtered from the tier-3 run; each adjustment session re-scores its own tier baseline.

## Compute

Baseline dev+test wall time 47.6 min; peak allocated 52.0 GB; accounting {'forward_calls': 4381, 'forward_seconds': 2841.428, 'generated_tokens': 2304, 'input_tokens': 782475, 'sequences': 14999}.
