"""Detailed baseline report across all benchmarks (dev tiers and test) -> reports/baseline/BASELINE.md"""
import sys, pathlib, collections, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from src.common import ROOT, read, write
from src.analyze import task_stats, load_condition


def ci(reps):
    lo, hi = np.percentile(reps * 100, [2.5, 97.5])
    return f'[{lo:.1f}, {hi:.1f}]'


def tier_view(recs, t):
    return [r for r in recs if r['tier'] is not None and r['tier'] <= t]


test = load_condition('baseline', 'test')
dev = load_condition('baseline', 'dev', 3)
L = ['# Baseline: unchanged google/gemma-4-12B-it (float32)', '',
     'Model revision `43c9d014b231474fa7212dbb48a762566c2b255d`; FP32 weights and compute, SDPA, no TF32. '
     'Primary minimal-pair protocol: sentence as the reply to the user turn "Skriv en mening på svenska." '
     '(thinking disabled; sentence tokens only; mean log-prob). Secondary: raw BOS-only (brief §7 default). '
     'Intervals: 95% cluster bootstrap (10,000 draws, seed 2026), single condition.', '',
     '**Exposure note:** these baseline *test* numbers were computed and viewed before any adjustment session, '
     'at the user\'s request (DESIGN_DECISIONS #4a, #9).', '']

# ---- headline
L += ['## Headline (test)', '', '| benchmark | n (groups) | metric | value | 95% CI |', '|---|---|---|---|---|']
for task, d in test.items():
    st, n, g = task_stats(task, {'baseline': d['records']}, 2026)
    for m, (p, reps) in st['baseline'].items():
        if reps is None:
            continue
        L.append(f'| {task} | {n} ({g}) | {m} | {100 * p:.2f} | {ci(reps)} |')
L += ['']

# ---- Swe-BLiMP detail
sb = test['swe_blimp_sv']['summary']; recs = test['swe_blimp_sv']['records']
L += ['## Swe-BLiMP (test, 2,458 pairs; 38 quarantined related pairs excluded)', '',
      f"- Macro accuracy (primary, chat-sv): **{100 * sb['macro_accuracy']:.2f}%**; micro {100 * sb['micro_accuracy']:.2f}%; "
      f"mean margin {sb['mean_margin']:+.4f} nats/token; ties {sb['ties']}.",
      f"- Sensitivity: total-log-prob accuracy {100 * sb['total_logp_accuracy']:.2f}%; equal-token-length subset "
      f"(n={sb['equal_length_n']}) {100 * sb['equal_length_accuracy']:.2f}%.",
      f"- Secondary raw-BOS protocol: macro {100 * sb['raw_macro_accuracy']:.2f}%, mean margin {sb['raw_mean_margin']:+.4f}.", '',
      '| phenomenon | n | chat-sv acc | raw-BOS acc | mean margin (chat) |', '|---|---|---|---|---|']
by = collections.defaultdict(list)
for r in recs:
    by[r['phenomenon']].append(r)
for p in sorted(by):
    rs = by[p]
    L.append(f"| {p} | {len(rs)} | {100 * np.mean([r['correct'] for r in rs]):.1f} | "
             f"{100 * np.mean([r['raw_correct'] for r in rs]):.1f} | {np.mean([r['margin'] for r in rs]):+.3f} |")
L += ['']

# ---- Idioms
ir = test['swedish_idioms_sv']['records']; si = test['swedish_idioms_sv']['summary']
L += ['## Swedish Idioms (test, 430 pairs)', '',
      f"- Pair accuracy (primary, chat-sv): **{100 * si['micro_accuracy']:.2f}%**; raw-BOS "
      f"{100 * np.mean([r['raw_correct'] for r in ir]):.2f}%; mean margin {si['mean_margin']:+.4f}; "
      f"total-log-prob accuracy {100 * si['total_logp_accuracy']:.2f}%.", '',
      '| subset | n | acc |', '|---|---|---|']
for p in ['challenge', 'standard']:
    rs = [r for r in ir if r['phenomenon'] == p]
    L.append(f"| {p} | {len(rs)} | {100 * np.mean([r['correct'] for r in rs]):.1f} |")
L += ['']

# ---- ScaLA
L += ['## ScaLA (test, 256 rows per language; 12 fixed official-train few-shots; greedy, strict parser)', '',
      'Custom protocol `scala_euroeval_wording_12shot_greedy_strict_v1` (EuroEval wording, public alexandrainst/scala data); '
      'not an official EuroEval score. MCC treats invalid outputs as a third predicted class.', '',
      '| lang | acc | MCC×100 | macro F1 | invalid | acc on correct | acc on incorrect | acc delete | acc flip_neighbours |',
      '|---|---|---|---|---|---|---|---|---|']
for l in ['sv', 'da', 'nb', 'en']:
    d = test[f'scala_{l}']; s = d['summary']; rs = d['records']
    def acc(f):
        x = [r['correct'] for r in rs if f(r)]
        return f'{100 * np.mean(x):.1f} (n={len(x)})' if x else '—'
    L.append(f"| {l} | {100 * s['accuracy']:.1f} | {100 * s['mcc']:.1f} | {100 * s['macro_f1']:.1f} | {100 * s['invalid_rate']:.1f}% | "
             f"{acc(lambda r: r['gold'] == 1)} | {acc(lambda r: r['gold'] == 0)} | "
             f"{acc(lambda r: r['gold'] == 0 and r['corruption_type'] == 'delete')} | "
             f"{acc(lambda r: r['gold'] == 0 and r['corruption_type'] == 'flip_neighbours')} |")
outs = collections.Counter(r['raw_output'] for l in ['sv', 'da', 'nb', 'en'] for r in test[f'scala_{l}']['records'])
L += ['', f'Distinct raw outputs (all languages): {dict(outs.most_common(8))}', '']

# ---- Belebele
n_pass = len({r['group'] for r in test['belebele_swe_Latn']['records']})
L += [f'## Belebele (test, 201 aligned questions in {n_pass} passages; same IDs across languages)', '',
      'Native-language instruction, chat template with thinking disabled, answer-label log-likelihood over A/B/C/D.', '',
      '| language | acc | 95% CI | mean gold margin (nats) |', '|---|---|---|---|']
for l in ['swe_Latn', 'dan_Latn', 'nob_Latn', 'fin_Latn', 'isl_Latn', 'eng_Latn', 'zho_Hans']:
    d = test[f'belebele_{l}']
    st, n, g = task_stats(f'belebele_{l}', {'b': d['records']}, 2026)
    L.append(f"| {l} | {100 * d['summary']['accuracy']:.1f} | {ci(st['b']['accuracy'][1])} | {d['summary']['mean_gold_margin']:.2f} |")
L += ['']

# ---- dev tiers
L += ['## Dev tiers (unchanged model; these are the only sets adjustment sessions may use)', '',
      '| task | tier 1 | tier 2 | tier 3 |', '|---|---|---|---|']
for task, d in dev.items():
    cells = []
    for t in (1, 2, 3):
        rs = tier_view(d['records'], t)
        if task.startswith('swe_blimp'):
            cats = collections.defaultdict(list)
            for r in rs: cats[r['phenomenon']].append(r['correct'])
            v = np.mean([np.mean(x) for x in cats.values()])
        else:
            v = np.mean([r['correct'] for r in rs])
        cells.append(f'{100 * v:.1f} (n={len(rs)})')
    L.append(f'| {task} | ' + ' | '.join(cells) + ' |')
L += ['', 'Tier views here are filtered from the tier-3 run; each adjustment session re-scores its own tier baseline.', '']
rt = read('results/baseline_runtime.json')
L += ['## Compute', '', f"Baseline dev+test wall time {rt['seconds'] / 60:.1f} min; peak allocated {rt['peak_gb']:.1f} GB; "
      f"accounting {rt['accounting']}.", '']
(ROOT / 'reports/baseline/BASELINE.md').write_text('\n'.join(L))
print('\n'.join(L))
