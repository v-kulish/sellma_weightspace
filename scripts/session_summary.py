"""Aggregate per-tier adjustment budgets: dev items exposed, candidates, model-forward tokens, agent tokens."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import ROOT, read, write
from src.data import TIER_SIZES

rows = []
for t in (1, 2, 3):
    files = sorted((ROOT / 'results/sessions').glob(f'tier{t}_round*_tokens.json'))
    ag = {k: 0 for k in ('api_calls', 'output', 'thinking', 'input_uncached', 'cache_creation_input', 'cache_read_input')}
    fw = dict(input_tokens=0, generated_tokens=0, forward_calls=0, sequences=0, forward_seconds=0.0)
    for f in files:
        d = json.loads(f.read_text())
        for k in ag: ag[k] += d['agent'][k]
        for v in d['model_forward'].values():
            for k in fw: fw[k] += v.get(k, 0)
    cands = sum(len(json.loads(p.read_text())['candidates']) for p in (ROOT / 'results/rounds').glob(f'tier{t}_round*.json'))
    lock = read(f'results/tiers/tier{t}_lock.json')
    exposed = dict(swe_blimp=TIER_SIZES['swe_blimp'][t], swedish_idioms=TIER_SIZES['swedish_idioms'][t],
                   scala_per_lang=TIER_SIZES['scala'][t], belebele_per_lang=TIER_SIZES['belebele'][t])
    exposed['total_labelled_items'] = exposed['swe_blimp'] + exposed['swedish_idioms'] + 4 * exposed['scala_per_lang'] + 7 * exposed['belebele_per_lang']
    rows.append(dict(tier=t, rounds=len(files), full_suite_candidates=cands, locked=lock['candidate'], exposed=exposed,
                     agent=ag, model_forward=fw, fewshot_context_examples='12 per ScaLA language (fixed evaluation context, not edit feedback)'))
write('results/session_summary.json', rows)
L = ['| tier | rounds | full-suite candidates | labelled dev items exposed (BLiMP / Idioms / ScaLA×4 / Belebele×7 = total) | '
     'Gemma forward tokens (in+gen) | GPU forward s | agent API calls | agent output tokens (thinking) | agent cache-read input |',
     '|---|---|---|---|---|---|---|---|---|']
for r in rows:
    e = r['exposed']; f = r['model_forward']; a = r['agent']
    L.append(f"| {r['tier']} | {r['rounds']} | {r['full_suite_candidates']} | {e['swe_blimp']} / {e['swedish_idioms']} / {e['scala_per_lang']} / "
             f"{e['belebele_per_lang']} = {e['total_labelled_items']} | {f['input_tokens'] + f['generated_tokens']:,} | {f['forward_seconds']:.0f} | "
             f"{a['api_calls']} | {a['output']:,} ({a['thinking']:,}) | {a['cache_read_input']:,} |")
(ROOT / 'results/session_summary.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
