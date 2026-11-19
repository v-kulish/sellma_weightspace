"""Unchanged-model inspection for an adjustment session, dev tier N only.

1. Grammar contrasts: pooled MLP neuron activations / down_proj outputs for good vs bad sentences of the
   tier-N Swe-BLiMP + Idioms dev pairs (primary chat context), every text layer.
2. Language selectivity: pooled neuron activations on the tier-N Belebele dev passages, parallel in 7 languages.
Writes results/inspect/tier{N}.pt (pooled statistics only) and a JSON summary.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from src import data
from src.common import ROOT, write, now
from src.model import load, text_layers
from src.scoring import Scorer
from src.session import pair_activations, text_activations, logit_lens

N = int(sys.argv[1])
model, tok = load(); sc = Scorer(model, tok)
layers = list(range(len(text_layers(model))))
rows = data.load_pairs('swe_blimp', 'dev', N) + data.load_pairs('swedish_idioms', 'dev', N)
acts = pair_activations(sc, rows, layers)
summary = dict(utc=now(), tier=N, n_pairs=len(rows), layers={})
store = dict(pairs=[r['id'] for r in rows], phenomena=[r['phenomenon'] for r in rows], grammar={}, language={})
for l in layers:
    g, b = acts[l]['neurons']['good'], acts[l]['neurons']['bad']
    dz = acts[l]['out']['good'] - acts[l]['out']['bad']
    d = g - b
    t = d.mean(0) / (d.std(0) / len(d) ** 0.5 + 1e-6)
    mean_dir = dz.mean(0)
    cos = torch.nn.functional.cosine_similarity(dz, mean_dir[None], dim=1)
    store['grammar'][l] = dict(neuron_diff_mean=d.mean(0), neuron_t=t, out_diff_mean=mean_dir,
                               out_good_mean=acts[l]['out']['good'].mean(0))
    summary['layers'][l] = dict(out_contrast_rel=float(mean_dir.norm() / acts[l]['out']['good'].norm(dim=1).mean()),
                                per_pair_cos_mean=float(cos.mean()), frac_pairs_cos_pos=float((cos > 0).float().mean()),
                                n_neurons_abs_t_gt4=int((t.abs() > 4).sum()))
# language selectivity on parallel Belebele dev passages
langs = data.BELEBELE_LANGS
passages = {}
for lang in langs:
    seen = {}
    for r in data.load_belebele(lang, 'dev', N):
        seen.setdefault(r['link'], r['row']['flores_passage'])
    passages[lang] = [seen[k] for k in sorted(seen)]
lang_acts = {lang: text_activations(sc, passages[lang], layers) for lang in langs}
for l in layers:
    M = {lang: lang_acts[lang][l]['neurons'] for lang in langs}   # passages x neurons
    store['language'][l] = {lang: M[lang].mean(0) for lang in langs}
    sv, scand = M['swe_Latn'], torch.cat([M['dan_Latn'], M['nob_Latn']])
    pooled = torch.cat([sv, scand]).std(0) + 1e-4
    sel = (sv.mean(0) - scand.mean(0)) / pooled
    store['language'][l]['sv_vs_dannob'] = sel
    summary['layers'][l]['sv_vs_dannob_top'] = float(sel.max()); summary['layers'][l]['sv_vs_dannob_bottom'] = float(sel.min())
    summary['layers'][l]['n_sv_selective_gt2'] = int((sel > 2).sum())
summary['passages_per_language'] = len(passages['swe_Latn'])
summary['accounting'] = sc.accounting()
(ROOT / 'results/inspect').mkdir(parents=True, exist_ok=True)
torch.save(store, ROOT / f'results/inspect/tier{N}.pt')
write(f'results/inspect/tier{N}_summary.json', summary)
print(json.dumps({l: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
                  for l, s in summary['layers'].items()}, indent=0)[:6000])
print('ACCOUNTING', summary['accounting'])
