"""Tier 1, round 1 inspection: direct-effect neuron scores (tier-1 Swe-BLiMP + Idioms dev pairs) and
Swedish-vs-Danish/Norwegian output directions (tier-1 Belebele dev passages)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, json
from src import data
from src.model import load
from src.scoring import Scorer
from src.session import dla_neuron_scores, language_direction, logit_lens
from src.common import ROOT, write
TIER = int(sys.argv[1]) if len(sys.argv) > 1 else 1
model, tok = load(); sc = Scorer(model, tok)
rows = data.load_pairs('swe_blimp', 'dev', TIER) + data.load_pairs('swedish_idioms', 'dev', TIER)
layers = list(range(12, 48))
S = dla_neuron_scores(sc, rows, layers)
store = torch.load(ROOT / f'results/inspect/tier{TIER}.pt')
summ = {}
for l in layers:
    s = S[l]; m = s.mean(0); t = m / (s.std(0) / len(s) ** 0.5 + 1e-9)
    frac = (s > 0).float().mean(0)
    d, rel = language_direction(store, l, model)
    summ[l] = dict(total_pos=float(m.clamp(min=0).sum()), total_neg=float(m.clamp(max=0).sum()), max_t=float(t.max()), min_t=float(t.min()),
                   n_t_gt3=int((t > 3).sum()), n_t_lt3=int((t < -3).sum()), top_mean=float(m.max()), lang_rel=rel)
torch.save(dict(scores=S, rows=[r['id'] for r in rows]), ROOT / f'results/inspect/tier{TIER}_dla.pt')
write(f'results/inspect/tier{TIER}_dla_summary.json', dict(layers=summ, accounting=sc.accounting()))
for l, v in summ.items():
    print(l, {k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()})
for l in (20, 24, 28, 32):
    d, rel = language_direction(store, l, model)
    print('LANG', l, round(rel, 3), logit_lens(model, tok, d, 10))
print('ACC', sc.accounting())
