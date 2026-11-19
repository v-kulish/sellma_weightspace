"""Within-dev check: do random neurons (same layers, K, s, same up/down split) move held-out-fold margins like
direct-effect-selected ones? Uses the same folds as dla_cv.py. Tier-N dev pairs only."""
import sys, pathlib, random; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, numpy as np
from src import data
from src.model import load
from src.scoring import Scorer
from src.edits import Edit, applied, neuron_scaling
from src.common import ROOT, write, read
TIER = int(sys.argv[1])
model, tok = load(); sc = Scorer(model, tok)
blimp = data.load_pairs('swe_blimp', 'dev', TIER); idi = data.load_pairs('swedish_idioms', 'dev', TIER)
rows = blimp + idi
cv = read(f'results/inspect/tier{TIER}_dla_cv.json'); fa = set(cv['fold_A_phenomena'])
fold = [0 if (r['benchmark'] == 'swe_blimp' and r['phenomenon'] in fa) or (r['benchmark'] != 'swe_blimp' and i % 2 == 0) else 1 for i, r in enumerate(rows)]
M = 'model.language_model.layers.{}.mlp.down_proj'
def margins(subset):
    return np.array([x['margin'] for x in sc.pairs(subset, 'cv', raw=False)])
held = {f: [r for r, g in zip(rows, fold) if g != f] for f in (0, 1)}
base = {f: margins(held[f]) for f in (0, 1)}
out = []
for layers, K, s in [((40, 41, 42, 43, 44, 45, 46, 47), 32, 0.5), ((40, 41, 42, 43, 44, 45, 46, 47), 8, 1.0), ((36, 38, 40, 42, 44, 46), 8, 1.0)]:
    for seed in range(4):
        g = torch.Generator().manual_seed(1000 + seed)
        ops = []
        for l in layers:
            idx = torch.randperm(15360, generator=g)[:K].tolist()
            ops.append(neuron_scaling(model, M.format(l), idx, [1 + s] * (K // 2) + [max(0.0, 1 - s)] * (K // 2)))
        with applied(model, Edit(ops, 'rand')):
            d = [float((margins(held[f]) - base[f]).mean()) for f in (0, 1)]
        out.append(dict(layers=list(layers), K=K, s=s, seed=seed, d_margin_folds=d, mean=float(np.mean(d))))
        print(layers, K, s, seed, [round(x, 4) for x in d], round(float(np.mean(d)), 4), flush=True)
write(f'results/inspect/tier{TIER}_dla_cv_random.json', dict(results=out, accounting=sc.accounting()))
print('ACC', sc.accounting())
