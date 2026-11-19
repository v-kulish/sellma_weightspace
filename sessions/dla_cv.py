"""Within-dev 2-fold cross-validation over phenomena for direct-effect neuron scaling.
Select neurons on fold A pairs (Swe-BLiMP phenomena + half of idioms), score held-out fold B pairs, and vice versa.
Uses only tier-N dev pairs; reports held-out-fold margin and accuracy change vs unchanged."""
import sys, pathlib, json, random, itertools; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, numpy as np
from src import data
from src.model import load
from src.scoring import Scorer
from src.edits import Edit, applied, neuron_scaling
from src.common import ROOT, write
TIER = int(sys.argv[1])
MODE = sys.argv[2] if len(sys.argv) > 2 else 'across'   # 'across' phenomena (default) or 'within' phenomena
STAT = sys.argv[3] if len(sys.argv) > 3 else 'mean'     # neuron ranking statistic: 'mean' or 't'
model, tok = load(); sc = Scorer(model, tok)
blimp = data.load_pairs('swe_blimp', 'dev', TIER); idi = data.load_pairs('swedish_idioms', 'dev', TIER)
rows = blimp + idi
S = torch.load(ROOT / f'results/inspect/tier{TIER}_dla.pt')
assert S['rows'] == [r['id'] for r in rows]
S = S['scores']
phen = sorted({r['phenomenon'] for r in blimp}); rng = random.Random(7); rng.shuffle(phen)
foldA_ph = set(phen[: len(phen) // 2])
if MODE == 'across':
    fold = [0 if (r['benchmark'] == 'swe_blimp' and r['phenomenon'] in foldA_ph) or (r['benchmark'] != 'swe_blimp' and i % 2 == 0) else 1
            for i, r in enumerate(rows)]
else:  # within: alternate pairs inside each phenomenon (mirrors the test split: unseen examples of seen phenomena)
    seen = {}
    fold = []
    for i, r in enumerate(rows):
        key = r['phenomenon'] if r['benchmark'] == 'swe_blimp' else 'idioms'
        seen[key] = seen.get(key, -1) + 1
        fold.append(seen[key] % 2)
M = 'model.language_model.layers.{}.mlp.down_proj'
def score(subset):
    recs = sc.pairs(subset, 'cv', raw=False)
    return np.array([x['margin'] for x in recs]), np.array([x['correct'] for x in recs])
base = {f: score([r for r, g in zip(rows, fold) if g == f]) for f in (0, 1)}
grid = dict(layers=[(44, 45, 46, 47), (40, 41, 42, 43, 44, 45, 46, 47), (36, 38, 40, 42, 44, 46)], K=[8, 32, 64], s=[0.25, 0.5, 1.0])
if STAT != 'mean':
    grid = dict(layers=[(44, 45, 46, 47), (40, 41, 42, 43, 44, 45, 46, 47)], K=[32, 64], s=[0.25, 0.5])
res = []
for layers, K, s in itertools.product(*grid.values()):
    out = {}
    for f in (0, 1):
        sel = [i for i, g in enumerate(fold) if g == f]          # select on fold f
        held = [r for r, g in zip(rows, fold) if g != f]          # evaluate on the other fold
        ops = []
        for l in layers:
            x = S[l][sel]
            m = x.mean(0) if STAT == 'mean' else x.mean(0) / (x.std(0) / len(x) ** 0.5 + 1e-9)
            k2 = K // 2
            top = torch.topk(m, k2).indices.tolist(); bot = torch.topk(-m, k2).indices.tolist()
            ops.append(neuron_scaling(model, M.format(l), top + bot, [1 + s] * k2 + [max(0.0, 1 - s)] * k2))
        with applied(model, Edit(ops, 'cv')):
            mg, cr = score(held)
        bm, bc = base[1 - f]
        isb = np.array([r['benchmark'] == 'swe_blimp' for r in held])
        out[f] = dict(d_margin=float((mg - bm).mean()), d_acc_blimp=float((cr[isb] - bc[isb]).mean()),
                      d_acc_idioms=float((cr[~isb] - bc[~isb]).mean()), n_better=int(((mg - bm) > 0).sum()), n=len(held))
    r = dict(layers=list(layers), K=K, s=s, folds=out,
             mean_d_margin=np.mean([out[f]['d_margin'] for f in (0, 1)]),
             mean_d_acc_blimp=np.mean([out[f]['d_acc_blimp'] for f in (0, 1)]),
             mean_d_acc_idioms=np.mean([out[f]['d_acc_idioms'] for f in (0, 1)]))
    res.append(r)
    print(f"{str(layers):34s} K={K:3d} s={s:.2f}  heldout dmargin={r['mean_d_margin']:+.4f}  dacc_blimp={100*r['mean_d_acc_blimp']:+.1f}pp  dacc_idioms={100*r['mean_d_acc_idioms']:+.1f}pp  better={[out[f]['n_better'] for f in (0,1)]}/{[out[f]['n'] for f in (0,1)]}", flush=True)
write(f'results/inspect/tier{TIER}_dla_cv' + ('' if MODE == 'across' else '_within') + ('' if STAT == 'mean' else '_' + STAT) + '.json', dict(mode=MODE, stat=STAT, fold_A_phenomena=sorted(foldA_ph), results=res, accounting=sc.accounting()))
print('ACC', sc.accounting())
