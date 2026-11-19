"""Generic direct-effect neuron-scaling round: candidates given on the command line as name=layers:K:s,
e.g.  t2r1_a=44-47:64:0.5   (K = total neurons per matrix: K/2 highest-score x(1+s), K/2 lowest x(1-s)).
Layers: 'a-b' (inclusive range) or 'a,b,c'. Usage: dla_round.py TIER ROUND spec [spec ...]"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from src.common import ROOT
from src.edits import Edit, neuron_scaling
from src.session import run_round
TIER, ROUND = int(sys.argv[1]), int(sys.argv[2])
M = 'model.language_model.layers.{}.mlp.down_proj'

def parse_layers(x):
    if '-' in x:
        a, b = map(int, x.split('-')); return list(range(a, b + 1))
    return [int(v) for v in x.split(',')]

def build(model, tok, scorer, store):
    dla = torch.load(ROOT / f'results/inspect/tier{TIER}_dla.pt')['scores']
    out = {}
    for spec in sys.argv[3:]:
        name, cfg = spec.split('=')
        parts = cfg.split(':'); lay, K, s = parts[:3]; layers, K, s = parse_layers(lay), int(K), float(s)
        subset = parts[3] if len(parts) > 3 else 'all'      # 'blimp' = select on Swe-BLiMP pairs only
        stat = parts[4] if len(parts) > 4 else 'mean'      # 't' = rank neurons by per-neuron t-statistic
        rows = torch.load(ROOT / f'results/inspect/tier{TIER}_dla.pt')['rows']
        idx = [i for i, r in enumerate(rows) if subset == 'all' or r.startswith('swe-')]
        ops = []
        for l in layers:
            x = dla[l][idx]; k2 = K // 2
            m = x.mean(0) if stat == 'mean' else x.mean(0) / (x.std(0) / len(x) ** 0.5 + 1e-9)
            top = torch.topk(m, k2).indices.tolist(); bot = torch.topk(-m, k2).indices.tolist()
            ops.append(neuron_scaling(model, M.format(l), top + bot, [1 + s] * k2 + [max(0.0, 1 - s)] * k2))
        out[name] = Edit(ops, name, f'direct-effect neuron scaling on tier-{TIER} dev pairs ({subset}, ranked by {stat}): layers {layers}, per layer '
                         f'{K // 2} highest-score neurons x{1 + s}, {K // 2} lowest x{max(0.0, 1 - s)}')
    return out

run_round(TIER, ROUND, build)
