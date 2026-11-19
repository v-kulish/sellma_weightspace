"""Tier 1 round 2: direct-effect neuron scaling configurations chosen by within-dev 2-fold CV (sessions/dla_cv.py)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from src.common import ROOT
from src.edits import Edit, neuron_scaling
from src.session import run_round
TIER, ROUND = 1, 2
M = 'model.language_model.layers.{}.mlp.down_proj'
L40 = (40, 41, 42, 43, 44, 45, 46, 47); L36 = (36, 38, 40, 42, 44, 46)

def dla_edit(model, dla, layers, K, s, name):
    ops = []
    for l in layers:
        m = dla[l].mean(0); k2 = K // 2
        top = torch.topk(m, k2).indices.tolist(); bot = torch.topk(-m, k2).indices.tolist()
        ops.append(neuron_scaling(model, M.format(l), top + bot, [1 + s] * k2 + [max(0.0, 1 - s)] * k2))
    return Edit(ops, name, f'direct-effect neuron scaling on all tier-{TIER} dev pairs: layers {list(layers)}, per layer '
                f'{K // 2} highest-score neurons x{1 + s}, {K // 2} lowest x{max(0.0, 1 - s)}')

def build(model, tok, scorer, store):
    dla = torch.load(ROOT / f'results/inspect/tier{TIER}_dla.pt')['scores']
    cfg = {'t1r2_dla40-47_K32_s0.5': (L40, 32, 0.5), 't1r2_dla40-47_K32_s1.0': (L40, 32, 1.0),
           't1r2_dla40-47_K8_s1.0': (L40, 8, 1.0), 't1r2_dla36-46e_K8_s1.0': (L36, 8, 1.0)}
    return {n: dla_edit(model, dla, *c, n) for n, c in cfg.items()}

if __name__ == '__main__':
    run_round(TIER, ROUND, build)
