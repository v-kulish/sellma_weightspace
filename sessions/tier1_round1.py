"""Tier 1 round 1 candidates (see reports/adjustments/tier1_round1.md for reasoning)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from src.common import ROOT
from src.edits import Edit, projection_scaling, neuron_scaling
from src.session import run_round, language_direction
TIER, ROUND = 1, 1
M = 'model.language_model.layers.{}.mlp.down_proj'

def build(model, tok, scorer, store):
    dla = torch.load(ROOT / f'results/inspect/tier{TIER}_dla.pt')['scores']
    c = {}
    d42, _ = language_direction(store, 42, model)
    c['t1r1_lang42_a0.5'] = Edit([projection_scaling(model, M.format(42), d42, 0.5)], 't1r1_lang42_a0.5',
        'projection scaling (1+0.5) along Swedish-minus-Danish/Norwegian output direction, layer 42 down_proj')
    ops = []
    for l in (40, 42, 44):
        d, _ = language_direction(store, l, model); ops.append(projection_scaling(model, M.format(l), d, 0.3))
    c['t1r1_lang40-42-44_a0.3'] = Edit(ops, 't1r1_lang40-42-44_a0.3', 'projection scaling (1+0.3) along sv-minus-da/nb direction, layers 40,42,44')
    ops = []
    for l in (44, 45, 46, 47):
        m = dla[l].mean(0)
        top = torch.topk(m, 32).indices.tolist(); bot = torch.topk(-m, 32).indices.tolist()
        ops.append(neuron_scaling(model, M.format(l), top + bot, [1.5] * 32 + [0.5] * 32))
    c['t1r1_dla44-47_k32_s0.5'] = Edit(ops, 't1r1_dla44-47_k32_s0.5',
        'per layer 44-47: 32 neurons with highest mean direct-effect pair score x1.5, 32 lowest x0.5')
    g = store['grammar'][24]['out_diff_mean']
    c['t1r1_gram24_a0.5'] = Edit([projection_scaling(model, M.format(24), g, 0.5)], 't1r1_gram24_a0.5',
        'projection scaling (1+0.5) along mean good-minus-bad down_proj output contrast, layer 24')
    return c

run_round(TIER, ROUND, build)
