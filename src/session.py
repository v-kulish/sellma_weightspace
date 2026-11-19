"""Tools for agent-driven adjustment sessions: activation inspection and dev-only candidate evaluation.

Nothing here can read test data: loaders are called with split='dev' and an explicit tier.
"""
import json
import time
from collections import defaultdict

import numpy as np
import torch

from . import data
from .common import ROOT, read, write, now
from .edits import Edit, applied, get_module
from .evaluate import suite, TASKS
from .model import text_layers
from .scoring import Scorer, belebele_prompt

SWEDISH_SECONDARY = ['swedish_idioms_sv', 'scala_sv', 'belebele_swe_Latn']
NON_SWEDISH = [f'scala_{l}' for l in ('da', 'nb', 'en')] + \
              [f'belebele_{l}' for l in data.BELEBELE_LANGS if l != 'swe_Latn']


# ------------------------------------------------------------------ activation inspection
class MLPRecorder:
    """Records pooled MLP neuron activations (down_proj input) and down_proj outputs at chosen layers,
    mean-pooled over scored positions per sequence. Retains only pooled FP32 statistics."""

    def __init__(self, model, layers, keep=('neurons', 'out')):
        self.layers, self.keep, self.handles = layers, keep, []
        self.mask = None
        self.idx = None
        self.store = {l: {k: {} for k in keep} for l in layers}
        for l in layers:
            m = text_layers(model)[l].mlp.down_proj
            self.handles.append(m.register_forward_hook(self._hook(l)))

    def set_batch(self, idx, score_mask, attention):
        self.idx, self.mask = idx, score_mask.float()

    def _hook(self, l):
        def fn(mod, inp, out):
            if self.mask is None:
                return
            w = self.mask / self.mask.sum(1, keepdim=True)
            for k, x in (('neurons', inp[0]), ('out', out)):
                if k in self.keep:
                    pooled = torch.einsum('bt,btd->bd', w, x.float()).cpu()
                    for j, i in enumerate(self.idx):
                        self.store[l][k][i] = pooled[j]
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()

    def matrix(self, l, k, n):
        return torch.stack([self.store[l][k][i] for i in range(n)])


def pair_activations(scorer, rows, layers, keep=('neurons', 'out'), protocol='chat'):
    """Pooled activations for good/bad sentences of dev pairs (primary chat context)."""
    texts = [r[k] for r in rows for k in ('good', 'bad')]
    rec = MLPRecorder(scorer.model, layers, keep)
    try:
        if protocol == 'chat':
            pre = scorer.chat_prefix()
            seqs = [pre + scorer.tok.encode(t, add_special_tokens=False) for t in texts]
            scorer.score_sequences(seqs, [len(pre)] * len(seqs), hook=rec)
        else:
            scorer.score_sequences([scorer.raw_ids(t) for t in texts], [1] * len(texts), hook=rec)
    finally:
        rec.remove()
    out = {}
    for l in layers:
        out[l] = {}
        for k in keep:
            M = rec.matrix(l, k, len(texts))
            out[l][k] = dict(good=M[0::2], bad=M[1::2])
    return out


def text_activations(scorer, texts, layers, keep=('neurons',), chat=True):
    """Pooled activations over arbitrary texts (e.g. parallel Belebele dev passages), sentence tokens only."""
    rec = MLPRecorder(scorer.model, layers, keep)
    try:
        pre = scorer.chat_prefix() if chat else [scorer.tok.bos_token_id]
        seqs = [pre + scorer.tok.encode(t, add_special_tokens=False) for t in texts]
        scorer.score_sequences(seqs, [len(pre)] * len(seqs), batch_size=4, hook=rec)
    finally:
        rec.remove()
    return {l: {k: rec.matrix(l, k, len(texts)) for k in keep} for l in layers}


def logit_lens(model, tok, vec, k=12):
    """Top/bottom vocabulary tokens for a residual-space direction through final norm scale + tied unembedding."""
    lm = model.get_output_embeddings().weight
    norm = model.model.language_model.norm
    v = vec.to(lm.device).float()
    if getattr(norm, 'weight', None) is not None:  # Gemma4 RMSNorm: x_normed * weight (no +1 offset)
        v = v * norm.weight.float()
    s = lm.float() @ v
    top, bot = s.topk(k), (-s).topk(k)
    return dict(top=[tok.decode([int(i)]) for i in top.indices], bottom=[tok.decode([int(i)]) for i in bot.indices])


# ------------------------------------------------------------------ candidate evaluation on dev
def evaluate_candidates(scorer, tier, candidates, tasks=TASKS):
    """candidates: dict name -> Edit (or None for baseline). Returns {name: {task: summary}}, accounting."""
    out = {}
    for name, edit in candidates.items():
        t = time.monotonic(); before = scorer.accounting()
        summ, stats = suite(scorer, 'dev', name, edit=edit, tier=tier, quiet=True)
        acc = {k: scorer.accounting()[k] - before[k] for k in before}
        out[name] = dict(summaries=summ, edit_stats=stats, seconds=time.monotonic() - t, accounting=acc,
                         intended=edit.intended_stats(scorer.model) if edit else None,
                         description=edit.description if edit else 'unchanged model')
        print('CANDIDATE', name, f'{time.monotonic() - t:.0f}s', json.dumps(compact(summ)), flush=True)
    return out


def primary(task, s):
    if task.startswith('swe_blimp'):
        return s['macro_accuracy']
    if task.startswith('swedish_idioms'):
        return s['micro_accuracy']
    return s['accuracy']


def compact(summ):
    return {k: round(primary(k, s), 4) for k, s in summ.items()}


def pooled_change(base, cand, tasks):
    """Item-weighted mean accuracy change (pp) over a task group."""
    n = sum(base[t]['n'] for t in tasks)
    return 100 * sum((primary(t, cand[t]) - primary(t, base[t])) * base[t]['n'] for t in tasks) / n


def decision_row(base, cand):
    return dict(swe_blimp_pp=100 * (cand['swe_blimp_sv']['macro_accuracy'] - base['swe_blimp_sv']['macro_accuracy']),
                swe_blimp_margin=cand['swe_blimp_sv']['mean_margin'] - base['swe_blimp_sv']['mean_margin'],
                swedish_secondary_pp=pooled_change(base, cand, SWEDISH_SECONDARY),
                non_swedish_pp=pooled_change(base, cand, NON_SWEDISH),
                per_task_pp={t: 100 * (primary(t, cand[t]) - primary(t, base[t])) for t in base})


def eligible(row):
    """Pre-registered lock eligibility (reports/adjustments/PROTOCOL.md)."""
    return row['swe_blimp_pp'] > 0 and row['swedish_secondary_pp'] >= -2.0 and row['non_swedish_pp'] >= -2.0


def markdown_table(base, results):
    lines = ['| candidate | Swe-BLiMP Δpp | Swe-BLiMP Δmargin | Swedish secondary Δpp | non-Swedish Δpp | eligible | rel. norm |',
             '|---|---|---|---|---|---|---|']
    for name, r in results.items():
        row = decision_row(base, r['summaries'])
        rel = ', '.join(f"{s['relative_fro']:.4f}" for s in r['edit_stats']) if r['edit_stats'] else '0'
        lines.append(f"| {name} | {row['swe_blimp_pp']:+.2f} | {row['swe_blimp_margin']:+.4f} | "
                     f"{row['swedish_secondary_pp']:+.2f} | {row['non_swedish_pp']:+.2f} | "
                     f"{'yes' if eligible(row) else 'no'} | {rel} |")
    return '\n'.join(lines)


def per_task_table(base, results):
    tasks = list(base)
    head = '| candidate | ' + ' | '.join(t.replace('belebele_', 'bel_').replace('swedish_idioms_sv', 'idioms')
                                          .replace('swe_blimp_sv', 'blimp') for t in tasks) + ' |'
    lines = [head, '|' + '---|' * (len(tasks) + 1),
             '| baseline (abs) | ' + ' | '.join(f'{100 * primary(t, base[t]):.1f}' for t in tasks) + ' |']
    for name, r in results.items():
        lines.append(f'| {name} (Δpp) | ' + ' | '.join(f'{100 * (primary(t, r["summaries"][t]) - primary(t, base[t])):+.1f}'
                                                     for t in tasks) + ' |')
    return '\n'.join(lines)


def run_round(tier, rnd, build):
    """Load model, ensure the tier-N unchanged dev baseline, build candidates with build(model, tok, scorer, store),
    evaluate them on the full tier-N dev suite, save artifacts + tables. Returns the round record."""
    from .model import load
    model, tok = load()
    scorer = Scorer(model, tok)
    base = evaluate_candidates(scorer, tier, {'baseline': None})['baseline']['summaries']
    scorer.reset()
    store_path = ROOT / f'results/inspect/tier{tier}.pt'
    store = torch.load(store_path) if store_path.exists() else None
    t = time.monotonic()
    cands = build(model, tok, scorer, store)
    build_acc = scorer.accounting()
    (ROOT / f'artifacts/tier{tier}').mkdir(parents=True, exist_ok=True)
    for name, e in cands.items():
        e.save(ROOT / f'artifacts/tier{tier}/{name}.pt')
    scorer.reset()
    res = evaluate_candidates(scorer, tier, cands)
    rec = dict(utc=now(), tier=tier, round=rnd, build_seconds=time.monotonic() - t, build_accounting=build_acc,
               eval_accounting=scorer.accounting(), baseline=base,
               candidates={k: dict(v, identity=cands[k].identity(), artifact=f'artifacts/tier{tier}/{k}.pt',
                                   decision=decision_row(base, v['summaries'])) for k, v in res.items()},
               table=markdown_table(base, res), per_task=per_task_table(base, res))
    write(f'results/rounds/tier{tier}_round{rnd}.json', rec)
    print(rec['table']); print(rec['per_task'])
    return rec
