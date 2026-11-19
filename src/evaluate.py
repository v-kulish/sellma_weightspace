"""Dev-tier and sealed test evaluation over the full suite, with resumable, keyed caches.

Test sealing (DESIGN_DECISIONS.md #9): edited/random conditions can only be test-evaluated after
manifests/experiment_lock.json exists and lists the condition with the identical edit identity.
"""
import argparse
import time

import torch

from . import data
from .common import ROOT, read, write, now, digest, source_hash
from .edits import Edit, applied
from .scoring import (Scorer, eval_belebele, eval_scala, pair_metrics, scala_metrics, belebele_metrics)

TASKS = [('swe_blimp', 'sv'), ('swedish_idioms', 'sv')] + [('scala', l) for l in data.SCALA_LANGS] + \
        [('belebele', l) for l in data.BELEBELE_LANGS]


def split_hash(benchmark):
    return digest((ROOT / f'manifests/splits/{benchmark}.json').read_bytes())[:16]


def run_task(scorer, benchmark, lang, split, tier, condition):
    if benchmark in ('swe_blimp', 'swedish_idioms'):
        rows = data.load_pairs(benchmark, split, tier)
        recs = scorer.pairs(rows, condition)
        return recs, pair_metrics(recs)
    if benchmark == 'scala':
        rows = data.load_scala(lang, split, tier)
        recs = eval_scala(scorer, rows, data.load_scala(lang, 'fewshot'), condition)
        return recs, scala_metrics(recs)
    rows = data.load_belebele(lang, split, tier)
    recs = eval_belebele(scorer, rows, condition)
    return recs, belebele_metrics(recs)


def suite(scorer, split, condition, edit=None, tier=None, tasks=TASKS, out_dir=None, quiet=False):
    """Evaluate every task under one condition. Returns {task: summary}. Item records saved per task."""
    if split == 'test':
        check_unsealed(condition, edit)
    out_dir = out_dir or (f'results/test/{condition}' if split == 'test' else f'results/dev/tier{tier}/{condition}')
    summaries, stats = {}, []
    with applied(scorer.model, edit, stats):
        for bench, lang in tasks:
            key = dict(condition=condition, edit=edit.identity() if edit else None, split=split, tier=tier,
                       split_hash=split_hash(bench), source=source_hash())
            path = f'{out_dir}/{bench}_{lang}.json'
            p = ROOT / path
            if p.exists() and read(path)['key'] == key:
                summaries[f'{bench}_{lang}'] = read(path)['summary']
                continue
            scorer_before = scorer.accounting()
            t = time.monotonic()
            recs, summ = run_task(scorer, bench, lang, split, tier, condition)
            acc = {k: scorer.accounting()[k] - scorer_before[k] for k in scorer_before}
            write(path, dict(key=key, utc=now(), seconds=time.monotonic() - t, accounting=acc, summary=summ,
                             edit_stats=stats, records=recs))
            summaries[f'{bench}_{lang}'] = summ
            if not quiet:
                print(f'[{split} {condition}] {bench}_{lang}', {k: v for k, v in summ.items() if k != 'per_phenomenon'},
                      f'{time.monotonic() - t:.0f}s', flush=True)
    return summaries, stats


def check_unsealed(condition, edit):
    if condition == 'baseline' and edit is None:
        return  # user-requested early baseline exposure (disclosed)
    lock = ROOT / 'manifests/experiment_lock.json'
    if not lock.exists():
        raise PermissionError('Test set is sealed: no experiment lock yet (all three tiers must be locked).')
    L = read('manifests/experiment_lock.json')
    assert condition in L['conditions'], f'{condition} not in lock'
    assert L['conditions'][condition]['edit_identity'] == edit.identity(), 'edit differs from locked edit'


def headline(summaries):
    """Compact per-task primary metric dict."""
    out = {}
    for k, s in summaries.items():
        if k.startswith('swe_blimp'):
            out[k] = dict(macro_acc=s['macro_accuracy'], margin=s['mean_margin'], raw_macro=s.get('raw_macro_accuracy'), n=s['n'])
        elif k.startswith('swedish_idioms'):
            out[k] = dict(acc=s['micro_accuracy'], margin=s['mean_margin'], raw_macro=s.get('raw_macro_accuracy'), n=s['n'])
        elif k.startswith('scala'):
            out[k] = dict(acc=s['accuracy'], mcc=s['mcc'], invalid=s['invalid_rate'], n=s['n'])
        else:
            out[k] = dict(acc=s['accuracy'], margin=s['mean_gold_margin'], n=s['n'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['baseline', 'test'])
    ap.add_argument('--conditions', nargs='*', default=[])
    a = ap.parse_args()
    from .model import load
    model, tok = load()
    scorer = Scorer(model, tok)
    if a.stage == 'baseline':
        t = time.monotonic()
        for tier in (3,):  # tier-3 dev contains tiers 1-2 (nested); per-tier views derived in analysis
            suite(scorer, 'dev', 'baseline', tier=tier)
        suite(scorer, 'test', 'baseline')
        write('results/baseline_runtime.json', dict(utc=now(), seconds=time.monotonic() - t,
                                                    accounting=scorer.accounting(),
                                                    peak_gb=torch.cuda.max_memory_allocated() / 1e9))
    else:
        L = read('manifests/experiment_lock.json')
        for c in a.conditions or list(L['conditions']):
            t = time.monotonic(); scorer.reset()
            if L['conditions'][c]['artifact'] is None:   # no-op lock: identical to the unchanged model
                print('NO-OP condition', c, '-> reuses baseline test records in analysis', flush=True)
                continue
            edit = Edit.load(ROOT / L['conditions'][c]['artifact'])
            suite(scorer, 'test', c, edit=edit)
            write(f'results/test/{c}/_runtime.json', dict(utc=now(), seconds=time.monotonic() - t,
                                                          accounting=scorer.accounting()))


if __name__ == '__main__':
    main()
