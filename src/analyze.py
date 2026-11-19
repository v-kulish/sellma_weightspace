"""Paired cluster-bootstrap statistics and tables from cached item records (no model needed).

Swe-BLiMP: resample identifiable groups within phenomenon-signature strata, recompute macro accuracy.
Idioms: resample groups, micro accuracy. ScaLA: resample relation groups, recompute accuracy and MCC
(invalid = third predicted class). Belebele: resample passages with draws shared across languages.
"""
import collections
import json
import math

import numpy as np

from .common import ROOT, read, write, now

B = 10000
SEED = 2026
PRIMARY_GAIN_PP = 2.0
BELEBELE_TOL_PP = -3.0


def load_condition(cond, split='test', tier=None):
    base = ROOT / ('results/test/' + cond if split == 'test' else f'results/dev/tier{tier}/{cond}')
    out = {}
    for p in sorted(base.glob('*.json')):
        if p.name.startswith('_'):
            continue
        d = json.loads(p.read_text())
        out[p.stem] = d
    return out


def weights_for(groups, strata, rng):
    """B x G multiplicity matrix: resample groups with replacement within each stratum."""
    G = len(groups)
    W = np.zeros((B, G), dtype=np.float32)
    by = collections.defaultdict(list)
    for gi, s in enumerate(strata):
        by[s].append(gi)
    for s in sorted(by):
        idx = np.array(by[s])
        draws = rng.integers(0, len(idx), size=(B, len(idx)))
        for col in range(len(idx)):
            np.add.at(W, (np.arange(B), idx[draws[:, col]]), 1)
    return W


def mcc_from_counts(cm):
    """cm[..., gold(2), pred(3)] -> multiclass MCC (sklearn convention)."""
    s = cm.sum(axis=(-1, -2))
    c = cm[..., 0, 0] + cm[..., 1, 1]
    t = cm.sum(-1)                      # gold totals (..., 2)
    p = cm.sum(-2)                      # pred totals (..., 3)
    pt = (p[..., :2] * t).sum(-1)
    num = c * s - pt
    den = np.sqrt((s ** 2 - (p ** 2).sum(-1)) * (s ** 2 - (t ** 2).sum(-1)))
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)


def task_stats(task, recs_by_cond, rng_seed):
    """Point estimates and bootstrap replicate arrays for each condition on one task."""
    conds = list(recs_by_cond)
    ids = [r['id'] for r in recs_by_cond[conds[0]]]
    for c in conds:
        assert [r['id'] for r in recs_by_cond[c]] == ids, f'item alignment broken for {task}/{c}'
    first = recs_by_cond[conds[0]]
    groups = sorted({r['group'] for r in first})
    gi = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(rng_seed)
    out = {}
    if task.startswith('swe_blimp') or task.startswith('swedish_idioms'):
        macro = task.startswith('swe_blimp')
        phen = sorted({r['phenomenon'] for r in first}) if macro else ['all']
        pi = {p: i for i, p in enumerate(phen)}
        sig = collections.defaultdict(set)
        for r in first:
            sig[r['group']].add(r['phenomenon'] if macro else 'all')
        strata = ['|'.join(sorted(sig[g])) for g in groups]
        W = weights_for(groups, strata, rng)
        for c in conds:
            for key in (['correct', 'raw_correct'] if 'raw_correct' in first[0] else ['correct']):
                C = np.zeros((len(groups), len(phen))); N = np.zeros_like(C)
                for r in recs_by_cond[c]:
                    C[gi[r['group']], pi[r['phenomenon'] if macro else 'all']] += r[key]
                    N[gi[r['group']], pi[r['phenomenon'] if macro else 'all']] += 1
                point = np.mean(C.sum(0) / N.sum(0))
                reps = np.mean((W @ C) / np.maximum(W @ N, 1e-9), axis=1)
                name = 'accuracy' if key == 'correct' else 'raw_accuracy'
                out.setdefault(c, {})[name] = (float(point), reps)
            m = np.array([r['margin'] for r in recs_by_cond[c]])
            out[c]['mean_margin'] = (float(m.mean()), None)
    elif task.startswith('scala'):
        W = weights_for(groups, ['all'] * len(groups), rng)
        for c in conds:
            CM = np.zeros((len(groups), 2, 3)); A = np.zeros(len(groups)); N = np.zeros(len(groups))
            for r in recs_by_cond[c]:
                CM[gi[r['group']], r['gold'], r['prediction'] if r['prediction'] >= 0 else 2] += 1
                A[gi[r['group']]] += r['correct']; N[gi[r['group']]] += 1
            cmB = np.einsum('bg,gij->bij', W, CM)
            out[c] = dict(accuracy=(float(A.sum() / N.sum()), (W @ A) / (W @ N)),
                          mcc=(float(mcc_from_counts(CM.sum(0))), mcc_from_counts(cmB)),
                          invalid_rate=(float(np.mean([r['prediction'] == -1 for r in recs_by_cond[c]])), None))
    else:  # belebele: passages; same seed for every language -> identical passage draws (shared ordering)
        W = weights_for(groups, ['all'] * len(groups), rng)
        for c in conds:
            A = np.zeros(len(groups)); N = np.zeros(len(groups))
            for r in recs_by_cond[c]:
                A[gi[r['group']]] += r['correct']; N[gi[r['group']]] += 1
            out[c] = dict(accuracy=(float(A.sum() / N.sum()), (W @ A) / (W @ N)))
    return out, len(ids), len(groups)


def diff(stats, a, b, metric, scale=100):
    pa, ra = stats[a][metric]; pb, rb = stats[b][metric]
    d = (ra - rb) * scale
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=(pa - pb) * scale, lo=float(lo), hi=float(hi))


def analyze(conditions, comparisons, split='test', tier=None):
    data = {c: load_condition(c, split, tier) for c in conditions}
    tasks = sorted(set.intersection(*[set(d) for d in data.values()]))
    res = {}
    for k, task in enumerate(tasks):
        recs = {c: data[c][task]['records'] for c in conditions}
        seed = SEED if task.startswith('belebele') else SEED + k  # belebele: identical passage draws across languages
        stats, n, ng = task_stats(task, recs, seed)
        metrics = [m for m, v in stats[conditions[0]].items() if v[1] is not None]
        res[task] = dict(n=n, groups=ng,
                         point={c: {m: v[0] for m, v in stats[c].items()} for c in conditions},
                         comparisons={f'{a}-{b}': {m: diff(stats, a, b, m, 100) for m in metrics}
                                      for a, b in comparisons if a in stats and b in stats})
    return res


def verdicts(res, comparisons):
    out = {}
    for a, b in comparisons:
        key = f'{a}-{b}'
        v = {}
        sb = res.get('swe_blimp_sv', {}).get('comparisons', {}).get(key)
        if sb:
            d = sb['accuracy']
            v['swe_blimp'] = ('meaningful gain' if d['delta'] >= PRIMARY_GAIN_PP and d['lo'] > 0 else
                              'positive, CI above zero but < +2 pp' if d['lo'] > 0 else
                              'inconclusive' if d['hi'] > 0 else 'loss (CI below zero)')
        for t, r in res.items():
            if t.startswith('belebele') and key in r['comparisons']:
                d = r['comparisons'][key]['accuracy']
                v[t] = ('loss beyond tolerance' if d['hi'] < BELEBELE_TOL_PP else
                        'preserved within -3 pp' if d['lo'] > BELEBELE_TOL_PP else
                        ('observed change below -3 pp' if d['delta'] < BELEBELE_TOL_PP else
                         'insufficient evidence of preservation'))
        out[key] = v
    return out


def fmt(d):
    return f"{d['delta']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"


def tables(res, conditions, comparisons, title):
    L = [f'## {title}', '', '### Point estimates (%; ScaLA MCC ×100)', '',
         '| task | n (groups) | ' + ' | '.join(conditions) + ' |', '|---|---|' + '---|' * len(conditions)]
    for t, r in res.items():
        for m in r['point'][conditions[0]]:
            if m in ('invalid_rate',) and all(r['point'][c][m] == 0 for c in conditions):
                continue
            vals = [r['point'][c][m] for c in conditions]
            s = ' | '.join((f'{100 * v:.2f}' if m != 'mean_margin' else f'{v:+.4f}') for v in vals)
            L.append(f'| {t} {m} | {r["n"]} ({r["groups"]}) | {s} |')
    L += ['', '### Paired differences, pp [95% cluster-bootstrap CI]', '',
          '| task | metric | ' + ' | '.join(f'{a} − {b}' for a, b in comparisons) + ' |',
          '|---|---|' + '---|' * len(comparisons)]
    for t, r in res.items():
        for m in next(iter(r['comparisons'].values()), {}):
            L.append(f'| {t} | {m} | ' + ' | '.join(fmt(r['comparisons'][f'{a}-{b}'][m]) if f'{a}-{b}' in r['comparisons']
                                                    else '—' for a, b in comparisons) + ' |')
    return '\n'.join(L)


def main():
    lock = read('manifests/experiment_lock.json') if (ROOT / 'manifests/experiment_lock.json').exists() else None
    conds = ['baseline'] + (sorted(lock['conditions']) if lock else [])
    conds = [c for c in conds if (ROOT / f'results/test/{c}').exists()]
    comps = []
    for t in (1, 2, 3):
        e, r = f'tier{t}_edit', f'tier{t}_random'
        if e in conds:
            comps.append((e, 'baseline'))
            if r in conds:
                comps += [(e, r), (r, 'baseline')]
    res = analyze(conds, comps)
    write('results/analysis.json', dict(utc=now(), conditions=conds, comparisons=comps, bootstrap=B, seed=SEED,
                                        results=res, verdicts=verdicts(res, comps)))
    (ROOT / 'results/tables.md').write_text(tables(res, conds, comps, 'Test results') + '\n')
    print((ROOT / 'results/tables.md').read_text()[:3000])


if __name__ == '__main__':
    main()
