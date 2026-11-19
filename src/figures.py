"""Figures and deterministic examples from cached test records (no model needed)."""
import collections
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT, read
from .analyze import load_condition

TIERS = [1, 2, 3]
COLORS = {'edit': '#2a6fdb', 'random': '#9a9a9a'}


def per_phenomenon_plot():
    base = load_condition('baseline')['swe_blimp_sv']['records']
    phen = sorted({r['phenomenon'] for r in base})
    acc = lambda recs, p: 100 * np.mean([r['correct'] for r in recs if r['phenomenon'] == p])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y = np.arange(len(phen))
    for k, t in enumerate(TIERS):
        for kind, off in (('edit', -0.25 + 0.17 * k), ('random', 0.0)):
            if kind == 'random' and t != 3:
                continue
            c = load_condition(f'tier{t}_{kind}')['swe_blimp_sv']['records']
            d = [acc(c, p) - acc(base, p) for p in phen]
            ax.barh(y + (off if kind == 'edit' else 0.27), d, height=0.15,
                    color=COLORS[kind] if kind == 'random' else ['#9ec5ff', '#5b9cf0', '#1f5fc4'][k],
                    label=f'tier {t} {kind}')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(y, [f'{p} ({acc(base, p):.0f}%)' for p in phen], fontsize=7)
    ax.set_xlabel('Δ accuracy vs unchanged, pp (Swe-BLiMP test; baseline accuracy in parentheses)', fontsize=8)
    ax.invert_yaxis(); ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout(); fig.savefig(ROOT / 'reports/phenomenon_changes.png', dpi=160); plt.close(fig)


def language_plot():
    A = read('results/analysis.json')['results']
    tasks = ['swe_blimp_sv', 'swedish_idioms_sv'] + [f'scala_{l}' for l in ('sv', 'da', 'nb', 'en')] + \
            [f'belebele_{l}' for l in ('swe_Latn', 'dan_Latn', 'nob_Latn', 'fin_Latn', 'isl_Latn', 'eng_Latn', 'zho_Hans')]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(tasks))
    for k, t in enumerate(TIERS):
        for kind in ('edit', 'random'):
            key = f'tier{t}_{kind}-baseline'
            d = [A[x]['comparisons'][key]['accuracy'] for x in tasks]
            off = -0.3 + 0.2 * k + (0.08 if kind == 'random' else 0)
            ax.errorbar([v['delta'] for v in d], y + off, xerr=[[v['delta'] - v['lo'] for v in d], [v['hi'] - v['delta'] for v in d]],
                        fmt='o' if kind == 'edit' else 'x', ms=3.5, lw=0.8, capsize=1.5,
                        color=['#9ec5ff', '#5b9cf0', '#1f5fc4'][k] if kind == 'edit' else ['#cfcfcf', '#a5a5a5', '#6f6f6f'][k],
                        label=f'tier {t} {kind}')
    ax.axvline(0, color='k', lw=0.8); ax.axvline(-3, color='#c33', lw=0.6, ls='--')
    ax.set_yticks(y, tasks, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel('Δ accuracy vs unchanged, pp (95% paired cluster-bootstrap CI); dashed: −3 pp tolerance', fontsize=8)
    ax.legend(fontsize=6.5, ncol=3, loc='lower left')
    fig.tight_layout(); fig.savefig(ROOT / 'reports/task_changes.png', dpi=160); plt.close(fig)


def examples(condition='tier3_edit', task='swe_blimp_sv', n=4):
    """Deterministic rule: within each outcome class, sort item ids by SHA-256 and take the first n."""
    b = {r['id']: r for r in load_condition('baseline')[task]['records']}
    e = {r['id']: r for r in load_condition(condition)[task]['records']}
    cls = collections.defaultdict(list)
    for i in b:
        k = {(0, 1): 'corrected', (1, 0): 'newly broken', (1, 1): 'unchanged correct', (0, 0): 'unchanged wrong'}[(b[i]['correct'], e[i]['correct'])]
        cls[k].append(i)
    out = {}
    for k, ids in cls.items():
        pick = sorted(ids, key=lambda x: hashlib.sha256(x.encode()).hexdigest())[:n]
        out[k] = dict(count=len(ids), examples=[dict(id=i, phenomenon=b[i].get('phenomenon'), baseline_margin=b[i]['margin'],
                                                     edited_margin=e[i]['margin']) for i in pick])
    return out


def main():
    per_phenomenon_plot(); language_plot()
    ex = {t: examples('tier3_edit', t) for t in ('swe_blimp_sv', 'swedish_idioms_sv')}
    (ROOT / 'results/examples.json').write_text(json.dumps(ex, indent=1))
    print(json.dumps({t: {k: v['count'] for k, v in d.items()} for t, d in ex.items()}))


if __name__ == '__main__':
    main()
