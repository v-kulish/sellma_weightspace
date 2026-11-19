"""Figures (PDF) and LaTeX tables for the ACL-format report (CPU only; reads saved results).

Output goes to overleaf/figures and overleaf/tables, the report's LaTeX folder, which is not tracked in git.

Run: .venv/bin/python -m src.paper
"""
import collections
import glob
import hashlib
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from .common import ROOT, read
from .analyze import load_condition

OUT = ROOT / 'overleaf'
FIG, TAB = OUT / 'figures', OUT / 'tables'

# Reference categorical palette, fixed slot order: 1 blue = targeted edit, 2 orange = matched random, 3 aqua = edit - random.
EDIT, RAND, DIFF = '#2a78d6', '#eb6834', '#1baf7a'
EDIT_LIGHT, RAND_LIGHT = '#9ec5f4', '#f6b89f'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
TIER_BLUES = ['#86b6ef', '#2a78d6', '#184f95']   # ordinal ramp for tiers 1-3
TIERS = [1, 2, 3]
COL1, COL2 = 3.03, 6.3                           # ACL column / text width, inches

plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8.5,
    'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False,
    'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

A = read('results/analysis.json')['results']
TASKS_OTHER = [('swedish_idioms_sv', 'Idioms (sv)'), ('scala_sv', 'ScaLA sv'), ('scala_da', 'ScaLA da'),
               ('scala_nb', 'ScaLA nb'), ('scala_en', 'ScaLA en')] + \
              [(f'belebele_{l}', f'Belebele {l[:3]}') for l in
               ('swe_Latn', 'dan_Latn', 'nob_Latn', 'fin_Latn', 'isl_Latn', 'eng_Latn', 'zho_Hans')]
PHEN_SHORT = {'ADJ. NOUN AGR. NUM. PL.': 'ADJ.N.AGR. NUM.PL.', 'ADJ. NOUN AGR. NUM. SG.': 'ADJ.N.AGR. NUM.SG.'}


def cmp(task, key, metric='accuracy'):
    return A[task]['comparisons'][key][metric]


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf')
    if os.environ.get('PAPER_PREVIEW'):
        fig.savefig(f"{os.environ['PAPER_PREVIEW']}/{name}.png", dpi=170)
    plt.close(fig)


def grid(ax, axis='x'):
    ax.grid(axis=axis, color=GRID, lw=0.5); ax.set_axisbelow(True)


# ------------------------------------------------------------------ figures
def fig_primary():
    """Swe-BLiMP test: accuracy and margin deltas with CIs, per tier."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 1.9), sharey=True)
    rows = [('tier{t}_edit-baseline', 'edit $-$ unchanged', EDIT, 'o', -0.22),
            ('tier{t}_random-baseline', 'random $-$ unchanged', RAND, 's', 0.0),
            ('tier{t}_edit-tier{t}_random', 'edit $-$ random', DIFF, 'D', 0.22)]
    for ax, metric, xl in ((axes[0], 'accuracy', 'Δ macro accuracy (pp)'),
                           (axes[1], 'mean_margin', 'Δ mean margin (nats/token)')):
        for k, t in enumerate(TIERS):
            for key, lab, c, m, off in rows:
                d = cmp('swe_blimp_sv', key.format(t=t), metric)
                ax.errorbar(d['delta'], k + off, xerr=[[d['delta'] - d['lo']], [d['hi'] - d['delta']]], fmt=m,
                            ms=4.5, color=c, mec='white', mew=0.6, elinewidth=1.4, capsize=0,
                            label=lab if k == 0 else None)
        ax.axvline(0, color=INK2, lw=0.8)
        if metric == 'accuracy':
            ax.axvline(2, color=INK2, lw=0.7, ls=(0, (3, 2)))
            ax.text(1.95, -0.62, 'pre-registered +2 pp bar', fontsize=6.3, color=INK2, va='center', ha='right')
        ax.set_xlabel(xl); grid(ax)
    axes[0].set_yticks(range(3), ['Tier 1\n(26 pairs)', 'Tier 2\n(52 pairs)', 'Tier 3, primary\n(104 pairs)'])
    axes[0].set_ylim(2.45, -0.85)
    axes[0].legend(loc='upper center', bbox_to_anchor=(1.05, 1.22), ncol=3, handletextpad=0.3, columnspacing=1.2)
    save(fig, 'fig_primary')


def fig_dev_vs_test():
    """In-sample dev gain at lock vs held-out test gain, edit and matched random."""
    dev_e, dev_r = [], []
    for t in TIERS:
        lk = read(f'results/tiers/tier{t}_lock.json')
        dev_e.append(lk['edit_dev']['swe_blimp_pp']); dev_r.append(lk['random_dev']['swe_blimp_pp'])
    te = [cmp('swe_blimp_sv', f'tier{t}_edit-baseline')['delta'] for t in TIERS]
    tr = [cmp('swe_blimp_sv', f'tier{t}_random-baseline')['delta'] for t in TIERS]
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    x = np.arange(3); w = 0.19
    bars = [(dev_e, -1.5, EDIT_LIGHT, 'edit, dev (in-sample)'), (te, -0.5, EDIT, 'edit, test'),
            (dev_r, 0.5, RAND_LIGHT, 'random, dev'), (tr, 1.5, RAND, 'random, test')]
    for vals, o, c, lab in bars:
        b = ax.bar(x + o * w, vals, w * 0.9, color=c, label=lab)
        for xi, v in zip(x + o * w, vals):
            ax.text(xi, v + 0.4, f'{v:.1f}', ha='center', va='bottom', fontsize=5.8, color=INK2)
    ax.set_xticks(x, ['Tier 1', 'Tier 2', 'Tier 3'])
    ax.set_ylabel('Δ Swe-BLiMP accuracy (pp)'); ax.axhline(0, color=INK2, lw=0.6); grid(ax, 'y')
    ax.legend(ncol=2, loc='upper right', fontsize=6.3, handlelength=1.0, columnspacing=0.8)
    ax.set_ylim(0, 27)
    save(fig, 'fig_dev_vs_test')


def phen_acc(recs):
    by = collections.defaultdict(list)
    for r in recs:
        by[r['phenomenon']].append(r['correct'])
    return {p: 100 * np.mean(v) for p, v in by.items()}


def fig_phenomena():
    base = load_condition('baseline')['swe_blimp_sv']['records']
    b = phen_acc(base)
    raw = collections.defaultdict(list)
    for r in base:
        raw[r['phenomenon']].append(r['raw_correct'])
    phen = sorted(b, key=lambda p: b[p])
    cols, mat = [], []
    for t in TIERS:
        for kind in ('edit', 'random'):
            c = phen_acc(load_condition(f'tier{t}_{kind}')['swe_blimp_sv']['records'])
            cols.append(f'T{t}\n{kind}'); mat.append([c[p] - b[p] for p in phen])
    M = np.array(mat).T
    cmap = LinearSegmentedColormap.from_list('div', [RAND, '#f1f0ed', EDIT])
    fig, (ax0, ax) = plt.subplots(1, 2, figsize=(COL2, 4.3), gridspec_kw=dict(width_ratios=[1.0, 2.2], wspace=0.04),
                                  sharey=True)
    y = np.arange(len(phen))
    ax0.hlines(y, [100 * np.mean(raw[p]) for p in phen], [b[p] for p in phen], color=GRID, lw=1.6)
    ax0.plot([100 * np.mean(raw[p]) for p in phen], y, 'o', ms=3.6, color='#a9a8a2', label='raw BOS')
    ax0.plot([b[p] for p in phen], y, 'o', ms=3.6, color=INK, label='chat cue (primary)')
    ax0.axvline(50, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax0.set_xlim(0, 100); ax0.set_xlabel('unchanged accuracy (%)'); grid(ax0)
    ax0.set_yticks(y, [PHEN_SHORT.get(p, p) for p in phen], fontsize=6.3)
    ax0.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=6, handletextpad=0.1, columnspacing=0.6)
    im = ax.imshow(M, cmap=cmap, vmin=-25, vmax=25, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f'{v:+.0f}' if abs(v) >= 0.5 else '0', ha='center', va='center', fontsize=5.6,
                    color='white' if abs(v) > 15 else INK)
    ax.set_xticks(range(len(cols)), cols, fontsize=6.5); ax.xaxis.tick_top()
    ax.tick_params(axis='y', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for j in (1.5, 3.5):
        ax.axvline(j, color='white', lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Δ accuracy vs unchanged (pp)', fontsize=7); cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
    save(fig, 'fig_phenomena')


def fig_regression():
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.9), sharey=True, sharex=True)
    y = np.arange(len(TASKS_OTHER))
    for ax, t in zip(axes, TIERS):
        for kind, c, m, off in (('edit', EDIT, 'o', -0.16), ('random', RAND, 's', 0.16)):
            d = [cmp(task, f'tier{t}_{kind}-baseline') for task, _ in TASKS_OTHER]
            ax.errorbar([v['delta'] for v in d], y + off,
                        xerr=[[v['delta'] - v['lo'] for v in d], [v['hi'] - v['delta'] for v in d]],
                        fmt=m, ms=3.4, color=c, mec='white', mew=0.5, elinewidth=1.1, capsize=0,
                        label=f'{kind} $-$ unchanged')
        ax.axvline(0, color=INK2, lw=0.7); ax.axvline(-3, color=INK2, lw=0.7, ls=(0, (3, 2)))
        ax.axhspan(-0.5, 0.5, color='#f6f5f2', zorder=-1); ax.axhspan(4.5, 11.5, color='#f6f5f2', zorder=-1)
        ax.set_title(f'Tier {t}'); grid(ax); ax.set_xlim(-5.5, 3.5)
    axes[0].set_yticks(y, [n for _, n in TASKS_OTHER]); axes[0].invert_yaxis()
    axes[1].set_xlabel('Δ accuracy vs unchanged (pp), 95% paired cluster-bootstrap CI; dashed: $-$3 pp tolerance')
    axes[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.07), ncol=2, fontsize=6.5, handletextpad=0.2)
    save(fig, 'fig_regression')


def transitions(cond):
    b = {r['id']: r['correct'] for r in load_condition('baseline')['swe_blimp_sv']['records']}
    e = {r['id']: r['correct'] for r in load_condition(cond)['swe_blimp_sv']['records']}
    fixed = sum(1 for i in b if b[i] == 0 and e[i] == 1); broke = sum(1 for i in b if b[i] == 1 and e[i] == 0)
    return fixed, broke


def fig_transitions():
    conds = [(f'tier{t}_{k}', f'T{t} {k}') for t in TIERS for k in ('edit', 'random')]
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    for i, (c, lab) in enumerate(conds):
        f, b = transitions(c)
        col = EDIT if 'edit' in c else RAND
        ax.barh(i, f, 0.62, color=col); ax.barh(i, -b, 0.62, color=col, alpha=0.45)
        ax.text(f + 4, i, f'+{f}', va='center', fontsize=6.3, color=INK2)
        ax.text(-b - 4, i, f'−{b}', va='center', ha='right', fontsize=6.3, color=INK2)
        ax.text(232, i, f'net {f - b:+d}', va='center', fontsize=6.3, color=INK)
    ax.set_yticks(range(len(conds)), [l for _, l in conds]); ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.7); ax.set_xlim(-200, 272); ax.set_xticks([-200, -100, 0, 100, 200]); grid(ax)
    ax.set_xlabel('test pairs newly broken (left) / newly corrected (right)')
    save(fig, 'fig_transitions')


def fig_margin_bins():
    """Mean margin change as a function of the unchanged model's margin (Swe-BLiMP test)."""
    base = {r['id']: r['margin'] for r in load_condition('baseline')['swe_blimp_sv']['records']}
    edges = np.array([-np.inf, -1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, np.inf])
    labels = ['<−1', '−1…−.5', '−.5…−.25', '−.25…−.1', '|m|<.1', '.1….25', '.25….5', '.5…1', '>1']
    ids = list(base); bm = np.array([base[i] for i in ids]); bins = np.digitize(bm, edges[1:-1])
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    x = np.arange(len(labels))
    for t, c in zip(TIERS, TIER_BLUES):
        e = {r['id']: r['margin'] for r in load_condition(f'tier{t}_edit')['swe_blimp_sv']['records']}
        d = np.array([e[i] - base[i] for i in ids])
        ax.plot(x, [d[bins == k].mean() for k in range(len(labels))], '-o', ms=3, lw=1.4, color=c, label=f'tier {t} edit')
    rr = []
    for t in TIERS:
        e = {r['id']: r['margin'] for r in load_condition(f'tier{t}_random')['swe_blimp_sv']['records']}
        rr.append(np.array([e[i] - base[i] for i in ids]))
    rr = np.mean(rr, axis=0)
    ax.plot(x, [rr[bins == k].mean() for k in range(len(labels))], '-s', ms=3, lw=1.2, color=RAND,
            label='random (mean of tiers)')
    counts = [int((bins == k).sum()) for k in range(len(labels))]
    for xi, n in zip(x, counts):
        ax.text(xi, -0.075, f'n={n}', ha='center', fontsize=5.2, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6); ax.axvspan(3.5, 4.5, color='#f6f5f2', zorder=-1)
    ax.set_xticks(x, labels, rotation=35, fontsize=6); ax.set_ylim(-0.09, 0.30)
    ax.set_xlabel('unchanged-model margin bin (nats/token)'); ax.set_ylabel('mean Δ margin'); grid(ax, 'y')
    ax.legend(fontsize=6, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.4, columnspacing=0.8)
    save(fig, 'fig_margin_bins')


def fig_tokens(S):
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 1.6))
    x = np.arange(3)
    out = [s['agent']['output'] / 1e3 for s in S]; th = [s['agent']['thinking'] / 1e3 for s in S]
    axes[0].bar(x, out, 0.6, color=EDIT_LIGHT, label='output'); axes[0].bar(x, th, 0.6, color=EDIT, label='of which thinking')
    axes[0].set_title('Agent output tokens (k)'); axes[0].legend(fontsize=5.8, loc='upper right'); axes[0].set_ylim(0, 45)
    fw = [(s['model_forward']['input_tokens'] + s['model_forward']['generated_tokens']) / 1e6 for s in S]
    axes[1].bar(x, fw, 0.6, color=INK2); axes[1].set_title('Gemma forward tokens (M)'); axes[1].set_ylim(0, 1.8)
    for ax, vals in zip(axes, (out, fw)):
        ax.set_xticks(x, ['T1', 'T2', 'T3']); grid(ax, 'y')
        for xi, v in zip(x, vals):
            ax.text(xi, v, f'{v:.1f}' if v < 10 else f'{v:.0f}', ha='center', va='bottom', fontsize=6, color=INK2)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig_tokens')


# ------------------------------------------------------------------ tables
def tex(s):
    return str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('#', r'\#')


def ci(d, f='{:+.2f}'):
    return f'{f.format(d["delta"])} [{f.format(d["lo"])}, {f.format(d["hi"])}]'


def write(name, body):
    body = re.sub(r'(?<![\w-])-(?=\d)', '$-$', body)   # typographic minus for negative numbers
    lines, out = body.strip().split('\n'), []
    for i, l in enumerate(lines):   # thin rule (\rowsep, defined in main.tex) between consecutive data rows
        out.append(l)
        nxt = lines[i + 1].lstrip() if i + 1 < len(lines) else '\\'
        if l.rstrip().endswith('\\\\') and not nxt.startswith('\\') and not l.lstrip().startswith('\\multicolumn'):
            out.append('\\rowsep')
    (TAB / f'{name}.tex').write_text('\n'.join(out) + '\n')


def tab_primary():
    L = [r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
         r' & Tier 1 & Tier 2 & Tier 3 (primary) \\', r'\midrule']
    dev = [read(f'results/tiers/tier{t}_lock.json') for t in TIERS]
    L.append(r'Dev Swe-BLiMP $\Delta$ at lock (in-sample) & ' +
             ' & '.join(f'{d["edit_dev"]["swe_blimp_pp"]:+.1f}' for d in dev) + r' \\')
    L.append(r'Dev $\Delta$, matched random & ' + ' & '.join(f'{d["random_dev"]["swe_blimp_pp"]:+.1f}' for d in dev) + r' \\')
    L.append(r'\midrule')
    rows = [('Edit $-$ unchanged, accuracy (pp)', 'edit-baseline', 'accuracy', '{:+.2f}'),
            ('Random $-$ unchanged, accuracy (pp)', 'random-baseline', 'accuracy', '{:+.2f}'),
            ('Edit $-$ random, accuracy (pp)', 'edit-tier{t}_random', 'accuracy', '{:+.2f}'),
            ('Edit $-$ unchanged, margin (nats/tok)', 'edit-baseline', 'mean_margin', '{:+.3f}'),
            ('Edit $-$ random, margin (nats/tok)', 'edit-tier{t}_random', 'mean_margin', '{:+.3f}'),
            ('Raw-BOS accuracy, edit $-$ unchanged', 'edit-baseline', 'raw_accuracy', '{:+.2f}')]
    for lab, key, m, f in rows:
        cells = []
        for t in TIERS:
            k = f'tier{t}_' + key.format(t=t)
            cells.append(r'\makecell{' + ci(cmp('swe_blimp_sv', k, m), f).replace(' [', r'\\ \scriptsize[') + '}')
        L.append(f'{lab} & ' + ' & '.join(cells) + r' \\')
    L.append(r'\midrule')
    L.append(r'Edited macro accuracy (unchanged 69.08) & ' + ' & '.join(
        f'{100 * A["swe_blimp_sv"]["point"][f"tier{t}_edit"]["accuracy"]:.2f}' for t in TIERS) + r' \\')
    L.append(r'Verdict (pre-registered rule) & \multicolumn{3}{c}{inconclusive: every accuracy CI crosses zero} \\')
    L += [r'\bottomrule', r'\end{tabular}']
    write('tab_primary', '\n'.join(L))


def tab_other():
    conds = ['baseline'] + [f'tier{t}_{k}' for t in TIERS for k in ('edit', 'random')]
    L = [r'\begin{tabular}{@{}lrr' + 'r' * 6 + '@{}}', r'\toprule',
         r'Task (metric) & $n$ & Unch. & \multicolumn{2}{c}{Tier 1} & \multicolumn{2}{c}{Tier 2} & \multicolumn{2}{c}{Tier 3} \\',
         r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}',
         r' & & & edit & rand. & edit & rand. & edit & rand. \\', r'\midrule']
    rows = [('swe_blimp_sv', 'accuracy', 'Swe-BLiMP (macro acc.)'), ('swe_blimp_sv', 'raw_accuracy', 'Swe-BLiMP (raw BOS)'),
            ('swe_blimp_sv', 'mean_margin', 'Swe-BLiMP (margin)'),
            ('swedish_idioms_sv', 'accuracy', 'Idioms (acc.)'), ('swedish_idioms_sv', 'raw_accuracy', 'Idioms (raw BOS)')]
    for l in ('sv', 'da', 'nb', 'en'):
        rows += [(f'scala_{l}', 'accuracy', f'ScaLA {l} (acc.)'), (f'scala_{l}', 'mcc', f'ScaLA {l} (MCC$\\times$100)')]
    rows += [(f'belebele_{l}', 'accuracy', f'Belebele {l[:3]} (acc.)') for l in
             ('swe_Latn', 'dan_Latn', 'nob_Latn', 'fin_Latn', 'isl_Latn', 'eng_Latn', 'zho_Hans')]
    for task, m, lab in rows:
        P = A[task]['point']; base = P['baseline'][m]
        f = (lambda v: f'{v:+.3f}') if m == 'mean_margin' else (lambda v: f'{100 * v:.2f}')
        cells = [f(base)]
        for c in conds[1:]:
            v = P[c][m]; s = f(v)
            if abs(v - base) > 1e-9:
                s = r'\textbf{' + s + '}'
            cells.append(s)
        L.append(f'{lab} & {A[task]["n"]} & ' + ' & '.join(cells) + r' \\')
        if task.startswith('swedish') and m == 'raw_accuracy' or (task == 'scala_en' and m == 'mcc'):
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}']
    write('tab_other', '\n'.join(L))


def candidate_rows():
    rows = []
    locks = {t: read(f'results/tiers/tier{t}_lock.json')['candidate'] for t in TIERS}
    for f in sorted(glob.glob(str(ROOT / 'results/rounds/tier*_round*.json'))):
        r = json.loads(open(f).read())
        for n, c in sorted(r['candidates'].items()):
            d = c['decision']; st = c['edit_stats']
            layers = sorted(int(s['module'].split('.')[3]) for s in st)
            desc = c['description']
            if 'neuron' in desc:
                mm = re.search(r'per layer (\d+) highest[- ]score neurons x([\d.]+), (\d+) lowest x([\d.]+)', desc) or \
                     re.search(r'(\d+) neurons with highest mean direct-effect pair score x([\d.]+), (\d+) lowest x([\d.]+)', desc)
                kind = f'neurons {mm.group(1)}$\\times${mm.group(2)} / {mm.group(3)}$\\times${mm.group(4)}'
                if 'ranked by t' in desc:
                    kind += ', $t$-ranked'
                if '(blimp)' in desc:
                    kind += ', BLiMP-only'
            elif 'contrast' in desc:
                kind = r'proj. $+0.5\,gg^\top W$ (grammar contrast)'
            else:
                a = re.search(r'\(1\+([\d.]+)\)', desc).group(1)
                kind = f'proj. $+{a}\\,dd^\\top W$ (sv$-$da/nb)'
            lay = (f'{layers[0]}' if len(layers) == 1 else
                   f'{layers[0]}--{layers[-1]}' if layers == list(range(layers[0], layers[-1] + 1)) else
                   ','.join(map(str, layers)) if len(layers) <= 3 else f'{layers[0]}--{layers[-1]} even')
            rel = [s['relative_fro'] * 100 for s in st]
            eligible = d['swe_blimp_pp'] > 0 and d['swedish_secondary_pp'] >= -2 and d['non_swedish_pp'] >= -2
            rows.append(dict(tier=r['tier'], round=r['round'], name=n, kind=kind, layers=lay,
                             rel=f'{min(rel):.1f}' if max(rel) - min(rel) < 0.05 else f'{min(rel):.1f}--{max(rel):.1f}',
                             d=d, eligible=eligible, locked=(n == locks[r['tier']])))
    return rows


def tab_candidates():
    L = [r'\begin{tabular}{@{}clllrrrrrc@{}}', r'\toprule',
         r'T/R & Candidate & Edit (down\_proj) & Layers & rel.\,$\|\Delta W\|$\,\% & BLiMP & margin & sv sec. & non-sv & elig. \\',
         r'\midrule']
    prev = None
    for r in candidate_rows():
        if prev and r['tier'] != prev:
            L.append(r'\midrule')
        prev = r['tier']
        d = r['d']; nm = tex(r['name'])
        if r['locked']:
            nm = r'\textbf{' + nm + '}'
        L.append(f"{r['tier']}/{r['round']} & \\texttt{{\\scriptsize {nm}}} & {r['kind']} & {r['layers']} & {r['rel']} & "
                 f"{d['swe_blimp_pp']:+.1f} & {d['swe_blimp_margin']:+.3f} & {d['swedish_secondary_pp']:+.1f} & "
                 f"{d['non_swedish_pp']:+.1f} & {'yes' if r['eligible'] else 'no'} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    write('tab_candidates', '\n'.join(L))


def round_agent_tokens():
    out = {}
    for f in sorted(glob.glob(str(ROOT / 'reports/adjustments/tier*_round*.md'))):
        txt = open(f).read()
        m = re.search(r'\*\*Agent \(Claude Opus 5\.5\) tokens.*?\n\n\|.*?\n\|.*?\n\| ([\d,]+) \| ([\d,]+) \| ([\d,]+) \|', txt, re.S)
        g = re.search(r'\*\*total\*\* \| \| \| \| ([\d,]+) \(in\+gen\)', txt)
        key = re.search(r'tier(\d)_round(\d)', f).groups()
        out[key] = dict(calls=int(m.group(1)), output=int(m.group(2).replace(',', '')),
                        thinking=int(m.group(3).replace(',', '')), gemma=int(g.group(1).replace(',', '')))
    return out


def tab_tokens(S):
    R = round_agent_tokens()
    L = [r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
         r'Session & Calls & Agent output & Thinking & Gemma tokens & Dev items \\', r'\midrule']
    for t, s in zip(TIERS, S):
        for rd in (1, 2):
            v = R[(str(t), str(rd))]
            L.append(f'Tier {t}, round {rd} & {v["calls"]} & {v["output"]:,} & {v["thinking"]:,} & {v["gemma"]:,} & \\\\')
        a, fw = s['agent'], s['model_forward']
        L.append(f'\\textit{{Tier {t} total}} & \\textit{{{a["api_calls"]}}} & \\textit{{{a["output"]:,}}} & '
                 f'\\textit{{{a["thinking"]:,}}} & \\textit{{{fw["input_tokens"] + fw["generated_tokens"]:,}}} & '
                 f'\\textit{{{s["exposed"]["total_labelled_items"]}}} \\\\')
        if t != 3:
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}']
    write('tab_tokens', re.sub(r'(?<=\d),(?=\d{3})', '{,}', '\n'.join(L)))


def tab_examples(n=2):
    """Deterministic examples (SHA-256 of item id) of corrected and newly broken test pairs for the tier-3 edit."""
    src = pd.read_csv(ROOT / 'data/swe_blimp/swe_blimp_v1.csv', sep=';')
    b = {r['id']: r for r in load_condition('baseline')['swe_blimp_sv']['records']}
    e = {r['id']: r for r in load_condition('tier3_edit')['swe_blimp_sv']['records']}
    L = [r'\begin{tabular}{@{}p{0.15\textwidth}p{0.33\textwidth}p{0.33\textwidth}rr@{}}', r'\toprule',
         r'Phenomenon & Grammatical & Ungrammatical & $m_0$ & $m_3$ \\', r'\midrule']
    for cls, cond in (('Corrected', (0, 1)), ('Newly broken', (1, 0)), ('Still wrong', (0, 0))):
        ids = sorted([i for i in b if (b[i]['correct'], e[i]['correct']) == cond],
                     key=lambda x: hashlib.sha256(x.encode()).hexdigest())[:n]
        L.append(r'\multicolumn{5}{@{}l}{\textit{' + cls + '}} \\\\')
        for i in ids:
            row = src.iloc[int(i.split('-')[1])]
            assert row['phenomenon'] == b[i]['phenomenon']
            L.append(f"{tex(row['phenomenon'])} & {tex(row['sentence_good'])} & {tex(row['sentence_bad'])} & "
                     f"{b[i]['margin']:+.2f} & {e[i]['margin']:+.2f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    write('tab_examples', '\n'.join(L))


def tab_baseline_phen():
    base = load_condition('baseline')['swe_blimp_sv']['records']
    b = phen_acc(base)
    raw = collections.defaultdict(list); n = collections.Counter(); mg = collections.defaultdict(list)
    for r in base:
        raw[r['phenomenon']].append(r['raw_correct']); n[r['phenomenon']] += 1; mg[r['phenomenon']].append(r['margin'])
    E = {t: phen_acc(load_condition(f'tier{t}_edit')['swe_blimp_sv']['records']) for t in TIERS}
    L = [r'\begin{tabular}{@{}lrrrrrrr@{}}', r'\toprule',
         r'Phenomenon & $n$ & chat & raw & margin & $\Delta$T1 & $\Delta$T2 & $\Delta$T3 \\', r'\midrule']
    for p in sorted(b):
        L.append(f'{tex(p)} & {n[p]} & {b[p]:.1f} & {100 * np.mean(raw[p]):.1f} & {np.mean(mg[p]):+.2f} & ' +
                 ' & '.join(f'{E[t][p] - b[p]:+.1f}' for t in TIERS) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    write('tab_baseline_phen', '\n'.join(L))


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    S = read('results/session_summary.json')
    fig_primary(); fig_dev_vs_test(); fig_phenomena(); fig_regression(); fig_transitions()
    tab_primary(); tab_other(); tab_candidates(); tab_tokens(S); tab_examples(); tab_baseline_phen()
    print(sorted(p.name for p in FIG.iterdir()), sorted(p.name for p in TAB.iterdir()))


if __name__ == '__main__':
    main()
