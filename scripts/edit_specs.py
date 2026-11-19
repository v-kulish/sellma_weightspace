"""Compact, exact edit specifications <-> low-rank factor artifacts.

  export:  edit_specs.py export   -> artifacts/specs/<tier>/<name>.json for every artifacts/tier*/<name>.pt
  rebuild: edit_specs.py rebuild  -> recreate .pt files from specs + original checkpoint (CPU), verify edit identities
                                     against manifests/experiment_lock.json and round records.
Neuron-scaling ops store (module, neurons, scales); projection ops store (module, unit direction, alpha);
matched random controls store (random_of, seed). Only the original weights are needed to rebuild.
"""
import json
import pathlib
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import ROOT, read
from src.edits import Edit, matched_random

CKPT = ROOT / 'models/gemma-4-12B-it/model.safetensors'


def weight(f, module):
    return f.get_tensor(module + '.weight').float()


def describe_op(o, W):
    U, V = o['U'], o['V']
    nz = (V != 0)
    if bool(((nz.sum(0) == 1) & (V.abs().max(0).values == 1)).all()):
        idx = V.argmax(0)
        s = []
        for j, k in enumerate(idx.tolist()):
            col = W[:, k]
            x = float((U[:, j] @ col) / (col @ col)) + 1
            snapped = round(x * 4096) / 4096          # all scales used are dyadic (multiples of 1/4)
            s.append(snapped if abs(snapped - x) < 1e-4 else x)
        return dict(type='neuron_scaling', module=o['module'], neurons=idx.tolist(), scales=s)
    if U.shape[1] == 1:
        a = float(U[:, 0].norm()); d = U[:, 0] / a
        if torch.allclose(W.T @ d, V[:, 0], rtol=1e-3, atol=1e-5):
            return dict(type='projection_scaling', module=o['module'], alpha=a, direction=d.tolist())
        return dict(type='projection_scaling', module=o['module'], alpha=-a, direction=(-d).tolist())
    raise ValueError('unrecognised op')


def build_op(spec, W):
    if spec['type'] == 'neuron_scaling':
        idx = torch.tensor(spec['neurons'])
        s = torch.tensor(spec['scales'], dtype=torch.float32)
        V = torch.zeros(W.shape[1], len(idx)); V[idx, torch.arange(len(idx))] = 1
        return dict(module=spec['module'], U=W[:, idx] * (s - 1), V=V)
    d = torch.tensor(spec['direction'], dtype=torch.float32); d = d / d.norm()
    return dict(module=spec['module'], U=(spec['alpha'] * d)[:, None], V=(W.T @ d)[:, None])


def export():
    f = safe_open(str(CKPT), 'pt')
    for p in sorted((ROOT / 'artifacts').glob('tier*/*.pt')):
        e = Edit.load(p)
        out = ROOT / 'artifacts/specs' / p.parent.name / (p.stem + '.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        if '__random' in p.stem:
            spec = dict(name=e.name, description=e.description, random_of=p.stem.split('__random')[0], seed=43,
                        identity=e.identity())
        else:
            spec = dict(name=e.name, description=e.description, identity=e.identity(),
                        ops=[describe_op(o, weight(f, o['module'])) for o in e.ops])
        out.write_text(json.dumps(spec, indent=1))
        print('exported', out.relative_to(ROOT))


def rebuild(write_pt=True):
    f = safe_open(str(CKPT), 'pt')
    built, report = {}, []
    specs = sorted((ROOT / 'artifacts/specs').glob('tier*/*.json'), key=lambda p: ('__random' in p.stem, p.stem))
    for p in specs:
        s = json.loads(p.read_text())
        if 'random_of' in s:
            e = matched_random(built[(p.parent.name, s['random_of'])], seed=s['seed'])
        else:
            e = Edit([build_op(o, weight(f, o['module'])) for o in s['ops']], s['name'], s['description'])
        built[(p.parent.name, p.stem)] = e
        ok = e.identity() == s['identity']
        report.append((str(p.relative_to(ROOT)), ok))
        if write_pt:
            (ROOT / 'artifacts' / p.parent.name).mkdir(parents=True, exist_ok=True)
            target = ROOT / 'artifacts' / p.parent.name / (p.stem + '.pt')
            if not target.exists():
                e.save(target)
    lock = read('manifests/experiment_lock.json')
    for c, v in lock['conditions'].items():
        if v['artifact']:
            key = (pathlib.Path(v['artifact']).parent.name, pathlib.Path(v['artifact']).stem)
            print(c, 'identity matches lock:', built[key].identity() == v['edit_identity'])
    for r in report:
        print(('OK  ' if r[1] else 'DIFF') + ' ' + r[0])
    return report


if __name__ == '__main__':
    {'export': export, 'rebuild': rebuild}[sys.argv[1]]()
