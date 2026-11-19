"""Low-rank weight edits on text-decoder MLP matrices, exact restoration, matched random controls.

Every edit is a list of ops {module, U[out,r], V[in,r]} meaning W <- W + U @ V.T (float32).
This covers rank-one projection scaling (U = a*d, V = W.T d), neuron/column scaling
(U = (s-1) * W[:, k], V = e_k) and general rank-r updates.
Allowed locations (declared before tier 1): gate_proj / up_proj / down_proj of text decoder layers.
"""
import contextlib
import re

import torch

from .common import digest

ALLOWED = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)$')
RANDOM_SEED = 43


def get_module(model, name):
    assert ALLOWED.match(name), f'edit location not allowed: {name}'
    m = model.get_submodule(name)
    assert isinstance(m, torch.nn.Linear) and m.bias is None
    return m


class Edit:
    def __init__(self, ops, name, description=''):
        self.ops = [dict(module=o['module'], U=o['U'].detach().float().cpu(), V=o['V'].detach().float().cpu())
                    for o in ops]
        self.name, self.description = name, description

    def identity(self):
        return digest(b''.join(o['module'].encode() + o['U'].numpy().tobytes() + o['V'].numpy().tobytes()
                               for o in self.ops))[:16]

    def save(self, path):
        torch.save(dict(name=self.name, description=self.description, ops=self.ops), path)

    @classmethod
    def load(cls, path):
        d = torch.load(path)
        return cls(d['ops'], d['name'], d['description'])

    def intended_stats(self, model):
        out = []
        for o in self.ops:
            W = get_module(model, o['module']).weight
            U, V = o['U'].cuda(), o['V'].cuda()
            # ||U V^T||_F via r x r Gram matrices; never materialize when not needed
            fro = float(torch.sqrt(torch.clamp(((U.T @ U) * (V.T @ V)).sum(), min=0)))
            out.append(dict(module=o['module'], rank=int(U.shape[1]), intended_fro=fro,
                            relative_fro=fro / float(W.norm())))
        return out


@contextlib.contextmanager
def applied(model, edit, stats=None):
    """Apply an edit in float32; restore the exact original tensors on exit (verified bitwise)."""
    if edit is None or not edit.ops:
        yield
        return
    originals = {}
    try:
        for o in edit.ops:
            m = get_module(model, o['module'])
            if o['module'] not in originals:
                originals[o['module']] = m.weight.detach().clone()
            m.weight.add_(o['U'].cuda() @ o['V'].cuda().T)
        if stats is not None:
            for name, orig in originals.items():
                W = get_module(model, name).weight
                d = W - orig
                stats.append(dict(module=name, realized_fro=float(d.norm()), relative_fro=float(d.norm() / orig.norm()),
                                  max_abs=float(d.abs().max()), changed_fraction=float((d != 0).float().mean()),
                                  changed_entries=int((d != 0).sum()), entries=int(d.numel())))
                del d
        yield
    finally:
        for name, orig in originals.items():
            W = get_module(model, name).weight
            W.copy_(orig)
            assert torch.equal(W, orig), f'restoration failed for {name}'
        del originals
        torch.cuda.empty_cache()


def matched_random(edit, seed=RANDOM_SEED):
    """Same modules, same rank and identical singular values per op (hence identical intended Frobenius norm);
    output- and input-space singular directions replaced by Haar-random orthonormal bases (seed 43)."""
    g = torch.Generator().manual_seed(seed)
    ops = []
    for o in edit.ops:
        U, V = o['U'].double(), o['V'].double()
        qu, ru = torch.linalg.qr(U)
        qv, rv = torch.linalg.qr(V)
        s = torch.linalg.svdvals(ru @ rv.T)
        r = U.shape[1]
        a = torch.linalg.qr(torch.randn(U.shape[0], r, generator=g, dtype=torch.float64))[0]
        b = torch.linalg.qr(torch.randn(V.shape[0], r, generator=g, dtype=torch.float64))[0]
        ops.append(dict(module=o['module'], U=(a * s).float(), V=b.float()))
    return Edit(ops, edit.name + '__random43', f'matched random control (seed {seed}) for {edit.name}')


def projection_scaling(model, module, d, alpha):
    """W <- W + alpha d (d^T W): scales the output component of `module` along unit vector d by (1+alpha)."""
    W = get_module(model, module).weight
    d = (d / d.norm()).float().to(W.device)
    return dict(module=module, U=(alpha * d)[:, None], V=(W.T @ d)[:, None])


def neuron_scaling(model, module, neurons, scales):
    """Scale input columns k of a down_proj (MLP neurons' write vectors) by factor s_k."""
    W = get_module(model, module).weight
    idx = torch.as_tensor(neurons, device=W.device)
    s = torch.as_tensor(scales, dtype=torch.float32, device=W.device)
    U = W[:, idx] * (s - 1)
    V = torch.zeros(W.shape[1], len(neurons), device=W.device)
    V[idx, torch.arange(len(neurons), device=W.device)] = 1
    return dict(module=module, U=U, V=V)


def output_hook_equivalent(model, edit):
    """Forward hooks z -> z + U V^T h on the edited Linear outputs (implementation test only)."""
    handles = []
    for o in edit.ops:
        m = get_module(model, o['module'])
        U, V = o['U'].cuda(), o['V'].cuda()

        def hook(mod, inp, out, U=U, V=V):
            return out + (inp[0] @ V) @ U.T
        handles.append(m.register_forward_hook(hook))
    return handles
