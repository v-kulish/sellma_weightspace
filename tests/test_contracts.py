"""Small tests targeting ways the experiment could report a false effect. CPU only, no model download."""
import collections

import pytest
import torch

from src import data
from src.common import read
from src.scoring import masked_sentence_scores, scala_parse, pair_metrics
from src import edits


# ---------------------------------------------------------------- split integrity
def test_swe_blimp_split_nested_and_disjoint():
    m = read('manifests/splits/swe_blimp.json')['items']
    dev = [x for x in m if x['split'] == 'dev']
    test = [x for x in m if x['split'] == 'test']
    assert {x['id'] for x in dev}.isdisjoint({x['id'] for x in test})
    assert {x['group'] for x in dev}.isdisjoint({x['group'] for x in test})
    counts = collections.Counter(x['tier'] for x in dev)
    assert counts == {1: 26, 2: 26, 3: 52}
    for tier, n in [(1, 26), (2, 52), (3, 104)]:
        rows = data.load_pairs('swe_blimp', 'dev', tier)
        assert len(rows) == n and len({r['phenomenon'] for r in rows}) == 26
    t1 = {r['id'] for r in data.load_pairs('swe_blimp', 'dev', 1)}
    t2 = {r['id'] for r in data.load_pairs('swe_blimp', 'dev', 2)}
    assert t1 < t2


def test_idioms_scala_belebele_tiers():
    m = read('manifests/splits/swedish_idioms.json')['items']
    assert {x['group'] for x in m if x['split'] == 'dev'}.isdisjoint({x['group'] for x in m if x['split'] == 'test'})
    assert [len(data.load_pairs('swedish_idioms', 'dev', t)) for t in (1, 2, 3)] == [12, 24, 48]
    s = read('manifests/splits/scala.json')['languages']
    for lang, v in s.items():
        groups = {k: {x['group'] for x in v[k]} for k in ('test', 'dev', 'fewshot')}
        assert groups['test'].isdisjoint(groups['dev']) and groups['test'].isdisjoint(groups['fewshot'])
        assert groups['dev'].isdisjoint(groups['fewshot'])
        assert len(v['test']) == 256 and collections.Counter(x['label'] for x in v['test']) == {'correct': 128, 'incorrect': 128}
        for t, n in [(1, 8), (2, 16), (3, 32)]:
            rows = [x for x in v['dev'] if x['tier'] <= t]
            assert len(rows) == n and collections.Counter(x['label'] for x in rows)['correct'] == n // 2
    b = read('manifests/splits/belebele.json')
    assert {x['link'] for x in b['dev']}.isdisjoint({x['link'] for x in b['test']})
    assert [sum(x['tier'] <= t for x in b['dev']) for t in (1, 2, 3)] == [10, 20, 40]
    assert all(set(b['row_hashes'][l]) == set(b['row_hashes']['swe_Latn']) for l in data.BELEBELE_LANGS)


def test_split_reproducible():
    rows = data.swe_blimp_raw()
    again = data.split_pairs(rows, per_cat_quota={1: 1, 2: 2, 3: 4})
    stored = read('manifests/splits/swe_blimp.json')['items']
    assert [(x['id'], x['split'], x['tier']) for x in again] == [(x['id'], x['split'], x['tier']) for x in stored]


def test_scala_relation_detection():
    rows = [dict(text='Jag gillar katter mycket', label='correct'), dict(text='Jag katter gillar mycket', label='incorrect'),
            dict(text='Jag gillar mycket', label='incorrect'), dict(text='Helt annan mening här', label='correct')]
    g = data.scala_relations(rows)
    assert g[0] == g[1] == g[2] and g[3] != g[0]


# ---------------------------------------------------------------- likelihood contract
def test_causal_shift_and_mask():
    V = 5
    logits = torch.full((1, 4, V), -1e4)
    logits[0, 0, 2] = 0; logits[0, 1, 3] = 0; logits[0, 2, 0] = 0   # position t predicts token t+1
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[False, True, True, False]])  # score tokens 1 and 2 only
    s, c = masked_sentence_scores(logits, ids, mask)
    assert int(c[0]) == 2 and abs(float(s[0])) < 1e-3
    mask2 = torch.tensor([[False, True, True, True]])   # token 3 is improbable -> must lower score
    s2, _ = masked_sentence_scores(logits, ids, mask2)
    assert float(s2[0]) < -100


def test_pair_orientation_and_ties():
    recs = [dict(phenomenon='a', correct=1, margin=0.5, total_margin=1, tie=False, good={'tokens': 3}, bad={'tokens': 3}),
            dict(phenomenon='a', correct=0, margin=0.0, total_margin=0, tie=True, good={'tokens': 3}, bad={'tokens': 4}),
            dict(phenomenon='b', correct=0, margin=-1, total_margin=-1, tie=False, good={'tokens': 2}, bad={'tokens': 2})]
    m = pair_metrics(recs)
    assert m['macro_accuracy'] == pytest.approx(0.25) and m['micro_accuracy'] == pytest.approx(1 / 3)
    assert m['ties'] == 1 and m['equal_length_n'] == 2


def test_scala_parser_strict():
    mp = {'correct': 'ja', 'incorrect': 'nej'}
    assert scala_parse(' Ja.', mp) == 1 and scala_parse('nej', mp) == 0
    assert scala_parse('ja, den är korrekt', mp) == -1 and scala_parse('', mp) == -1


# ---------------------------------------------------------------- edits
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.Module(); layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(6, 4, bias=False)
        layer.mlp.up_proj = torch.nn.Linear(4, 6, bias=False)
        self.model = torch.nn.Module(); self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([layer])


@pytest.fixture
def toy(monkeypatch):
    m = Toy().requires_grad_(False)  # mirrors the real model (requires_grad disabled)
    # edits move factors to cuda; on CPU-only test machines map .cuda() to identity
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    return m


def test_projection_scaling_math_and_restoration(toy):
    name = 'model.language_model.layers.0.mlp.down_proj'
    W0 = toy.get_submodule(name).weight.detach().clone()
    d = torch.tensor([1., 0, 0, 0])
    e = edits.Edit([edits.projection_scaling(toy, name, d, 0.5)], 't')
    up0 = toy.get_submodule('model.language_model.layers.0.mlp.up_proj').weight.clone()
    with edits.applied(toy, e):
        W = toy.get_submodule(name).weight
        assert torch.allclose(W[0], 1.5 * W0[0]) and torch.equal(W[1:], W0[1:])
        assert torch.equal(toy.get_submodule('model.language_model.layers.0.mlp.up_proj').weight, up0)
    assert torch.equal(toy.get_submodule(name).weight, W0)
    with edits.applied(toy, edits.Edit([edits.projection_scaling(toy, name, d, 0.0)], 'zero')):
        assert torch.equal(toy.get_submodule(name).weight, W0)


def test_weight_hook_equivalence(toy):
    name = 'model.language_model.layers.0.mlp.down_proj'
    lin = toy.get_submodule(name)
    e = edits.Edit([edits.neuron_scaling(toy, name, [1, 4], [0.0, 2.0])], 'n')
    h = torch.randn(3, 6)
    with edits.applied(toy, e):
        a = lin(h)
    hs = edits.output_hook_equivalent(toy, e)
    b = lin(h)
    for x in hs:
        x.remove()
    assert torch.allclose(a, b, atol=1e-6)


def test_matched_random_spectrum(toy):
    name = 'model.language_model.layers.0.mlp.down_proj'
    e = edits.Edit([edits.neuron_scaling(toy, name, [0, 2, 3], [0.5, 1.5, 0.0])], 'n')
    r = edits.matched_random(e)
    d1 = e.ops[0]['U'] @ e.ops[0]['V'].T
    d2 = r.ops[0]['U'] @ r.ops[0]['V'].T
    assert torch.allclose(torch.linalg.svdvals(d1), torch.linalg.svdvals(d2), atol=1e-5)
    assert r.ops[0]['module'] == name and r.identity() == edits.matched_random(e).identity()  # seeded
    assert not torch.allclose(d1, d2)


def test_disallowed_location(toy):
    with pytest.raises(AssertionError):
        edits.get_module(toy, 'model.language_model.layers.0.self_attn.o_proj')


# ---------------------------------------------------------------- sealing
def test_test_sealed_without_lock(tmp_path, monkeypatch):
    from src import evaluate
    import pathlib
    if (pathlib.Path(evaluate.ROOT) / 'manifests/experiment_lock.json').exists():
        pytest.skip('lock exists (post-lock state)')
    e = edits.Edit([], 'x')
    with pytest.raises(PermissionError):
        evaluate.check_unsealed('tier1_edit', e)
    evaluate.check_unsealed('baseline', None)  # baseline allowed (disclosed user request)


# ---------------------------------------------------------------- analysis
def test_bootstrap_alignment_and_pp():
    from src.analyze import task_stats, diff
    base = [dict(id=f'i{k}', group=f'g{k // 2}', phenomenon='p' + str(k % 3), correct=k % 2, margin=0.1) for k in range(60)]
    better = [dict(r, correct=1) for r in base]
    st, n, g = task_stats('swe_blimp_sv', {'a': better, 'b': base}, 1)
    d = diff(st, 'a', 'b', 'accuracy')
    assert n == 60 and g == 30
    assert d['delta'] == pytest.approx(50.0) and d['lo'] > 0          # percentage points, not relative %
    st2, _, _ = task_stats('swe_blimp_sv', {'a': base, 'b': base}, 1)
    z = diff(st2, 'a', 'b', 'accuracy')
    assert z['delta'] == 0 and z['lo'] == 0 and z['hi'] == 0
    with pytest.raises(AssertionError):
        task_stats('swe_blimp_sv', {'a': base, 'b': base[::-1]}, 1)   # misaligned items must fail


def test_mcc_matches_sklearn():
    import numpy as np
    from sklearn.metrics import matthews_corrcoef
    from src.analyze import mcc_from_counts
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 2, 200); pred = np.where(rng.random(200) < 0.1, -1, np.where(rng.random(200) < 0.8, gold, 1 - gold))
    cm = np.zeros((2, 3))
    for g, p in zip(gold, pred):
        cm[g, p if p >= 0 else 2] += 1
    assert mcc_from_counts(cm) == pytest.approx(matthews_corrcoef(gold, pred))
