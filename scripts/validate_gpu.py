"""Live numerical/implementation checks on the real model, using only synthetic and dev inputs.
Writes results/validation.json. Run before any baseline or adjustment."""
import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from src import data
from src.common import write, now
from src.model import load, chat_prompt
from src.scoring import Scorer, belebele_prompt, eval_scala, masked_sentence_scores
from src.edits import Edit, applied, output_hook_equivalent, projection_scaling, matched_random

model, tok = load()
sc = Scorer(model, tok)
R = dict(utc=now())
synthetic = ['Hunden sover i sin korg.', 'Hunden sova i sin korg.', 'Jag har aldrig sett en sådan stor älg förut, sa hon.',
             'The cat sat on the mat.', 'The cat sat mat the on.', 'Det regnar.', 'Kan du hjälpa mig med det här problemet i morgon bitti?']
dev = data.load_pairs('swe_blimp', 'dev', 1)
texts = synthetic + [r[k] for r in dev[:12] for k in ('good', 'bad')]

# 1. batch vs single, repeat
b = sc.sentences(texts)
s = [sc.sentences([t], batch_size=1)[0] for t in texts]
b2 = sc.sentences(texts)
R['batch_single_max_diff'] = max(abs(x['mean_logp'] - y['mean_logp']) for x, y in zip(b, s))
R['repeat_max_diff'] = max(abs(x['mean_logp'] - y['mean_logp']) for x, y in zip(b, b2))
R['synthetic'] = {t: round(x['mean_logp'], 4) for t, x in zip(synthetic, b)}

# 2. scoring mask sanity on known logits
logits = torch.zeros(1, 4, 5); logits[0, 0, 2] = 10; logits[0, 1, 3] = 10; logits[0, 2, 4] = 10
ids = torch.tensor([[1, 2, 3, 4]]); m = torch.tensor([[False, True, True, False]])
sums, counts = masked_sentence_scores(logits, ids, m)
R['synthetic_logits_check'] = dict(sum=float(sums[0]), count=int(counts[0]))
assert int(counts[0]) == 2

# 3. zero edit, weight vs hook, restoration, no unintended mutation
name = 'model.language_model.layers.23.mlp.down_proj'
def param_fingerprint():
    return {n: float(p.double().sum()) for n, p in model.named_parameters()}
fp0 = param_fingerprint()
g = torch.Generator().manual_seed(0)
d = torch.randn(3840, generator=g)
e = Edit([projection_scaling(model, name, d, 0.25)], 'smoke')
z = Edit([projection_scaling(model, name, d, 0.0)], 'zero')
with applied(model, z):
    bz = sc.sentences(texts)
R['zero_edit_max_diff'] = max(abs(x['mean_logp'] - y['mean_logp']) for x, y in zip(b, bz))
st = []
with applied(model, e, st):
    be = sc.sentences(texts)
    fpe = param_fingerprint()
R['edit_stats'] = st
R['mutated_params_during_edit'] = [n for n in fp0 if fp0[n] != fpe[n]]
hs = output_hook_equivalent(model, e)
bh = sc.sentences(texts)
for h in hs: h.remove()
R['weight_vs_hook_max_diff'] = max(abs(x['mean_logp'] - y['mean_logp']) for x, y in zip(be, bh))
R['edit_effect_max'] = max(abs(x['mean_logp'] - y['mean_logp']) for x, y in zip(b, be))
br = sc.sentences(texts)
R['restored_max_diff'] = max(abs(x['mean_logp'] - y['mean_logp']) for x, y in zip(b, br))
R['restored_params_equal'] = param_fingerprint() == fp0
rnd = matched_random(e)
R['random_norm'] = [rnd.intended_stats(model), e.intended_stats(model)]

# 4. answer-label shortcut vs full teacher forcing
row = data.load_belebele('swe_Latn', 'dev', 1)[0]
prompt = belebele_prompt(tok, 'swe_Latn', row['row'])
fast, n = sc.choice_scores(prompt, ['A', 'B', 'C', 'D'])
enc = [sc.continuation(prompt, L) for L in 'ABCD']
full = [x['sum_logp'] for x in sc.score_sequences([p + q for p, q in enc], [len(p) for p, q in enc], batch_size=1)]
R['choice_shortcut_max_diff'] = max(abs(x - y) for x, y in zip(fast, full))
R['label_ids'] = [q for _, q in enc]
R['chat_tail'] = prompt[-80:]

# 5. prefix-cache generation equivalence (dev ScaLA sv)
rows = data.load_scala('sv', 'dev', 1)
shots = data.load_scala('sv', 'fewshot')
t = time.monotonic(); with_cache = eval_scala(sc, rows, shots, 'check'); t_cache = time.monotonic() - t
from src import scoring
orig = scoring.Scorer.generate
t = time.monotonic()
scoring.Scorer.generate = lambda self, p, n, prefix_cache=True: orig(self, p, n, prefix_cache=False)
plain = eval_scala(sc, rows, shots, 'check'); t_plain = time.monotonic() - t
scoring.Scorer.generate = orig
R['prefix_cache_identical_outputs'] = all(a['output_ids'] == b_['output_ids'] for a, b_ in zip(with_cache, plain))
R['prefix_cache_outputs'] = [(a['raw_output'], b_['raw_output']) for a, b_ in zip(with_cache, plain)]
R['scala_seconds'] = dict(cache=t_cache, plain=t_plain, n=len(rows))
R['scala_prompt_tokens'] = [a['prompt_tokens'] for a in with_cache]

# 6. timing: belebele per prompt, swe pairs
t = time.monotonic(); data_rows = data.load_belebele('eng_Latn', 'dev', 1)
from src.scoring import eval_belebele
eval_belebele(sc, data_rows, 'check'); R['belebele_seconds_per_q'] = (time.monotonic() - t) / len(data_rows)
t = time.monotonic(); sc.pairs(data.load_pairs('swe_blimp', 'dev', 3), 'check'); R['swe_104_seconds'] = time.monotonic() - t
R['peak_gb'] = torch.cuda.max_memory_allocated() / 1e9
write('results/validation.json', R)
print(json.dumps({k: v for k, v in R.items() if k not in ('synthetic',)}, indent=1, ensure_ascii=False)[:4000])
