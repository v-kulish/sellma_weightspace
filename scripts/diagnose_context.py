"""Dev-only (tier 3: 104 Swe-BLiMP + 48 Idioms pairs) comparison of leading-context conventions."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from src import data
from src.model import load, chat_prompt
from src.scoring import Scorer, pair_metrics
from src.common import write
model, tok = load(); sc = Scorer(model, tok)
out = {}
for bench in ['swe_blimp', 'swedish_idioms']:
    rows = data.load_pairs(bench, 'dev', 3)
    raw = sc.pairs(rows, 'raw')
    out[bench + '_raw'] = pair_metrics(raw)
    for name, instr in [('chat_sv', 'Skriv en mening på svenska.'), ('chat_plain', 'Skriv en mening.')]:
        pre = tok.encode(chat_prompt(tok, [{'role': 'user', 'content': instr}]), add_special_tokens=False)
        seqs, starts = [], []
        for r in rows:
            for k in ('good', 'bad'):
                ids = tok.encode(r[k], add_special_tokens=False)
                seqs.append(pre + ids); starts.append(len(pre))
        s = sc.score_sequences(seqs, starts)
        recs = [dict(phenomenon=r['phenomenon'], margin=s[2*i]['mean_logp'] - s[2*i+1]['mean_logp'],
                     total_margin=s[2*i]['sum_logp'] - s[2*i+1]['sum_logp'], tie=False,
                     correct=int(s[2*i]['mean_logp'] > s[2*i+1]['mean_logp'])) for i, r in enumerate(rows)]
        out[f'{bench}_{name}'] = pair_metrics(recs)
        out[f'{bench}_{name}_mean_good_logp'] = float(np.mean([s[2*i]['mean_logp'] for i in range(len(rows))]))
    out[bench + '_raw_mean_good_logp'] = float(np.mean([r['good']['mean_logp'] for r in raw]))
write('results/context_diagnostic_dev.json', out)
for k, v in out.items():
    print(k, v if not isinstance(v, dict) else {x: round(y, 3) for x, y in v.items() if x in ('macro_accuracy', 'micro_accuracy', 'mean_margin', 'total_logp_accuracy')})
