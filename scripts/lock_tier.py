"""Lock a tier: record the chosen candidate (or none), build its matched random control (seed 43), dev-evaluate the control.
Usage: lock_tier.py TIER CANDIDATE_NAME|none"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import ROOT, read, write, now
from src.edits import Edit, matched_random
from src.session import evaluate_candidates, decision_row
N, name = int(sys.argv[1]), sys.argv[2]
rounds = sorted((ROOT / 'results/rounds').glob(f'tier{N}_round*.json'))
cands = {}
for p in rounds:
    cands.update(json.loads(p.read_text())['candidates'])
if name == 'none':
    write(f'results/tiers/tier{N}_lock.json', dict(utc=now(), tier=N, candidate=None, edit_artifact=None, random_artifact=None,
                                                  edit_identity=None, random_identity=None, rounds=[p.name for p in rounds]),
          immutable=True)
    sys.exit()
c = cands[name]
e = Edit.load(ROOT / c['artifact']); assert e.identity() == c['identity']
r = matched_random(e)
rp = f'artifacts/tier{N}/{r.name}.pt'; r.save(ROOT / rp)
from src.model import load
from src.scoring import Scorer
model, tok = load(); sc = Scorer(model, tok)
res = evaluate_candidates(sc, N, {'baseline': None, f'tier{N}_random': r})
base = res['baseline']['summaries']
lock = dict(utc=now(), tier=N, candidate=name, edit_artifact=c['artifact'], edit_identity=e.identity(),
            random_artifact=rp, random_identity=r.identity(), rounds=[p.name for p in rounds],
            n_candidates_evaluated=len(cands), edit_dev=c['decision'],
            random_dev=decision_row(base, res[f'tier{N}_random']['summaries']),
            random_intended=r.intended_stats(model), edit_intended=e.intended_stats(model),
            random_realized=res[f'tier{N}_random']['edit_stats'], random_eval_accounting=res[f'tier{N}_random']['accounting'])
write(f'results/tiers/tier{N}_lock.json', lock, immutable=True)
print(json.dumps({k: v for k, v in lock.items() if k in ('candidate', 'edit_dev', 'random_dev')}, indent=1))
