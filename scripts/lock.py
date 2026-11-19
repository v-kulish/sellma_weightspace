"""Write manifests/experiment_lock.json once all three tiers are locked (unseals test evaluation)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import ROOT, read, write, now, digest, file_digest, source_hash
from src.edits import Edit
from src.scoring import CHAT_INSTRUCTION, CHAT_TEMPLATE, RAW_TEMPLATE, BELEBELE_TEMPLATE, SCALA_TEMPLATE, BELEBELE_INSTRUCTIONS

tiers = {t: read(f'results/tiers/tier{t}_lock.json') for t in (1, 2, 3)}
conditions = {}
for t, L in tiers.items():
    for kind in ('edit', 'random'):
        art = L.get(f'{kind}_artifact')
        if art is None:
            conditions[f'tier{t}_{kind}'] = dict(artifact=None, edit_identity=None, no_op=True,
                                                 note='no eligible edit locked; test result = unchanged model')
            continue
        e = Edit.load(ROOT / art)
        assert e.identity() == L[f'{kind}_identity']
        conditions[f'tier{t}_{kind}'] = dict(artifact=art, edit_identity=e.identity(), artifact_sha256=file_digest(art),
                                             candidate=L.get('candidate'), description=e.description)
lock = dict(utc=now(), conditions=conditions, tier_locks=tiers, source_hash=source_hash(),
            split_hashes={b: file_digest(f'manifests/splits/{b}.json') for b in
                          ('swe_blimp', 'swedish_idioms', 'scala', 'belebele')},
            protocols=dict(pairs_primary=CHAT_TEMPLATE, chat_instruction=CHAT_INSTRUCTION, pairs_secondary=RAW_TEMPLATE,
                           belebele=BELEBELE_TEMPLATE, belebele_instructions=BELEBELE_INSTRUCTIONS, scala=SCALA_TEMPLATE),
            protocol_file_sha256=file_digest('reports/adjustments/PROTOCOL.md'),
            planned_comparisons=['tier3_edit vs baseline on Swe-BLiMP test macro accuracy (primary)',
                                 'tierN_edit vs tierN_random (direction added value), N=1,2,3',
                                 'tierN_edit vs baseline on every task (transfer and regression), N=1,2,3',
                                 'tierN_random vs baseline on every task'],
            decision_rules=dict(primary_gain_pp=2.0, belebele_tolerable_loss_pp=-3.0, bootstrap=10000, seed=2026))
write('manifests/experiment_lock.json', lock, immutable=True)
print('LOCKED', {k: v['edit_identity'] for k, v in conditions.items()})
