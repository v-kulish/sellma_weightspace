"""Fill the <!--TOKENS--> block of a session report with agent and model-forward token accounting.
Usage: fill_tokens.py TIER ROUND [extra_accounting_json ...]
Agent window: results/sessions/tier{N}_round{R}.start -> now (calls writing the filled block itself fall outside)."""
import sys, pathlib, json, datetime as dt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.agent_tokens import usage_between
from src.common import ROOT, write
N, R = int(sys.argv[1]), int(sys.argv[2])
start = (ROOT / f'results/sessions/tier{N}_round{R}.start').read_text().strip()
endf = ROOT / f'results/sessions/tier{N}_round{R}.end'
end = endf.read_text().strip() if endf.exists() else dt.datetime.now(dt.timezone.utc).isoformat()
endf.write_text(end)
a = usage_between(start, end)
fw = {}
rp = ROOT / f'results/rounds/tier{N}_round{R}.json'
if rp.exists():
    r = json.loads(rp.read_text())
    fw['candidate construction (build)'] = r['build_accounting']
    fw['dev evaluation of candidates'] = r['eval_accounting']
for x in sys.argv[3:]:
    name, path, *key = x.split('|')
    d = json.loads((ROOT / path).read_text())
    for k in key: d = d[k]
    fw[name] = d
write(f'results/sessions/tier{N}_round{R}_tokens.json', dict(agent=a, model_forward=fw))
tot = sum(v.get('input_tokens', 0) + v.get('generated_tokens', 0) for v in fw.values())
L = [f'Window {start} → {end} (UTC). Includes the call that opened the session (it writes the start marker); a boundary call may be counted in both adjacent sessions; calls after the end mark (writing this block) are excluded.', '',
     '**Agent (Claude Opus 5.5) tokens, reasoning + executing the edit,** summed over the session transcript\'s per-call usage records (`scripts/agent_tokens.py`):', '',
     '| API calls | output tokens | of which thinking | output non-thinking | input uncached | cache-write input | cache-read input |', '|---|---|---|---|---|---|---|',
     f"| {a['api_calls']} | {a['output']:,} | {a['thinking']:,} | {a['output_non_thinking']:,} | {a['input_uncached']:,} | {a['cache_creation_input']:,} | {a['cache_read_input']:,} |", '',
     '**Gemma model-forward tokens** (exact counts from the scorer; non-padding input tokens plus generated tokens):', '',
     '| stage | forward calls | sequences | input tokens | generated tokens | GPU forward s |', '|---|---|---|---|---|---|']
for k, v in fw.items():
    L.append(f"| {k} | {v.get('forward_calls', 0)} | {v.get('sequences', 0)} | {v.get('input_tokens', 0):,} | {v.get('generated_tokens', 0):,} | {v.get('forward_seconds', 0):.0f} |")
L.append(f'| **total** | | | | {tot:,} (in+gen) | |')
p = ROOT / f'reports/adjustments/tier{N}_round{R}.md'
s = p.read_text()
head = s.split('\n## Token accounting\n')[0]
p.write_text(head + '\n## Token accounting\n' + '\n'.join(L) + '\n')
print('\n'.join(L))
