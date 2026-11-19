"""Sum the coding agent's own API token usage between two UTC timestamps.

Source: the Claude Code session transcript (JSONL), whose assistant entries carry per-API-call `usage`
(input, cache creation, cache read, output, thinking). Entries are de-duplicated by message id.
Usage: agent_tokens.py START_ISO END_ISO [--transcript PATH]
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys

DEFAULT_GLOB = os.path.expanduser('~/.claude/projects/-home-dianegative-sellma-claude/*.jsonl')


def parse(t):
    return dt.datetime.fromisoformat(t.replace('Z', '+00:00'))


def usage_between(start, end, transcript=None, include_prev_call=True):
    """include_prev_call: also count the last API call before `start` (the call that writes the start marker,
    which carries the session's opening reasoning)."""
    files = [transcript] if transcript else sorted(glob.glob(DEFAULT_GLOB), key=os.path.getmtime)[-1:]
    s, e = parse(start), parse(end)
    by_id, prev = {}, (None, None, None)
    for f in files:
        for line in open(f, encoding='utf-8'):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('type') != 'assistant' or 'timestamp' not in d:
                continue
            t = parse(d['timestamp'])
            msg = d.get('message', {})
            u = msg.get('usage')
            if not u:
                continue
            key = msg.get('id') or d.get('uuid')
            if t < s:
                if prev[0] is None or t >= prev[0]:
                    prev = (t, key, u)
                continue
            if t <= e:
                by_id[key] = u
    if include_prev_call and prev[1] is not None:
        by_id[prev[1]] = prev[2]
    tot = dict(api_calls=len(by_id), input_uncached=0, cache_creation_input=0, cache_read_input=0, output=0,
               thinking=0)
    for u in by_id.values():
        tot['input_uncached'] += u.get('input_tokens', 0)
        tot['cache_creation_input'] += u.get('cache_creation_input_tokens', 0)
        tot['cache_read_input'] += u.get('cache_read_input_tokens', 0)
        tot['output'] += u.get('output_tokens', 0)
        tot['thinking'] += (u.get('output_tokens_details') or {}).get('thinking_tokens', 0)
    tot['output_non_thinking'] = tot['output'] - tot['thinking']
    tot['transcript'] = os.path.basename(files[0]) if files else None
    tot['window_utc'] = [start, end]
    return tot


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('start'); ap.add_argument('end', nargs='?', default=None)
    ap.add_argument('--transcript')
    a = ap.parse_args()
    end = a.end or dt.datetime.now(dt.timezone.utc).isoformat()
    json.dump(usage_between(a.start, end, a.transcript), sys.stdout, indent=1)
    print()
