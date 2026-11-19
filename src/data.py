"""Dataset adapters and immutable, label-aware but outcome-independent split manifests.

Every benchmark gets nested adjustment tiers (1 < 2 < 3) that are excluded from its test set.
Split seed 42. Manifests are written once (immutable) before any adjustment.
Raw third-party text stays in ignored data/; manifests store ids and hashes.
"""
import collections
import csv
import json
import random
import re

import pandas as pd

from .common import ROOT, digest, file_digest, read, write, now

SEED = 42
BELEBELE_LANGS = ['swe_Latn', 'dan_Latn', 'nob_Latn', 'fin_Latn', 'isl_Latn', 'eng_Latn', 'zho_Hans']
SCALA_LANGS = ['sv', 'da', 'nb', 'en']
TIER_SIZES = {
    'swe_blimp': {1: 26, 2: 52, 3: 104},           # 1/2/4 per phenomenon
    'swedish_idioms': {1: 12, 2: 24, 3: 48},
    'scala': {1: 8, 2: 16, 3: 32},                  # per language, label-stratified
    'belebele': {1: 10, 2: 20, 3: 40},              # questions; whole 2-question passages
}
SCALA_TEST_PER_LANG = 256
BELEBELE_TEST_TARGET = 200


def normalized(text):
    return tuple(re.findall(r'\w+', text.casefold()))


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def pair_groups(rows, near_within='phenomenon'):
    """Union pairs that share a normalized sentence (globally) or that are one-word variants
    on BOTH sides within the same category (transitive closure). Conservative and transparent;
    cannot rule out all templatic leakage."""
    uf = UnionFind(len(rows))
    seen, by_cat, toks = {}, collections.defaultdict(list), []
    for i, r in enumerate(rows):
        pair = [normalized(r['good']), normalized(r['bad'])]
        toks.append(pair)
        for s in pair:
            if s in seen:
                uf.union(i, seen[s])
            seen[s] = i
        cat = r.get(near_within, '') if near_within else ''
        for j in by_cat[cat]:
            if all(len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= 1 for a, b in zip(pair, toks[j])):
                uf.union(i, j)
        by_cat[cat].append(i)
    return [f'g{uf.find(i):04d}' for i in range(len(rows))]


# ---------------------------------------------------------------- raw loaders

def swe_blimp_raw():
    with (ROOT / 'data/swe_blimp/swe_blimp_v1.csv').open(encoding='utf-8') as f:
        return [dict(id=f'swe-{i:04d}', row=i, phenomenon=r['phenomenon'], good=r['sentence_good'],
                     bad=r['sentence_bad']) for i, r in enumerate(csv.DictReader(f, delimiter=';'))]


def idioms_raw():
    d = pd.read_parquet(ROOT / 'data/idioms.parquet')
    return [dict(id=f'idiom-{i:04d}', row=i, phenomenon='challenge' if r['Challenge Set'] == 'x' else 'standard',
                 good=r['Positive (Minimal Pairs Setup)'], bad=r['Negative (Minimal Pairs Setup)'])
            for i, r in d.iterrows()]


def scala_raw(lang, split):
    d = pd.read_parquet(ROOT / f'data/scala/{lang}_{split}.parquet').fillna('')
    return d.to_dict('records')


def belebele_raw(lang):
    rows = [json.loads(x) for x in (ROOT / f'data/belebele/{lang}.jsonl').read_text(encoding='utf-8').splitlines()]
    return {(r['link'], str(r['question_number'])): r for r in rows}


# ---------------------------------------------------------------- split builders

def tier_of(rank, sizes_per_unit):
    """Smallest tier whose per-unit quota includes this 1-based rank."""
    for t in (1, 2, 3):
        if rank <= sizes_per_unit[t]:
            return t
    return None


def split_pairs(rows, per_cat_quota=None, total_quota=None, near_within='phenomenon'):
    rng = random.Random(SEED)
    groups = pair_groups(rows, near_within)
    reserve = {}
    if per_cat_quota:  # Swe-BLiMP: ranks within phenomenon
        for cat in sorted({r['phenomenon'] for r in rows}):
            ids = [i for i, r in enumerate(rows) if r['phenomenon'] == cat]
            rng.shuffle(ids)
            chosen, used = [], set()
            for i in ids:  # at most one reserved row per group within a category
                if groups[i] in used:
                    continue
                used.add(groups[i]); chosen.append(i)
                if len(chosen) == per_cat_quota[3]:
                    break
            for rank, i in enumerate(chosen, 1):
                reserve[i] = tier_of(rank, per_cat_quota)
    else:
        ids = list(range(len(rows)))
        rng.shuffle(ids)
        chosen, used = [], set()
        for i in ids:
            if groups[i] in used:
                continue
            used.add(groups[i]); chosen.append(i)
            if len(chosen) == total_quota[3]:
                break
        for rank, i in enumerate(chosen, 1):
            reserve[i] = tier_of(rank, total_quota)
    reserved_groups = {groups[i] for i in reserve}
    items = []
    for i, r in enumerate(rows):
        split = 'dev' if i in reserve else ('quarantine' if groups[i] in reserved_groups else 'test')
        items.append(dict(id=r['id'], row=r['row'], phenomenon=r['phenomenon'], group=groups[i],
                          good_hash=digest(r['good']), bad_hash=digest(r['bad']), split=split,
                          tier=reserve.get(i)))
    return items


def scala_variants(words):
    """All one-word deletions and adjacent swaps of a correct sentence: ScaLA's corruption types."""
    out = set()
    for k in range(len(words)):
        out.add(words[:k] + words[k + 1:])
    for k in range(len(words) - 1):
        w = list(words); w[k], w[k + 1] = w[k + 1], w[k]
        out.add(tuple(w))
    return out


def scala_relations(rows):
    """Group rows that are identical or plausibly corruption/source of each other."""
    uf = UnionFind(len(rows))
    norm = [normalized(r['text']) for r in rows]
    by_text = collections.defaultdict(list)
    for i, n in enumerate(norm):
        by_text[n].append(i)
    for ids in by_text.values():
        for j in ids[1:]:
            uf.union(ids[0], j)
    for i, r in enumerate(rows):
        if r['label'] != 'correct':
            continue
        for v in scala_variants(norm[i]):
            for j in by_text.get(v, []):
                if j != i:
                    uf.union(i, j)
    return [uf.find(i) for i in range(len(rows))]


def split_scala(lang):
    test_rows = scala_raw(lang, 'test')
    train_rows = scala_raw(lang, 'train')
    rows = [dict(r, origin='test', row=i) for i, r in enumerate(test_rows)] + \
           [dict(r, origin='train', row=i) for i, r in enumerate(train_rows)]
    rel = scala_relations(rows)
    rng = random.Random(f'{SEED}-scala-{lang}')
    test_idx = [i for i, r in enumerate(rows) if r['origin'] == 'test']
    used_groups = set()

    def take(pool, label, k, avoid):
        cand = [i for i in pool if rows[i]['label'] == label]
        rng.shuffle(cand)
        out = []
        for i in cand:
            if rel[i] in avoid:
                continue
            avoid.add(rel[i]); out.append(i)
            if len(out) == k:
                break
        assert len(out) == k, (lang, label, k)
        return out

    test = take(test_idx, 'correct', SCALA_TEST_PER_LANG // 2, used_groups) + \
        take(test_idx, 'incorrect', SCALA_TEST_PER_LANG // 2, used_groups)
    rng.shuffle(test)
    # Dev tiers drawn from official-test rows NOT in the test sample and unrelated to it.
    dev_pos = take(test_idx, 'correct', 16, used_groups)
    dev_neg = take(test_idx, 'incorrect', 16, used_groups)
    dev = []
    for rank in range(16):  # interleave so nested tiers stay label balanced
        dev += [(dev_pos[rank], rank + 1), (dev_neg[rank], rank + 1)]
    # Frozen 12-shot evaluation context from official train, unrelated to test/dev.
    train_idx = [i for i, r in enumerate(rows) if r['origin'] == 'train']
    shots = take(train_idx, 'correct', 6, used_groups) + take(train_idx, 'incorrect', 6, used_groups)
    rng.shuffle(shots)
    per_label = {1: 4, 2: 8, 3: 16}

    def rec(i, split, tier=None):
        r = rows[i]
        return dict(id=f'scala-{lang}-{r["origin"]}-{r["row"]:04d}', origin=r['origin'], row=r['row'],
                    label=r['label'], corruption_type=r['corruption_type'], text_hash=digest(r['text']),
                    group=f'{lang}-{rel[i]:05d}', split=split, tier=tier)
    return dict(test=[rec(i, 'test') for i in test],
                dev=[rec(i, 'dev', tier_of(rank, per_label)) for i, rank in dev],
                fewshot=[rec(i, 'fewshot') for i in shots])


def split_belebele():
    data = {lang: belebele_raw(lang) for lang in BELEBELE_LANGS}
    keys = sorted(data['swe_Latn'])
    for lang in BELEBELE_LANGS:
        assert len(data[lang]) == 900 and sorted(data[lang]) == keys, lang
        for k in keys:
            assert str(data[lang][k]['correct_answer_num']) == str(data['swe_Latn'][k]['correct_answer_num']), (lang, k)
    passages = collections.defaultdict(list)
    for link, q in keys:
        passages[link].append(q)
    links = sorted(passages)
    rng = random.Random(f'{SEED}-belebele')
    rng.shuffle(links)
    dev, test, n_test = [], [], 0
    two = [l for l in links if len(passages[l]) == 2]
    dev_links = two[:20]                      # 5/10/20 passages -> 10/20/40 questions
    for rank, link in enumerate(dev_links, 1):
        tier = 1 if rank <= 5 else (2 if rank <= 10 else 3)
        for q in sorted(passages[link]):
            dev.append(dict(id=f'{link}#{q}', link=link, question_number=q, passage=link, split='dev', tier=tier))
    for link in links:
        if link in dev_links:
            continue
        if n_test >= BELEBELE_TEST_TARGET:
            break
        for q in sorted(passages[link]):
            test.append(dict(id=f'{link}#{q}', link=link, question_number=q, passage=link, split='test', tier=None))
            n_test += 1
    hashes = {lang: {r['id']: digest(json.dumps(data[lang][(r['link'], r['question_number'])], sort_keys=True,
                                                ensure_ascii=False)) for r in dev + test} for lang in BELEBELE_LANGS}
    return dict(dev=dev, test=test, row_hashes=hashes)


def prepare():
    code = digest((ROOT / 'src/data.py').read_bytes())
    sources = read('manifests/sources.json')
    swe = split_pairs(swe_blimp_raw(), per_cat_quota={1: 1, 2: 2, 3: 4})
    idi = split_pairs(idioms_raw(), total_quota=TIER_SIZES['swedish_idioms'], near_within=None)
    sca = {lang: split_scala(lang) for lang in SCALA_LANGS}
    bel = split_belebele()
    common = dict(seed=SEED, created_utc=now(), split_code_sha256=code, tier_sizes=TIER_SIZES)
    write('manifests/splits/swe_blimp.json', common | dict(
        source=sources['swe_blimp'], counts=dict(collections.Counter(x['split'] for x in swe)),
        group_rule='shared normalized sentence globally; <=1-word substitution on both sides within phenomenon; '
                   'transitive closure; at most one reserved row per group per phenomenon; test rows in a '
                   'reserved group are quarantined (excluded from test and dev).',
        items=swe), immutable=True)
    write('manifests/splits/swedish_idioms.json', common | dict(
        source=sources['swedish_idioms'], counts=dict(collections.Counter(x['split'] for x in idi)),
        group_rule='shared normalized sentence; <=1-word substitution on both sides (whole dataset); quarantine as Swe-BLiMP',
        note='Publisher split name is "train"; in this project all rows are evaluation rows: dev tiers or test.',
        items=idi), immutable=True)
    write('manifests/splits/scala.json', common | dict(
        source=sources['scala'], languages=sca,
        group_rule='exact normalized text, plus one-word-deletion / adjacent-swap relation to a correct row '
                   '(ScaLA corruption types); related rows never straddle test/dev/few-shot.',
        protocol='scala_euroeval_wording_12shot_greedy_strict_v1'), immutable=True)
    write('manifests/splits/belebele.json', common | dict(
        source=sources['belebele'], languages=BELEBELE_LANGS, **bel,
        counts=dict(dev=len(bel['dev']), test=len(bel['test']),
                    test_passages=len({r['link'] for r in bel['test']}))), immutable=True)
    print('swe_blimp', collections.Counter(x['split'] for x in swe))
    print('idioms', collections.Counter(x['split'] for x in idi))
    print('scala', {l: {k: len(v) for k, v in s.items()} for l, s in sca.items()})
    print('belebele', len(bel['dev']), len(bel['test']))


# ---------------------------------------------------------------- loaders for evaluation

def _in_tier(item, tier):
    return item['split'] == 'dev' and item['tier'] is not None and item['tier'] <= tier


def load_pairs(benchmark, split, tier=None):
    raw = {r['id']: r for r in (swe_blimp_raw() if benchmark == 'swe_blimp' else idioms_raw())}
    man = read(f'manifests/splits/{benchmark}.json')['items']
    out = []
    for m in man:
        if split == 'test' and m['split'] != 'test':
            continue
        if split == 'dev' and not _in_tier(m, tier):
            continue
        r = raw[m['id']]
        assert digest(r['good']) == m['good_hash'] and digest(r['bad']) == m['bad_hash']
        out.append(dict(m, good=r['good'], bad=r['bad'], benchmark=benchmark, language='sv'))
    return out


def load_scala(lang, split, tier=None):
    man = read('manifests/splits/scala.json')['languages'][lang]
    raw = {'test': scala_raw(lang, 'test'), 'train': scala_raw(lang, 'train')}
    rows = man['test'] if split == 'test' else (man['fewshot'] if split == 'fewshot' else
                                                [m for m in man['dev'] if _in_tier(m, tier)])
    out = []
    for m in rows:
        r = raw[m['origin']][m['row']]
        assert digest(r['text']) == m['text_hash'] and r['label'] == m['label']
        out.append(dict(m, text=r['text'], gold=int(r['label'] == 'correct'), benchmark='scala', language=lang))
    return out


def load_belebele(lang, split, tier=None):
    man = read('manifests/splits/belebele.json')
    data = belebele_raw(lang)
    rows = man['test'] if split == 'test' else [m for m in man['dev'] if _in_tier(m, tier)]
    out = []
    for m in rows:
        r = data[(m['link'], m['question_number'])]
        assert digest(json.dumps(r, sort_keys=True, ensure_ascii=False)) == man['row_hashes'][lang][m['id']]
        out.append(dict(m, row=r, gold=int(r['correct_answer_num']) - 1, benchmark='belebele', language=lang))
    return out


if __name__ == '__main__':
    prepare()
