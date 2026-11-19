"""Likelihood scoring (raw sentences, answer-label continuations) and strict ScaLA generation."""
import copy
import re
import time

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, f1_score

from .model import chat_prompt

RAW_TEMPLATE = 'raw_bos_no_eos_mean_logp_v1'            # secondary protocol
CHAT_INSTRUCTION = 'Skriv en mening på svenska.'
CHAT_TEMPLATE = 'chat_user_skriv_en_mening_pa_svenska_nothink_mean_logp_v1'  # primary protocol


def token_logps(logits, targets, chunk=64):
    """FP32 log p(target) per position, chunked over the sequence to bound vocab-sized temporaries."""
    out = []
    for s in range(0, logits.shape[1], chunk):
        x = logits[:, s:s + chunk].float()
        y = targets[:, s:s + chunk]
        out.append(x.gather(-1, y.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(x, dim=-1))
    return torch.cat(out, dim=1)


def masked_sentence_scores(logits, ids, score_mask):
    """score_mask[b, t] marks tokens t (t>=1) to be scored as p(ids[t] | ids[<t]). Causal shift applied here."""
    lp = token_logps(logits[:, :-1], ids[:, 1:])
    m = score_mask[:, 1:].to(lp.dtype)
    sums = (lp * m).sum(-1)
    counts = m.sum(-1)
    return sums, counts


class Scorer:
    def __init__(self, model, tok, batch_size=32, max_length=4096):
        self.model, self.tok, self.batch_size, self.max_length = model, tok, batch_size, max_length
        self.reset()

    def reset(self):
        self.calls = self.sequences = self.tokens = self.generated = 0
        self.seconds = 0.0

    def accounting(self):
        return dict(forward_calls=self.calls, sequences=self.sequences, input_tokens=self.tokens,
                    generated_tokens=self.generated, forward_seconds=round(self.seconds, 3))

    # ------------------------------------------------------------ raw sentences
    def raw_ids(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)
        assert ids, text
        assert not set(ids) & set(self.tok.all_special_ids), 'special token inside raw sentence'
        return [self.tok.bos_token_id] + ids

    @torch.inference_mode()
    def score_sequences(self, seqs, starts, batch_size=None, hook=None):
        """Sum/mean log prob of seq[start:] given seq[:start]. Never truncates. Right padding, masked."""
        bs = batch_size or self.batch_size
        assert all(0 < s < len(q) <= self.max_length for q, s in zip(seqs, starts)), 'invalid or overlength'
        order = sorted(range(len(seqs)), key=lambda i: (len(seqs[i]), i))
        res = [None] * len(seqs)
        for off in range(0, len(order), bs):
            idx = order[off:off + bs]
            L = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), L), self.tok.pad_token_id, dtype=torch.long, device='cuda')
            att = torch.zeros_like(ids)
            sm = torch.zeros_like(ids, dtype=torch.bool)
            for j, i in enumerate(idx):
                n = len(seqs[i])
                ids[j, :n] = torch.tensor(seqs[i], device='cuda')
                att[j, :n] = 1
                sm[j, starts[i]:n] = True
            if hook is not None:
                hook.set_batch(idx, sm, att)
            torch.cuda.synchronize(); t = time.monotonic()
            logits = self.model(input_ids=ids, attention_mask=att, use_cache=False).logits
            sums, counts = masked_sentence_scores(logits, ids, sm)
            torch.cuda.synchronize(); self.seconds += time.monotonic() - t
            self.calls += 1; self.sequences += len(idx); self.tokens += int(att.sum())
            for j, i in enumerate(idx):
                res[i] = dict(sum_logp=float(sums[j]), tokens=int(counts[j]), mean_logp=float(sums[j] / counts[j]))
            del logits
        return res

    def sentences(self, texts, **kw):
        seqs = [self.raw_ids(t) for t in texts]
        return self.score_sequences(seqs, [1] * len(seqs), **kw)

    def chat_prefix(self):
        return self.tok.encode(chat_prompt(self.tok, [{'role': 'user', 'content': CHAT_INSTRUCTION}]),
                               add_special_tokens=False)

    def chat_sentences(self, texts, **kw):
        """Primary protocol: sentence as the model's reply to a fixed user turn; only sentence tokens scored."""
        pre = self.chat_prefix()
        seqs = []
        for t in texts:
            ids = self.tok.encode(t, add_special_tokens=False)
            assert ids and not set(ids) & set(self.tok.all_special_ids)
            seqs.append(pre + ids)
        return self.score_sequences(seqs, [len(pre)] * len(seqs), **kw)

    def pairs(self, rows, condition, raw=True, **kw):
        texts = [r[k] for r in rows for k in ('good', 'bad')]
        sc = self.chat_sentences(texts, **kw)
        rs = self.sentences(texts) if raw else None
        out = []
        for i, r in enumerate(rows):
            g, b = sc[2 * i], sc[2 * i + 1]
            margin = g['mean_logp'] - b['mean_logp']
            rec = dict(id=r['id'], benchmark=r['benchmark'], language='sv', split=r['split'], tier=r['tier'],
                       phenomenon=r['phenomenon'], group=r['group'], condition=condition,
                       margin=margin, correct=int(margin > 0), tie=margin == 0,
                       total_margin=g['sum_logp'] - b['sum_logp'], good=g, bad=b,
                       prompt_template=CHAT_TEMPLATE)
            if raw:
                rg, rb = rs[2 * i], rs[2 * i + 1]
                rec.update(raw_margin=rg['mean_logp'] - rb['mean_logp'],
                           raw_correct=int(rg['mean_logp'] - rb['mean_logp'] > 0), raw_good=rg, raw_bad=rb,
                           raw_template=RAW_TEMPLATE)
            out.append(rec)
        return out

    # ------------------------------------------------------------ answer-label continuation
    def continuation(self, prompt, label):
        pre = self.tok.encode(prompt, add_special_tokens=False)
        whole = self.tok.encode(prompt + label, add_special_tokens=False)
        if whole[:len(pre)] != pre or len(whole) <= len(pre):
            raise ValueError('label boundary retokenizes the prompt')
        return pre, whole[len(pre):]

    @torch.inference_mode()
    def choice_scores(self, prompt, labels):
        enc = [self.continuation(prompt, l) for l in labels]
        pre = enc[0][0]
        assert all(p == pre for p, _ in enc) and len(pre) <= self.max_length
        if all(len(s) == 1 for _, s in enc):  # exact shortcut: shared prefix, single-token labels
            ids = torch.tensor([pre], device='cuda')
            torch.cuda.synchronize(); t = time.monotonic()
            lg = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                            logits_to_keep=1).logits[0, -1].float()
            scores = [float(lg[s[0]] - torch.logsumexp(lg, -1)) for _, s in enc]
            torch.cuda.synchronize(); self.seconds += time.monotonic() - t
            self.calls += 1; self.sequences += 1; self.tokens += len(pre)
        else:
            r = self.score_sequences([p + s for p, s in enc], [len(p) for p, s in enc], batch_size=1)
            scores = [x['sum_logp'] for x in r]
        return scores, len(pre)

    # ------------------------------------------------------------ strict greedy generation
    @torch.inference_mode()
    def generate(self, prompts, max_new_tokens, prefix_cache=True):
        """Greedy decoding, batch 1. With prefix_cache, the longest common token prefix of all prompts is
        prefilled once and its KV cache deep-copied per prompt (validated equal to plain generation)."""
        enc = [self.tok.encode(p, add_special_tokens=False) for p in prompts]
        assert all(len(e) + max_new_tokens <= self.max_length for e in enc)
        past, P = None, 0
        if prefix_cache and len(enc) > 1:
            P = min(len(e) for e in enc) - 1
            for e in enc[1:]:
                P = min(P, next((k for k, (a, b) in enumerate(zip(enc[0], e)) if a != b), P))
            if P >= 16:
                ids = torch.tensor([enc[0][:P]], device='cuda')
                torch.cuda.synchronize(); t = time.monotonic()
                past = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True).past_key_values
                torch.cuda.synchronize(); self.seconds += time.monotonic() - t
                self.calls += 1; self.tokens += P
        outs = []
        for e in enc:
            ids = torch.tensor([e], device='cuda')
            kw = {}
            if past is not None:
                kw['past_key_values'] = copy.deepcopy(past)
            torch.cuda.synchronize(); t = time.monotonic()
            o = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                    do_sample=False, use_cache=True, top_p=None, top_k=None, temperature=None, **kw)
            torch.cuda.synchronize(); self.seconds += time.monotonic() - t
            new = o[0, len(e):].tolist()
            self.calls += max(1, len(new)); self.sequences += 1
            self.tokens += len(e) - (P if past is not None else 0); self.generated += len(new)
            outs.append(dict(ids=new, text=self.tok.decode(new, skip_special_tokens=True), prompt_tokens=len(e),
                             cached_prefix=P if past is not None else 0))
        return outs


# ------------------------------------------------------------ Belebele
BELEBELE_INSTRUCTIONS = {
    'swe_Latn': 'Läs texten och besvara frågan. Svara endast med bokstaven A, B, C eller D.',
    'dan_Latn': 'Læs teksten og besvar spørgsmålet. Svar kun med bogstavet A, B, C eller D.',
    'nob_Latn': 'Les teksten og svar på spørsmålet. Svar bare med bokstaven A, B, C eller D.',
    'fin_Latn': 'Lue teksti ja vastaa kysymykseen. Vastaa vain kirjaimella A, B, C tai D.',
    'isl_Latn': 'Lestu textann og svaraðu spurningunni. Svaraðu aðeins með bókstafnum A, B, C eða D.',
    'eng_Latn': 'Read the passage and answer the question. Reply only with the letter A, B, C, or D.',
    'zho_Hans': '阅读短文并回答问题。只回答字母 A、B、C 或 D。',
}
BELEBELE_TEMPLATE = 'belebele_native_instruction_chat_nothink_label_logp_v1'


def belebele_prompt(tok, lang, r):
    body = BELEBELE_INSTRUCTIONS[lang] + '\n\n' + r['flores_passage'] + '\n\n' + r['question'] + '\n'
    body += '\n'.join(f'{L}. {r["mc_answer" + str(i + 1)]}' for i, L in enumerate('ABCD'))
    return chat_prompt(tok, [{'role': 'user', 'content': body}])


def eval_belebele(scorer, rows, condition):
    out = []
    for r in rows:
        prompt = belebele_prompt(scorer.tok, r['language'], r['row'])
        sc, n = scorer.choice_scores(prompt, ['A', 'B', 'C', 'D'])
        pred = int(np.argmax(sc))
        out.append(dict(id=r['id'], benchmark='belebele', language=r['language'], split=r['split'], tier=r['tier'],
                        group=r['passage'], condition=condition, gold=r['gold'], prediction=pred,
                        correct=int(pred == r['gold']), scores=sc, prompt_tokens=n,
                        gold_margin=sc[r['gold']] - max(s for k, s in enumerate(sc) if k != r['gold']),
                        prompt_template=BELEBELE_TEMPLATE))
    return out


# ------------------------------------------------------------ ScaLA
SCALA_TEMPLATE = 'scala_euroeval_wording_12shot_greedy_strict_v1'
SCALA_MAX_NEW = 50


def scala_prompts_config():
    import ast
    from .common import ROOT
    tree = ast.parse((ROOT / 'data/euroeval_prompts/linguistic_acceptability.py').read_text())
    wanted = {'SWEDISH': 'sv', 'DANISH': 'da', 'NORWEGIAN_BOKMÅL': 'nb', 'ENGLISH': 'en'}
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Name) and k.id in wanted:
                    f = {}
                    for kw in v.keywords:
                        if kw.arg == 'default_prompt_label_mapping':
                            f[kw.arg] = {x.arg: ast.literal_eval(x.value) for x in kw.value.keywords}
                        else:
                            try:
                                f[kw.arg] = ast.literal_eval(kw.value)
                            except ValueError:
                                pass
                    out[wanted[k.id]] = f
    assert set(out) == set(wanted.values())
    return out


def scala_parse(text, mapping):
    """Predeclared strict parser: whole stripped, casefolded output equals a label, optionally + '.'/'!'."""
    clean = re.sub(r'[.!]$', '', text.strip().casefold()).strip()
    if clean == mapping['correct'].casefold():
        return 1
    if clean == mapping['incorrect'].casefold():
        return 0
    return -1


def scala_messages(lang, shots, text):
    cfg = scala_prompts_config()[lang]
    mp = cfg['default_prompt_label_mapping']
    labels_str = ', '.join(repr(mp[x]) for x in ['correct', 'incorrect'])

    def q(t):
        return cfg['default_instruction_prompt'].format(text=t, labels_str=labels_str)
    msgs = []
    for s in shots:
        msgs += [{'role': 'user', 'content': q(s['text'])}, {'role': 'assistant', 'content': mp[s['label']]}]
    return msgs + [{'role': 'user', 'content': q(text)}], mp


def eval_scala(scorer, rows, shots, condition):
    if not rows:
        return []
    lang = rows[0]['language']
    prompts, mp = [], None
    for r in rows:
        msgs, mp = scala_messages(lang, shots, r['text'])
        prompts.append(chat_prompt(scorer.tok, msgs))
    gens = scorer.generate(prompts, SCALA_MAX_NEW)
    out = []
    for r, g in zip(rows, gens):
        pred = scala_parse(g['text'], mp)
        out.append(dict(id=r['id'], benchmark='scala', language=lang, split=r['split'], tier=r['tier'],
                        group=r['group'], corruption_type=r['corruption_type'], condition=condition, gold=r['gold'],
                        prediction=pred, correct=int(pred == r['gold']), invalid=pred == -1, raw_output=g['text'],
                        output_ids=g['ids'], truncated=len(g['ids']) >= SCALA_MAX_NEW,
                        prompt_tokens=g['prompt_tokens'], prompt_template=SCALA_TEMPLATE))
    return out


def scala_metrics(recs):
    gold = [r['gold'] for r in recs]; pred = [r['prediction'] for r in recs]
    return dict(n=len(recs), accuracy=float(np.mean(np.array(gold) == np.array(pred))),
                mcc=float(matthews_corrcoef(gold, pred)) if len(set(gold)) > 1 else float('nan'),
                macro_f1=float(f1_score(gold, pred, labels=[0, 1], average='macro', zero_division=0)),
                invalid_rate=float(np.mean(np.array(pred) == -1)))


def pair_metrics(recs):
    from collections import defaultdict
    cats, rcats = defaultdict(list), defaultdict(list)
    for r in recs:
        cats[r['phenomenon']].append(r['correct'])
        if 'raw_correct' in r:
            rcats[r['phenomenon']].append(r['raw_correct'])
    eq = [r for r in recs if r['good']['tokens'] == r['bad']['tokens']]
    out = dict(n=len(recs), macro_accuracy=float(np.mean([np.mean(v) for v in cats.values()])),
               micro_accuracy=float(np.mean([r['correct'] for r in recs])),
               mean_margin=float(np.mean([r['margin'] for r in recs])),
               total_logp_accuracy=float(np.mean([r['total_margin'] > 0 for r in recs])),
               equal_length_n=len(eq), equal_length_accuracy=float(np.mean([r['correct'] for r in eq])) if eq else None,
               ties=int(sum(r['tie'] for r in recs)),
               per_phenomenon={k: float(np.mean(v)) for k, v in sorted(cats.items())})
    if rcats:
        out.update(raw_macro_accuracy=float(np.mean([np.mean(v) for v in rcats.values()])),
                   raw_mean_margin=float(np.mean([r['raw_margin'] for r in recs])))
    return out


def belebele_metrics(recs):
    return dict(n=len(recs), accuracy=float(np.mean([r['correct'] for r in recs])),
                mean_gold_margin=float(np.mean([r['gold_margin'] for r in recs])))
