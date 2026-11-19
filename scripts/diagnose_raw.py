import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from src.model import load, chat_prompt
model, tok = load()
def show(text, add_bos=True):
    ids = tok.encode(text, add_special_tokens=False)
    if add_bos: ids = [tok.bos_token_id] + ids
    x = torch.tensor([ids], device='cuda')
    with torch.inference_mode():
        lg = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0].float()
    lp = torch.log_softmax(lg, -1)
    toks = [tok.decode([i]) for i in ids]
    print(repr(text[:60]))
    for t in range(1, len(ids)):
        top = lp[t-1].topk(3)
        print(f'  {toks[t]!r:>14} lp={lp[t-1, ids[t]].item():7.2f}  top={[(tok.decode([int(i)]), round(v.item(),2)) for v, i in zip(top.values, top.indices)]}')
show('The capital of France is Paris.')
show('Hunden sover i sin korg.')
p = chat_prompt(tok, [{'role':'user','content':'Skriv en mening om en hund.'}])
show(p + 'Hunden sover i sin korg.', add_bos=False)
print(repr(p[:40]), tok.encode(p, add_special_tokens=False)[:3], tok.bos_token_id)
x = torch.tensor([[tok.bos_token_id] + tok.encode('The capital of France is', add_special_tokens=False)], device='cuda')
o = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=12, do_sample=False)
print('GEN', repr(tok.decode(o[0])))
