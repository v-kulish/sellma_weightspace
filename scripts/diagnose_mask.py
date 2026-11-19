import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import transformers.models.gemma4_unified.modeling_gemma4_unified as G
orig = G.create_masks_for_generate
def wrapped(**kw):
    out = orig(**kw)
    if kw['inputs_embeds'].shape[1] <= 8:
        for k, v in (out.items() if isinstance(out, dict) else [('m', out)]):
            print('MASK', k, type(v), None if v is None else (v.dtype, v.shape)); 
            if v is not None and hasattr(v, 'shape'): print(v[0, 0].int() if v.dtype == torch.bool else (v[0,0] > -1).int())
    return out
G.create_masks_for_generate = wrapped
from src.model import load
model, tok = load()
ids = [2, 105, 2364, 107]  # <bos><|turn>user\n
x = torch.tensor([ids], device='cuda')
def lp_user(**kw):
    with torch.inference_mode():
        lg = model(input_ids=x, use_cache=False, **kw).logits[0].float()
    lp = torch.log_softmax(lg, -1)
    return [round(lp[t, ids[t+1]].item(), 3) for t in range(len(ids)-1)], [tok.decode([int(lg[t].argmax())]) for t in range(len(ids))]
print('with mask', lp_user(attention_mask=torch.ones_like(x)))
print('no mask', lp_user())
print('mm zeros', lp_user(attention_mask=torch.ones_like(x), mm_token_type_ids=torch.zeros_like(x)))
