import sys, pathlib, gc; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer, AutoModelForMultimodalLM
P = 'models/gemma-4-12B-it'
tok = AutoTokenizer.from_pretrained(P)
texts = ['The capital of France is Paris.', 'Hunden sover i sin korg.']
chat = tok.apply_chat_template([{'role':'user','content':'Vad är huvudstaden i Sverige? Svara kort.'}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
for dtype, attn in [(torch.float32, 'eager'), (torch.bfloat16, 'sdpa'), (torch.bfloat16, 'eager')]:
    m = AutoModelForMultimodalLM.from_pretrained(P, dtype=dtype, device_map={'':'cuda:0'}, attn_implementation=attn).eval()
    for t in texts:
        ids = [2] + tok.encode(t, add_special_tokens=False); x = torch.tensor([ids], device='cuda')
        with torch.inference_mode(): lp = torch.log_softmax(m(input_ids=x).logits[0].float(), -1)
        print(dtype, attn, t, [round(lp[i, ids[i+1]].item(), 2) for i in range(len(ids)-1)])
    x = torch.tensor([tok.encode(chat, add_special_tokens=False)], device='cuda')
    with torch.inference_mode(): o = m.generate(input_ids=x, max_new_tokens=20, do_sample=False)
    print('CHAT', repr(tok.decode(o[0, x.shape[1]:])))
    del m; gc.collect(); torch.cuda.empty_cache()
