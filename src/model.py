"""Gemma 4 12B IT loading in float32 (weights and compute), text-only, eager PyTorch."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import platform
import time

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForMultimodalLM

from .common import ROOT, read, write, now

MODEL_PATH = ROOT / 'models/gemma-4-12B-it'
REVISION = '43c9d014b231474fa7212dbb48a762566c2b255d'


def load(attn='sdpa'):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    t = time.monotonic()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tok.padding_side = 'right'
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_PATH, dtype=torch.float32, device_map={'': 'cuda:0'},
                                                     attn_implementation=attn, local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    layers = model.model.language_model.layers
    meta = dict(utc=now(), python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                transformers=transformers.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                dtype='float32 weights and compute (BF16 checkpoint values upcast exactly)', attention=attn,
                tf32=False, execution='eager, no torch.compile', class_=type(model).__name__,
                n_layers=len(layers), load_seconds=time.monotonic() - t, model_revision=REVISION,
                parameters=sum(p.numel() for p in model.parameters()),
                text_parameters=sum(p.numel() for p in model.model.language_model.parameters()),
                mlp_shapes={k: list(getattr(layers[0].mlp, k).weight.shape) for k in ('gate_proj', 'up_proj', 'down_proj')},
                bos=tok.bos_token_id, eos=tok.eos_token_id, pad=tok.pad_token_id)
    write('manifests/environment.json', meta)
    print('MODEL READY', meta['load_seconds'], flush=True)
    return model, tok


def text_layers(model):
    return model.model.language_model.layers


def chat_prompt(tok, messages):
    """Final-answer context with thinking disabled via the template's supported switch."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert text.endswith('<|channel>thought\n<channel|>'), repr(text[-120:])
    assert '<|think|>' not in text
    return text
