from __future__ import annotations
import os
import torch
from torch import nn
from typing import Dict, List, Tuple
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

otc: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
icc: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
oc_dbg: Dict[str, List[torch.Tensor]] = {}


def layer_omax_hook(m, i, o):
    n = m.name
    if not isinstance(o, torch.Tensor):
        return
    if o.ndim == 3:
        xmax = torch.amax(o, [0, 1])
        xmin = torch.amin(o, [0, 1])
    elif o.ndim == 2:
        xmax = torch.amax(o, [0])
        xmin = torch.amin(o, [0])
    else:
        return
    if n not in otc:
        otc[n] = (xmax.detach_(), xmin.detach_())
    else:
        otc[n] = (torch.max(otc[n][0], xmax).detach_(), torch.min(otc[n][1], xmin).detach_())


def layer_i0max_hook(m, i, o):
    n = m.name
    if len(i) == 0 or not isinstance(i[0], torch.Tensor):
        return
    if i[0].ndim == 3:
        xmax = torch.amax(i[0], [0, 1])
        xmin = torch.amin(i[0], [0, 1])
    elif i[0].ndim == 2:
        xmax = torch.amax(i[0], [0])
        xmin = torch.amin(i[0], [0])
    else:
        return
    if n not in icc:
        icc[n] = (xmax.detach_(), xmin.detach_())
    else:
        icc[n] = (torch.max(icc[n][0], xmax).detach_(), torch.min(icc[n][1], xmin).detach_())


class ActSampler:
    def __init__(self, cap_rows: int = 32768):
        self.cap_rows = cap_rows
        self.data: Dict[str, torch.Tensor] = {}
    
    def _append(self, name: str, out: torch.Tensor):
        if not isinstance(out, torch.Tensor):
            return
        if out.ndim == 3:
            B, T, D = out.shape
            mat = out.reshape(B * T, D)
        elif out.ndim == 2:
            mat = out
        else:
            return
        mat = mat.detach().to(torch.float32).cpu()
        cur = self.data.get(name)
        if cur is None:
            self.data[name] = mat[: self.cap_rows]
        else:
            remain = max(0, self.cap_rows - cur.shape[0])
            if remain > 0:
                self.data[name] = torch.cat([cur, mat[:remain]], dim=0)
    
    def hook(self, m, i, o):
        self._append(m.name, o)


def iter_block_modules(block: nn.Module) -> Dict[str, nn.Module]:
    t = {}
    t['input_layernorm'] = block.input_layernorm
    t['post_attention_layernorm'] = block.post_attention_layernorm
    attn = block.self_attn
    t['self_attn.q_proj'] = attn.q_proj
    t['self_attn.k_proj'] = attn.k_proj
    t['self_attn.v_proj'] = attn.v_proj
    t['self_attn.o_proj'] = attn.o_proj
    mlp = block.mlp
    t['mlp.gate_proj'] = mlp.gate_proj
    t['mlp.up_proj'] = mlp.up_proj
    t['mlp.down_proj'] = mlp.down_proj
    return t


@torch.no_grad()
def pick_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_arg)


def str2dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s in ['float32', 'fp32']:
        return torch.float32
    if s in ['float16', 'fp16']:
        return torch.float16
    if s in ['bfloat16', 'bf16']:
        return torch.bfloat16
    return torch.float32


@torch.no_grad()
def build_wikitext_calib(tokenizer: AutoTokenizer, dataset: str, dataset_name: str, split: str, seq_len: int, max_samples: int) -> List[torch.Tensor]:
    ds = load_dataset(dataset, dataset_name, split=split)
    texts = ds['text']
    joined = "\n\n".join(texts)
    toks = tokenizer(joined, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
    chunks = []
    i = 0
    while i + seq_len <= toks.numel() and len(chunks) < max_samples:
        chunks.append(toks[i:i + seq_len].clone())
        i += seq_len
    return chunks


@torch.no_grad()
def collect_block4_stats_and_samples(model_id: str, cache_path: str, device: torch.device, dtype: torch.dtype, seq_len: int, max_samples: int, batch_size: int, dataset: str, dataset_name: str, split: str, max_act_rows: int) -> Tuple[Dict, Dict, Dict]:
    if os.path.exists(cache_path):
        pack = torch.load(cache_path, map_location='cpu')
        return pack['oc_stats'], pack['ic_stats'], pack['acts']
    
    otc.clear(); icc.clear(); oc_dbg.clear()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map='auto' if device.type == 'cuda' else None)
    if device.type != 'cuda':
        model.to(device)
    model.eval()
    inputs = build_wikitext_calib(tok, dataset, dataset_name, split, seq_len, max_samples)
    layer_idx = 3
    block = model.model.layers[layer_idx]
    modules = iter_block_modules(block)
    sampler = ActSampler(cap_rows=max_act_rows)
    handles = []
    for name, mod in modules.items():
        mod.name = name
        def combined_hook(m, i, o):
            layer_omax_hook(m, i, o)
            layer_i0max_hook(m, i, o)
            sampler.hook(m, i, o)
        handles.append(mod.register_forward_hook(combined_hook))
    for i in range(0, len(inputs), batch_size):
        batch_ids = inputs[i:i + batch_size]
        max_len = max(x.numel() for x in batch_ids)
        batch = torch.stack([torch.nn.functional.pad(x, (0, max_len - x.numel()), value=tok.pad_token_id) for x in batch_ids], dim=0).to(device)
        _ = model(batch)
    for h in handles:
        h.remove()
    oc_stats = {k: (v[0].cpu(), v[1].cpu()) for k, v in otc.items()}
    ic_stats = {k: (v[0].cpu(), v[1].cpu()) for k, v in icc.items()}
    acts = {k: v.cpu() for k, v in sampler.data.items()}
    torch.save({'oc_stats': oc_stats, 'ic_stats': ic_stats, 'acts': acts}, cache_path)
    return oc_stats, ic_stats, acts
