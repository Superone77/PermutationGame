from __future__ import annotations
import os
import re
import json
import argparse
from typing import Dict, List, Tuple
import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.cluster import KMeans
from utils import nvfp4_blockwise_quant, sanitize_name, write_channel_stats_csv, write_block_mse_csv, visualize_modules

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


def tensor_calc_reorder_index(xmax, xmin, n_clusters, n_heads=None):
    if n_heads is None:
        n_heads = 1
    if isinstance(xmax, list):
        n = len(xmax)
        xmax = torch.cat([_.unsqueeze(-1) for _ in xmax], -1)
        xmin = torch.cat([_.unsqueeze(-1) for _ in xmin], -1)
        npdatamax = xmax.view(n_heads, -1, n).cpu().numpy()
        npdatamin = xmin.view(n_heads, -1, n).cpu().numpy()
    else:
        npdatamax = xmax.view(n_heads, -1, 1).cpu().numpy()
        npdatamin = xmin.view(n_heads, -1, 1).cpu().numpy()
    npdata = np.concatenate([npdatamax, npdatamin], -1)
    cnt = 0
    all_index = []
    all_counts = []
    for data in npdata:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(data)
        counts = np.bincount(kmeans.labels_)
        labels = torch.from_numpy(kmeans.labels_)
        index = torch.argsort(labels)
        index += cnt
        all_index.append(index)
        all_counts.append(counts)
        cnt += len(data)
    all_index = torch.hstack(all_index)
    all_counts = np.hstack(all_counts)
    return all_index, all_counts


def peg_tensor_calc_reorder_index(xmax, xmin, n_clusters, n_heads=None):
    if n_heads is None:
        n_heads = 1
    if isinstance(xmax, list):
        n = len(xmax)
        xmax = torch.cat([_.unsqueeze(-1) for _ in xmax], -1).cpu()
        xmin = torch.cat([_.unsqueeze(-1) for _ in xmin], -1).cpu()
        tdatamax = xmax.view(n_heads, -1, n)
        tdatamin = xmin.view(n_heads, -1, n)
        tdata = (tdatamax[:, :, 0] - tdatamin[:, :, 0]).reshape(n_heads, -1)
    else:
        tdatamax = xmax.view(n_heads, -1, 1).cpu()
        tdatamin = xmin.view(n_heads, -1, 1).cpu()
        tdata = (tdatamax - tdatamin).reshape(n_heads, -1)
    cnt = 0
    all_index = []
    all_counts = []
    for data in tdata:
        index = torch.argsort(data)
        counts = [int(index.numel()) // n_clusters] * n_clusters
        index = index + cnt
        all_index.append(index)
        all_counts.append(np.array(counts))
        cnt += index.numel()
    all_index = torch.hstack(all_index)
    all_counts = np.hstack(all_counts)
    return all_index, all_counts


def interval_tensor_calc_reorder_index(xmax, xmin, n_clusters, n_heads=None, key: str = 'center'):
    if n_heads is None:
        n_heads = 1
    if isinstance(xmax, list):
        n = len(xmax)
        xmax = torch.cat([_.unsqueeze(-1) for _ in xmax], -1).cpu()
        xmin = torch.cat([_.unsqueeze(-1) for _ in xmin], -1).cpu()
        tmax = xmax.view(n_heads, -1, n)[:, :, 0]
        tmin = xmin.view(n_heads, -1, n)[:, :, 0]
    else:
        tmax = xmax.view(n_heads, -1).cpu()
        tmin = xmin.view(n_heads, -1).cpu()
    center = (tmin + tmax) / 2
    width = (tmax - tmin)
    cnt = 0
    all_index = []
    all_counts = []
    for c_row, w_row, xmin_row, xmax_row in zip(center, width, tmin, tmax):
        if key == 'center':
            keys = torch.stack([c_row, w_row], dim=0)
            idx_w = torch.argsort(keys[1], stable=True)
            c_sorted = keys[0][idx_w]
            idx_c = torch.argsort(c_sorted, stable=True)
            index = idx_w[idx_c]
        else:
            keys = torch.stack([xmin_row, xmax_row], dim=0)
            idx_xmax = torch.argsort(keys[1], stable=True)
            xmin_sorted = keys[0][idx_xmax]
            idx_xmin = torch.argsort(xmin_sorted, stable=True)
            index = idx_xmax[idx_xmin]
        counts = [int(index.numel()) // n_clusters] * n_clusters
        index = index + cnt
        all_index.append(index)
        all_counts.append(np.array(counts))
        cnt += index.numel()
    all_index = torch.hstack(all_index)
    all_counts = np.hstack(all_counts)
    return all_index, all_counts


def balanced_kmeans2d_calc_reorder_index(xmax, xmin, n_clusters, block_size, n_heads=None):
    if n_heads is None:
        n_heads = 1
    if isinstance(xmax, list):
        n = len(xmax)
        xmax = torch.cat([_.unsqueeze(-1) for _ in xmax], -1)
        xmin = torch.cat([_.unsqueeze(-1) for _ in xmin], -1)
        np_xmax = xmax.view(n_heads, -1, n).cpu().numpy()
        np_xmin = xmin.view(n_heads, -1, n).cpu().numpy()
    else:
        np_xmax = xmax.view(n_heads, -1, 1).cpu().numpy()
        np_xmin = xmin.view(n_heads, -1, 1).cpu().numpy()
    cnt = 0
    all_index = []
    all_counts = []
    def _greedy_capacity(data_np: np.ndarray, K: int, cap_size: int):
        if data_np.ndim == 3:
            data_np = data_np[..., 0]
        D_head = data_np.shape[0]
        assert D_head == K * cap_size
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(data_np)
        centers = torch.from_numpy(km.cluster_centers_.astype(np.float32))
        pts = torch.from_numpy(data_np.astype(np.float32))
        dists = torch.cdist(pts, centers, p=2)
        prefs = torch.argsort(dists, dim=1).cpu().numpy()
        if K >= 2:
            d_sorted, _ = torch.sort(dists, dim=1)
            margin = (d_sorted[:, 1] - d_sorted[:, 0]).cpu().numpy()
        else:
            margin = np.full((D_head,), np.inf)
        order = np.argsort(-margin)
        cap = np.full((K,), cap_size, dtype=np.int32)
        assign = -np.ones((D_head,), dtype=np.int32)
        for i in order:
            for c in prefs[i]:
                if cap[c] > 0:
                    assign[i] = c
                    cap[c] -= 1
                    break
        assert (assign >= 0).all() and cap.sum() == 0
        return assign, centers, torch.cdist(pts, centers, p=2)
    for x_min, x_max in zip(np_xmin, np_xmax):
        data = np.concatenate([x_min, x_max], axis=-1)
        if data.ndim == 3:
            data = data[..., 0]
        D_head = data.shape[0]
        K = n_clusters
        assert D_head == K * block_size
        used_lib = False
        labels = None
        centers_t = None
        dists = None
        try:
            from kmeans_pytorch import KMeans as TorchKMeans
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            X = torch.from_numpy(data.astype(np.float32)).to(device).view(-1, 2)
            kmp = TorchKMeans(n_clusters=K, device=device, balanced=True)
            X = X.clamp(min=-1e6, max=1e6).nan_to_num(0.0)
            _ = kmp.fit(X=X, distance='euclidean', iter_limit=100, tqdm_flag=False)
            labels_t = kmp.predict(X=X)
            centers_dev = kmp.cluster_centers
            labels = labels_t.to('cpu', dtype=torch.long)
            centers_t = centers_dev.to('cpu', dtype=torch.float32)
            pts = torch.from_numpy(data.astype(np.float32))
            dists = torch.cdist(pts, centers_t, p=2)
            counts_chk = torch.bincount(labels, minlength=K).cpu().numpy()
            used_lib = np.all(counts_chk == block_size)
        except Exception:
            used_lib = False
        if not used_lib:
            assign_np, centers_t, dists = _greedy_capacity(data, K, block_size)
            labels = torch.from_numpy(assign_np)
        centers_center = centers_t.mean(dim=1)
        cluster_order = torch.argsort(centers_center).cpu().numpy().tolist()
        dnp = dists.cpu().numpy()
        idx_list = []
        counts = []
        for k in cluster_order:
            members = torch.nonzero(labels == k, as_tuple=False).view(-1).cpu().numpy()
            members = members[np.argsort(dnp[members, k])]
            idx_list.append(torch.from_numpy(members + cnt))
            counts.append(len(members))
        cnt += D_head
        all_index.append(torch.hstack(idx_list))
        all_counts.append(np.array(counts))
    all_index = torch.hstack(all_index)
    all_counts = np.hstack(all_counts)
    return all_index, all_counts


def hybrid_interval_kmeans_reorder_index(xmax: torch.Tensor, xmin: torch.Tensor, block_size: int, top_pct: float = 0.10, key: str = 'center') -> Tuple[torch.Tensor, np.ndarray]:
    xmx = xmax.view(-1).cpu()
    xmn = xmin.view(-1).cpu()
    D = xmx.numel()
    assert D == xmn.numel()
    assert D % block_size == 0
    K = D // block_size
    center = (xmn + xmx) / 2
    width = (xmx - xmn)
    if key == 'center':
        idx_w = torch.argsort(width, stable=True)
        c_sorted = center[idx_w]
        idx_c = torch.argsort(c_sorted, stable=True)
        idx_sorted = idx_w[idx_c]
    else:
        idx_xmax = torch.argsort(xmx, stable=True)
        xmin_sorted = xmn[idx_xmax]
        idx_xmin = torch.argsort(xmin_sorted, stable=True)
        idx_sorted = idx_xmax[idx_xmin]
    blocks = idx_sorted.view(K, block_size)
    top_pct = float(max(0.0, min(0.5, top_pct)))
    keep = int(round(K * top_pct))
    if keep == 0:
        idx_mid, counts_mid = balanced_kmeans2d_calc_reorder_index(xmax, xmin, K, block_size)
        return idx_mid.to(torch.long), counts_mid
    K_rem = K - 2 * keep
    if K_rem <= 0:
        return idx_sorted.to(torch.long), np.full((K,), block_size)
    left_keep = blocks[:keep].reshape(-1)
    right_keep = blocks[K - keep:].reshape(-1)
    mid_idx = blocks[keep:K - keep].reshape(-1)
    idx_mid_local, counts_mid = balanced_kmeans2d_calc_reorder_index(xmax[mid_idx], xmin[mid_idx], K_rem, block_size)
    mid_ordered = mid_idx[idx_mid_local]
    final_idx = torch.cat([left_keep, mid_ordered, right_keep], dim=0).to(torch.long)
    counts = np.hstack([np.full((keep,), block_size), counts_mid, np.full((keep,), block_size)])
    return final_idx, counts


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


@torch.no_grad()
def compute_reorders(oc_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]], block_size: int, method: str = 'peg', interval_key: str = 'center', acts_for_mse: Dict[str, torch.Tensor] | None = None, sls_iters: int = 20, sls_block_axis: str = 'feature', sls_scale_format: str = 'e4m3', hybrid_top_pct: float = 0.10) -> Dict[str, torch.Tensor]:
    perms: Dict[str, torch.Tensor] = {}
    for name, (xmax, xmin) in oc_stats.items():
        D = xmax.numel()
        if D % block_size != 0:
            continue
        n_clusters = D // block_size
        if method == 'kmeans':
            idx, counts = balanced_kmeans2d_calc_reorder_index(xmax, xmin, n_clusters, block_size)
        elif method == 'interval':
            idx, counts = interval_tensor_calc_reorder_index(xmax, xmin, n_clusters, key=interval_key)
        elif method == 'hybrid':
            idx, counts = hybrid_interval_kmeans_reorder_index(xmax, xmin, block_size, top_pct=hybrid_top_pct, key=interval_key)
        else:
            idx, counts = peg_tensor_calc_reorder_index(xmax, xmin, n_clusters)
        if idx.numel() != D:
            continue
        perms[name] = idx.to(torch.long)
    return perms


@torch.no_grad()
def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


@torch.no_grad()
def evaluate_nvfp4_mse(acts: Dict[str, torch.Tensor], perms: Dict[str, torch.Tensor], block_size: int, scale_format: str, block_axis: str = 'feature', csv_dir: str | None = None, save_csv: bool = False) -> Dict[str, Dict[str, float]]:
    results = {}
    for name, X in acts.items():
        N, D = X.shape
        if block_axis == 'feature' and D % block_size != 0:
            continue
        if block_axis == 'token' and N < block_size:
            continue
        if block_axis == 'feature':
            Dm = (D // block_size) * block_size
            Xw = X[:, :Dm]
        else:
            Nm = (N // block_size) * block_size
            Xw = X[:Nm, :]
        X_q = nvfp4_blockwise_quant(Xw, block_size=block_size, scale_format=scale_format, block_axis=block_axis)
        mse_before = mse(Xw, X_q).item()
        P = perms.get(name)
        if block_axis == 'feature':
            Xr = Xw.index_select(dim=-1, index=P) if P is not None else Xw
        else:
            Xr = Xw
        Xr_q = nvfp4_blockwise_quant(Xr, block_size=block_size, scale_format=scale_format, block_axis=block_axis)
        mse_after = mse(Xr, Xr_q).item()
        if save_csv and csv_dir is not None:
            mod_dir = os.path.join(csv_dir, sanitize_name(name))
            write_channel_stats_csv(os.path.join(mod_dir, 'stats_before.csv'), os.path.join(mod_dir, 'stats_after.csv'), Xw, P if P is not None else torch.arange(Xw.shape[1]))
            if block_axis == 'feature':
                B = Xw.shape[1] // block_size
                err_b = (Xw.view(-1, B, block_size) - X_q.view(-1, B, block_size)) ** 2
                mse_blocks_before = err_b.mean(dim=(0, 2)).cpu().numpy()
                err_a = (Xr.view(-1, B, block_size) - Xr_q.view(-1, B, block_size)) ** 2
                mse_blocks_after = err_a.mean(dim=(0, 2)).cpu().numpy()
            else:
                B = Xw.shape[0] // block_size
                err_b = (Xw.view(B, block_size, -1) - X_q.view(B, block_size, -1)) ** 2
                mse_blocks_before = err_b.mean(dim=(1, 2)).cpu().numpy()
                err_a = (Xr.view(B, block_size, -1) - Xr_q.view(B, block_size, -1)) ** 2
                mse_blocks_after = err_a.mean(dim=(1, 2)).cpu().numpy()
            write_block_mse_csv(os.path.join(mod_dir, 'block_mse.csv'), mse_blocks_before, mse_blocks_after)
        results[name] = {'N': float(Xw.shape[0]), 'D': float(Xw.shape[1]), 'mse_before': mse_before, 'mse_after': mse_after, 'improvement_%': (mse_before - mse_after) / max(mse_before, 1e-12) * 100.0}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', type=str, default='meta-llama/Meta-Llama-3.1-8B')
    ap.add_argument('--cache-dir', type=str, default='./cache')
    ap.add_argument('--dataset', type=str, default='wikitext')
    ap.add_argument('--dataset-name', type=str, default='wikitext-2-raw-v1')
    ap.add_argument('--split', type=str, default='validation')
    ap.add_argument('--seq-len', type=int, default=2048)
    ap.add_argument('--max-samples', type=int, default=16)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    ap.add_argument('--dtype', type=str, default='float16', choices=['float32', 'float16', 'bfloat16', 'fp32', 'fp16', 'bf16'])
    ap.add_argument('--scale-format', type=str, default='e4m3', choices=['e8m0', 'e4m3', 'ue5m3', 'bf16'])
    ap.add_argument('--block-size', type=int, default=16)
    ap.add_argument('--max-act-rows', type=int, default=32768)
    ap.add_argument('--reorder-method', type=str, default='hybrid', choices=['hybrid','kmeans','interval','peg'])
    ap.add_argument('--interval-key', type=str, default='center', choices=['center','lexi'])
    ap.add_argument('--hybrid-top-pct', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--viz', action='store_true', default=True)
    ap.add_argument('--viz-dir', type=str, default=None)
    ap.add_argument('--viz-max-rows', type=int, default=1024)
    ap.add_argument('--viz-bins', type=int, default=200)
    ap.add_argument('--viz-clip-pct', type=float, default=99.5)
    ap.add_argument('--viz-logy', action='store_true', default=False)
    ap.add_argument('--block-axis', type=str, default='token', choices=['token','feature'])
    ap.add_argument('--csv', action='store_true', default=True)
    ap.add_argument('--csv-dir', type=str, default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.cache_dir, exist_ok=True)
    stats_path = os.path.join(args.cache_dir, 'llama31_8b_layer4_all_acts.pt')
    perm_path = os.path.join(args.cache_dir, f'rptq_layer4_block{args.block_size}_perms.pt')
    device = pick_device(args.device)
    dtype = str2dtype(args.dtype)
    oc_stats, ic_stats, acts = collect_block4_stats_and_samples(model_id=args.model_id, cache_path=stats_path, device=device, dtype=dtype, seq_len=args.seq_len, max_samples=args.max_samples, batch_size=args.batch_size, dataset=args.dataset, dataset_name=args.dataset_name, split=args.split, max_act_rows=args.max_act_rows)
    perms = compute_reorders(oc_stats, block_size=args.block_size, method=args.reorder_method, interval_key=args.interval_key, acts_for_mse=acts, sls_iters=0, sls_block_axis='feature', sls_scale_format=args.scale_format, hybrid_top_pct=args.hybrid_top_pct)
    torch.save(perms, perm_path)
    csv_dir = args.csv_dir or os.path.join(args.cache_dir, 'csv_layer4')
    results = evaluate_nvfp4_mse(acts, perms, block_size=args.block_size, scale_format=args.scale_format, block_axis=args.block_axis, csv_dir=csv_dir, save_csv=args.csv)
    if args.viz:
        viz_dir = args.viz_dir or os.path.join(args.cache_dir, 'viz_layer4')
        visualize_modules(acts, perms, viz_dir, block_size=args.block_size, max_rows=args.viz_max_rows, bins=args.viz_bins, clip_pct=args.viz_clip_pct, logy=args.viz_logy, block_axis=args.block_axis)
    header = f"{'name':40s}  {'N':>7s} {'D':>6s}  {'MSE_before':>12s}  {'MSE_after':>12s}  {'improv%':>8s}"
    print(header)
    print('-' * len(header))
    for name in sorted(results.keys()):
        r = results[name]
        print(f"{name:40s}  {int(r['N']):7d} {int(r['D']):6d}  {r['mse_before']:12.6e}  {r['mse_after']:12.6e}  {r['improvement_%']:8.2f}")
    json_path = os.path.join(args.cache_dir, 'nvfp4_mse_report.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
