
#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import math
import json
import argparse
from typing import Dict, List, Tuple

import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
from sklearn.cluster import KMeans
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ============================ NVFP4 quantization (per 16) ============================
FP8_E4M3_MAX = 448.0

@torch.no_grad()
def fp4_121_positive(x: torch.Tensor, stochastic_rounding: bool = False) -> torch.Tensor:
    if stochastic_rounding:
        noise = torch.rand_like(x) - 0.5
        step1 = torch.round(2.0 * x + noise) / 2.0
        step2 = torch.round(x + noise)
        step3 = 2.0 * torch.round(x / 2.0 + noise)
    else:
        step1 = torch.round(2.0 * x) / 2.0
        step2 = torch.round(x)
        step3 = 2.0 * torch.round(x / 2.0)
    mask1 = x < 2.0
    mask2 = x < 4.0
    return step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)

@torch.no_grad()
def ue5m3(x: torch.Tensor) -> torch.Tensor:
    mask = x <= 2 ** (-17)
    x_1 = x * mask
    x_2 = x * (~mask) + torch.ones_like(x) * mask
    x_1 = torch.round(x_1 / 2 ** (-17)) * (2 ** (-17))
    e = torch.floor(torch.log2(x_2)) - 3
    s = 2 ** e
    x_2 = torch.round(x_2 / s) * s
    return x_1 * mask + x_2 * (~mask)

@torch.no_grad()
def fp4_121_scaled(x: torch.Tensor,
                   stochastic_rounding: bool = False,
                   scale_format: str = 'e8m0') -> torch.Tensor:
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    if scale_format == 'e8m0':
        scale = torch.pow(2.0, torch.floor(torch.log2(fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0])))
    elif scale_format == 'e4m3':
        nvfp4_max = fp4_121_max * FP8_E4M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
        x_abs_scaled = x_abs / scale_per_t
        scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
        try:
            down_cast = (fp4_121_max / scale_per_b).to(torch.float8_e4m3fn)
        except Exception:
            down_cast = (fp4_121_max / scale_per_b).to(torch.float16)
        scale_per_b = down_cast.to(scale_per_b.dtype)
        scale_per_b = torch.where((0 < scale_per_b) & (scale_per_b < torch.inf), scale_per_b, torch.ones_like(scale_per_b))
        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b
        return sign * x_fp4_abs * scale_per_t
    elif scale_format == 'ue5m3':
        UE5M3_MAX = 114688.0
        nvfp4_max = fp4_121_max * UE5M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
        x_abs_scaled = x_abs / scale_per_t
        scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
        scale_per_b = ue5m3(fp4_121_max / scale_per_b)
        scale_per_b = torch.where((0 < scale_per_b) & (scale_per_b < torch.inf), scale_per_b, torch.ones_like(scale_per_b))
        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b
        return sign * x_fp4_abs * scale_per_t
    else:  # bf16
        scale = fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0]
    scale = torch.where((0 < scale) & (scale < torch.inf), scale, torch.ones_like(scale))
    x_fp4_abs = fp4_121_positive(x_abs * scale, stochastic_rounding) / scale
    return sign * x_fp4_abs

@torch.no_grad()
def nvfp4_blockwise_quant(x2d: torch.Tensor, block_size: int = 16, scale_format: str = 'e4m3',
                          stochastic_rounding: bool = False, block_axis: str = 'feature') -> torch.Tensor:
    N, H = x2d.shape
    if block_axis == 'feature':
        assert H % block_size == 0, f"Dim {H} not divisible by block_size {block_size}"
        x_view = x2d.view(N, H // block_size, block_size)
        x_q = fp4_121_scaled(x_view, stochastic_rounding=stochastic_rounding, scale_format=scale_format)
        return x_q.view(N, H)
    else:  # 'token'
        assert N % block_size == 0, f"Rows {N} not divisible by block_size {block_size}"
        xt = x2d.t().contiguous()                      # [D, N]
        xt_view = xt.view(H, N // block_size, block_size)
        xt_q = fp4_121_scaled(xt_view, stochastic_rounding=stochastic_rounding, scale_format=scale_format)
        return xt_q.view(H, N).t().contiguous()

def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:/-]", "_", name)

# ============================ CSV Export Helpers ============================

def write_channel_stats_csv(path_before: str, path_after: str, Xw: torch.Tensor, P: torch.Tensor | None):
    os.makedirs(os.path.dirname(path_before), exist_ok=True)
    # BEFORE
    amin_b = Xw.min(dim=0).values.cpu().numpy()
    amax_b = Xw.max(dim=0).values.cpu().numpy()
    with open(path_before, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['channel_idx', 'amin', 'amax'])
        for i in range(Xw.shape[1]):
            w.writerow([i, float(amin_b[i]), float(amax_b[i])])
    # AFTER (reordered positions)
    if P is None:
        P = torch.arange(Xw.shape[1])
    Xr = Xw.index_select(dim=-1, index=P)
    amin_a = Xr.min(dim=0).values.cpu().numpy()
    amax_a = Xr.max(dim=0).values.cpu().numpy()
    with open(path_after, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['position_after', 'orig_channel', 'amin', 'amax'])
        for pos in range(Xr.shape[1]):
            orig = int(P[pos])
            w.writerow([pos, orig, float(amin_a[pos]), float(amax_a[pos])])


def write_block_mse_csv(path_blocks: str, mse_before: np.ndarray, mse_after: np.ndarray):
    os.makedirs(os.path.dirname(path_blocks), exist_ok=True)
    with open(path_blocks, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['block_id', 'mse_before', 'mse_after'])
        for b in range(len(mse_before)):
            w.writerow([b, float(mse_before[b]), float(mse_after[b])])

# ============================ Visualization ============================

@torch.no_grad()
def clip_for_heatmap(M: torch.Tensor, pct: float = 99.5) -> torch.Tensor:
    """Symmetric percentile clipping for better contrast in heatmaps."""
    if M.numel() == 0:
        return M
    k = max(1, int(M.numel() * pct / 100.0))
    th = M.abs().view(-1).kthvalue(min(k, M.numel())).values.item()
    th = max(th, 1e-6)
    return torch.clamp(M, min=-th, max=th)


def draw_block_grid(ax, D: int, block: int, axis: str = 'token', N: int | None = None):
    if axis == 'feature':
        for c in range(block, D, block):
            ax.axvline(c - 0.5, color='w', alpha=0.25, lw=0.6)
    else:
        assert N is not None
        for r in range(block, N, block):
            ax.axhline(r - 0.5, color='w', alpha=0.25, lw=0.6)


@torch.no_grad()
def visualize_modules(acts: Dict[str, torch.Tensor], perms: Dict[str, torch.Tensor], out_dir: str,
                      block_size: int = 16, max_rows: int = 1024, bins: int = 200,
                      clip_pct: float = 99.5, logy: bool = False, block_axis: str = 'token'):
    os.makedirs(out_dir, exist_ok=True)
    for name, X in acts.items():
        P = perms.get(name)
        if P is None:
            continue
        N, D = X.shape
        nrows = min(N, max_rows)
        Xs = X[:nrows]
        Xrs = Xs.index_select(dim=-1, index=P)

        # Heatmaps
        Xc = clip_for_heatmap(Xs, pct=clip_pct)
        Xrc = clip_for_heatmap(Xrs, pct=clip_pct)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        im0 = axes[0].imshow(Xc.numpy(), aspect='auto', cmap='magma', interpolation='nearest')
        axes[0].set_title(f"{name} — before reorder")
        draw_block_grid(axes[0], D, block_size, axis=block_axis, N=nrows)
        im1 = axes[1].imshow(Xrc.numpy(), aspect='auto', cmap='magma', interpolation='nearest')
        axes[1].set_title(f"{name} — after reorder")
        draw_block_grid(axes[1], D, block_size, axis=block_axis, N=nrows)
        fig.colorbar(im1, ax=axes, location='right', shrink=0.8)
        plt.savefig(os.path.join(out_dir, f"heatmap_{sanitize_name(name)}.png"), dpi=160)
        plt.close(fig)

        # Histograms (value distribution)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        ax.hist(Xs.flatten().numpy(), bins=bins, density=True, alpha=0.5, label='before')
        ax.hist(Xrs.flatten().numpy(), bins=bins, density=True, alpha=0.5, label='after')
        ax.set_title(f"Value distribution — {name}")
        if logy:
            ax.set_yscale('log')
        ax.legend()
        plt.savefig(os.path.join(out_dir, f"hist_{sanitize_name(name)}.png"), dpi=160)
        plt.close(fig)