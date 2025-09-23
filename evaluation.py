from __future__ import annotations
import os
import torch
from typing import Dict, Tuple
import numpy as np
from utils import nvfp4_blockwise_quant, sanitize_name, write_channel_stats_csv, write_block_mse_csv


@torch.no_grad()
def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


@torch.no_grad()
def evaluate_nvfp4_mse(acts: Dict[str, torch.Tensor], perms: Dict[str, torch.Tensor], block_size: int, scale_format: str, block_axis: str = 'feature', csv_dir: str | None = None, save_csv: bool = False, partial_quant: bool = False, top_pct: float = 0.10) -> Dict[str, Dict[str, float]]:
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
        
        P = perms.get(name)
        if P is None:
            P = torch.arange(Xw.shape[1] if block_axis == 'feature' else Xw.shape[0])
        
        if partial_quant and block_axis == 'feature':
            n_blocks = Dm // block_size
            keep_blocks = int(n_blocks * top_pct)
            keep_channels = keep_blocks * block_size
            
            Xr = Xw.index_select(dim=-1, index=P)
            Xr_keep = Xr[:, :keep_channels]
            Xr_quant = Xr[:, keep_channels:]
            
            X_q_keep = Xr_keep
            X_q_quant = nvfp4_blockwise_quant(Xr_quant, block_size=block_size, scale_format=scale_format, block_axis=block_axis)
            
            X_q = torch.cat([X_q_keep, X_q_quant], dim=-1)
            Xr_full = Xr
        else:
            if block_axis == 'feature':
                Xr = Xw.index_select(dim=-1, index=P) if P is not None else Xw
            else:
                Xr = Xw
            X_q = nvfp4_blockwise_quant(Xr, block_size=block_size, scale_format=scale_format, block_axis=block_axis)
            Xr_full = Xr
        
        mse_before = mse(Xw, nvfp4_blockwise_quant(Xw, block_size=block_size, scale_format=scale_format, block_axis=block_axis)).item()
        mse_after = mse(Xr_full, X_q).item()
        
        if save_csv and csv_dir is not None:
            mod_dir = os.path.join(csv_dir, sanitize_name(name))
            write_channel_stats_csv(os.path.join(mod_dir, 'stats_before.csv'), os.path.join(mod_dir, 'stats_after.csv'), Xw, P if P is not None else torch.arange(Xw.shape[1]))
            if block_axis == 'feature':
                B = Xw.shape[1] // block_size
                err_b = (Xw.view(-1, B, block_size) - nvfp4_blockwise_quant(Xw, block_size=block_size, scale_format=scale_format, block_axis=block_axis).view(-1, B, block_size)) ** 2
                mse_blocks_before = err_b.mean(dim=(0, 2)).cpu().numpy()
                err_a = (Xr_full.view(-1, B, block_size) - X_q.view(-1, B, block_size)) ** 2
                mse_blocks_after = err_a.mean(dim=(0, 2)).cpu().numpy()
            else:
                B = Xw.shape[0] // block_size
                err_b = (Xw.view(B, block_size, -1) - nvfp4_blockwise_quant(Xw, block_size=block_size, scale_format=scale_format, block_axis=block_axis).view(B, block_size, -1)) ** 2
                mse_blocks_before = err_b.mean(dim=(1, 2)).cpu().numpy()
                err_a = (Xr_full.view(B, block_size, -1) - X_q.view(B, block_size, -1)) ** 2
                mse_blocks_after = err_a.mean(dim=(1, 2)).cpu().numpy()
            write_block_mse_csv(os.path.join(mod_dir, 'block_mse.csv'), mse_blocks_before, mse_blocks_after)
        
        results[name] = {
            'N': float(Xw.shape[0]), 
            'D': float(Xw.shape[1]), 
            'mse_before': mse_before, 
            'mse_after': mse_after, 
            'improvement_%': (mse_before - mse_after) / max(mse_before, 1e-12) * 100.0
        }
    return results
