#!/usr/bin/env python3
"""
Standalone visualizer for RPTQ/NVFP4 CSV outputs.

Reads per-module CSVs produced by your main script and generates intuitive plots:
  - Per-block MSE Top-K bar chart (before vs after)
  - Per-block MSE CDF (before vs after)
  - Per-block MSE scatter (before vs after, with y=x)
  - Channel intervals after reorder: center ± half-width vs channel position (sampled)
  - (amin, amax) 2D scatter colored by block id (sampled)
  - Per-block improvement histogram ((before-after)/before)

Usage example:
  python rptq_csv_visualizer.py \
    --csv-dir ./cache/csv_layer4 \
    --out-dir ./cache/viz_from_csv \
    --block-size 16 --topk 20 --maxpoints 4096 --style darkgrid

CSV layout (per module directory):
  stats_before.csv  -> channel_idx, amin, amax
  stats_after.csv   -> position_after, orig_channel, amin, amax
  block_mse.csv     -> block_id, mse_before, mse_after
"""
from __future__ import annotations

import os
import csv
import argparse
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ------------------------------ IO helpers ------------------------------

def _read_rows(path: str) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, 'r') as f:
        r = csv.reader(f)
        _ = next(r, None)  # skip header
        for row in r:
            if row:
                rows.append(row)
    return rows


def _load_module_arrays(mod_dir: str):
    sb = os.path.join(mod_dir, 'stats_before.csv')
    sa = os.path.join(mod_dir, 'stats_after.csv')
    bm = os.path.join(mod_dir, 'block_mse.csv')
    if not (os.path.exists(sb) and os.path.exists(sa) and os.path.exists(bm)):
        return None
    b_rows = _read_rows(sb)
    a_rows = _read_rows(sa)
    m_rows = _read_rows(bm)

    amin_b = np.array([float(r[1]) for r in b_rows], dtype=np.float64)
    amax_b = np.array([float(r[2]) for r in b_rows], dtype=np.float64)
    pos_a  = np.array([int(r[0])   for r in a_rows], dtype=np.int64)
    orig_a = np.array([int(r[1])   for r in a_rows], dtype=np.int64)
    amin_a = np.array([float(r[2]) for r in a_rows], dtype=np.float64)
    amax_a = np.array([float(r[3]) for r in a_rows], dtype=np.float64)
    mse_b  = np.array([float(r[1]) for r in m_rows], dtype=np.float64)
    mse_a  = np.array([float(r[2]) for r in m_rows], dtype=np.float64)
    return (amin_b, amax_b, pos_a, orig_a, amin_a, amax_a, mse_b, mse_a)


# ------------------------------ Plot helpers ------------------------------

def _maybe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def _format_axes(ax, grid: bool = True):
    if grid:
        ax.grid(True, which='major', ls='--', alpha=0.3)
    ax.tick_params(axis='both', labelsize=9)


def plot_block_mse_topk(mse_b: np.ndarray, mse_a: np.ndarray, out_path: str, topk: int = 20):
    B = len(mse_b)
    if B == 0:
        return
    idx = np.argsort(-mse_b)[:min(topk, B)]
    x = np.arange(idx.size)
    fig, ax = plt.subplots(figsize=(max(6, 0.35*idx.size+2), 3.8), constrained_layout=True)
    ax.bar(x - 0.18, mse_b[idx], width=0.36, label='before')
    ax.bar(x + 0.18, mse_a[idx], width=0.36, label='after')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in idx], rotation=45, ha='right')
    ax.set_xlabel('block id (worst-K by before-MSE)')
    ax.set_ylabel('MSE')
    ax.set_title('Per-block MSE (Top-K)')
    ax.legend(frameon=False)
    _format_axes(ax)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_block_mse_cdf(mse_b: np.ndarray, mse_a: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for arr, lab in [(np.sort(mse_b), 'before'), (np.sort(mse_a), 'after')]:
        if arr.size == 0:
            continue
        y = np.linspace(0, 1, arr.size, endpoint=False)
        ax.plot(arr, y, label=lab)
    ax.set_xlabel('MSE')
    ax.set_ylabel('CDF')
    ax.set_title('Per-block MSE CDF')
    ax.legend(frameon=False)
    _format_axes(ax)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_block_mse_scatter(mse_b: np.ndarray, mse_a: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(mse_b, mse_a, s=10, alpha=0.5, linewidths=0)
    lim = (0, max(1e-12, mse_b.max() if mse_b.size else 0, mse_a.max() if mse_a.size else 0))
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('MSE before')
    ax.set_ylabel('MSE after')
    ax.set_title('Per-block MSE (scatter)')
    _format_axes(ax)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_block_improvement_hist(mse_b: np.ndarray, mse_a: np.ndarray, out_path: str):
    if mse_b.size == 0:
        return
    imp = (mse_b - mse_a) / np.maximum(mse_b, 1e-12)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(imp, bins=40, alpha=0.8, color='C2')
    ax.set_xlabel('improvement = (before - after) / before')
    ax.set_ylabel('count of blocks')
    ax.set_title('Per-block improvement histogram')
    _format_axes(ax)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_intervals_after(pos_a: np.ndarray, amin_a: np.ndarray, amax_a: np.ndarray,
                         block_size: int, out_path: str, maxpoints: int = 4096, alpha: float = 0.25):
    centers = (amin_a + amax_a) / 2.0
    widths = (amax_a - amin_a) / 2.0
    D = centers.shape[0]
    if D == 0:
        return
    step = max(1, D // maxpoints)
    idx = np.arange(0, D, step)

    fig, ax = plt.subplots(figsize=(12, 3.6), constrained_layout=True)
    ax.errorbar(idx, centers[idx], yerr=widths[idx], fmt='.', color='C0', alpha=alpha, markersize=2)
    ax.set_xlabel('channel position (after reorder)')
    ax.set_ylabel('interval center ± half-width')
    ax.set_title('Channel intervals after reorder (sampled)')
    # Block grid
    for c in range(block_size, D, block_size):
        ax.axvline(c - 0.5, color='k', alpha=0.15, lw=0.5)
    _format_axes(ax)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_amin_amax_scatter(amin_a: np.ndarray, amax_a: np.ndarray, block_size: int, out_path: str,
                           maxpoints: int = 4096, alpha: float = 0.25):
    D = amin_a.size
    if D == 0:
        return
    step = max(1, D // maxpoints)
    idx = np.arange(0, D, step)
    block_ids = (np.arange(D) // block_size)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    sc = ax.scatter(amin_a[idx], amax_a[idx], c=block_ids[idx], cmap='tab20', s=6, alpha=alpha, linewidths=0)
    ax.set_xlabel('amin')
    ax.set_ylabel('amax')
    ax.set_title('(amin, amax) scatter colored by block id')
    _format_axes(ax)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


# ------------------------------ Driver ------------------------------

def process_one_module(mod_name: str, mod_dir: str, out_dir: str, block_size: int, topk: int, maxpoints: int):
    arrs = _load_module_arrays(mod_dir)
    if arrs is None:
        return False
    amin_b, amax_b, pos_a, orig_a, amin_a, amax_a, mse_b, mse_a = arrs

    m_out = os.path.join(out_dir, mod_name)
    _maybe_mkdir(m_out)

    plot_block_mse_topk(mse_b, mse_a, os.path.join(m_out, 'block_mse_topk.png'), topk=topk)
    plot_block_mse_cdf(mse_b, mse_a, os.path.join(m_out, 'block_mse_cdf.png'))
    plot_block_mse_scatter(mse_b, mse_a, os.path.join(m_out, 'block_mse_scatter.png'))
    plot_block_improvement_hist(mse_b, mse_a, os.path.join(m_out, 'block_improvement_hist.png'))

    plot_intervals_after(pos_a, amin_a, amax_a, block_size, os.path.join(m_out, 'intervals_after.png'), maxpoints=maxpoints)
    plot_amin_amax_scatter(amin_a, amax_a, block_size, os.path.join(m_out, 'amin_amax_scatter.png'), maxpoints=maxpoints)

    return True


def main():
    ap = argparse.ArgumentParser(description='Visualize RPTQ/NVFP4 CSV outputs')
    ap.add_argument('--csv-dir', type=str, required=True, help='directory that contains per-module CSV folders')
    ap.add_argument('--out-dir', type=str, required=True, help='directory to save generated figures')
    ap.add_argument('--block-size', type=int, default=16)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--maxpoints', type=int, default=4096, help='sampling cap for large-D plots')
    ap.add_argument('--style', type=str, default='default', choices=['default','darkgrid','seaborn','ggplot'])
    ap.add_argument('--dpi', type=int, default=160)
    args = ap.parse_args()

    # Style
    if args.style == 'darkgrid':
        plt.style.use('seaborn-v0_8-darkgrid')
    elif args.style == 'seaborn':
        plt.style.use('seaborn-v0_8')
    elif args.style == 'ggplot':
        plt.style.use('ggplot')

    _maybe_mkdir(args.out_dir)

    # Per-module processing
    modules = [d for d in os.listdir(args.csv_dir) if os.path.isdir(os.path.join(args.csv_dir, d))]
    modules.sort()

    any_ok = False
    for mod in modules:
        ok = process_one_module(mod, os.path.join(args.csv_dir, mod), args.out_dir, args.block_size, args.topk, args.maxpoints)
        any_ok = any_ok or ok

    if not any_ok:
        # no modules processed; create a sentinel empty file
        open(os.path.join(args.out_dir, 'EMPTY.txt'), 'w').close()


if __name__ == '__main__':
    main()
