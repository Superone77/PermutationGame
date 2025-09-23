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
import json

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


def create_summary_plots(modules: List[str], csv_dir: str, out_dir: str, block_size: int):
    """Create summary plots across all modules"""
    all_mse_before = []
    all_mse_after = []
    all_improvements = []
    module_info = []
    
    for mod in modules:
        mod_dir = os.path.join(csv_dir, mod)
        arrs = _load_module_arrays(mod_dir)
        if arrs is None:
            continue
        
        amin_b, amax_b, pos_a, orig_a, amin_a, amax_a, mse_b, mse_a = arrs
        
        if len(mse_b) > 0 and len(mse_a) > 0:
            avg_mse_b = np.mean(mse_b)
            avg_mse_a = np.mean(mse_a)
            improvement = (avg_mse_b - avg_mse_a) / max(avg_mse_b, 1e-12) * 100
            
            all_mse_before.append(avg_mse_b)
            all_mse_after.append(avg_mse_a)
            all_improvements.append(improvement)
            module_info.append(mod)
    
    if not all_mse_before:
        return
    
    # Summary MSE comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    
    # MSE before vs after scatter
    ax1.scatter(all_mse_before, all_mse_after, alpha=0.6, s=30)
    lim = (0, max(max(all_mse_before), max(all_mse_after)))
    ax1.plot(lim, lim, 'k--', alpha=0.5)
    ax1.set_xlabel('MSE before')
    ax1.set_ylabel('MSE after')
    ax1.set_title('Module-wise MSE Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Improvement histogram
    ax2.hist(all_improvements, bins=20, alpha=0.7, color='C2')
    ax2.set_xlabel('Improvement (%)')
    ax2.set_ylabel('Number of modules')
    ax2.set_title('Improvement Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(out_dir, 'summary_mse_comparison.png'), dpi=160)
    plt.close(fig)
    
    # Layer-wise analysis if applicable
    layer_stats = {}
    for i, mod in enumerate(module_info):
        if mod.startswith('layer_'):
            layer_idx = mod.split('_')[1]
            if layer_idx not in layer_stats:
                layer_stats[layer_idx] = {'mse_before': [], 'mse_after': [], 'improvements': []}
            layer_stats[layer_idx]['mse_before'].append(all_mse_before[i])
            layer_stats[layer_idx]['mse_after'].append(all_mse_after[i])
            layer_stats[layer_idx]['improvements'].append(all_improvements[i])
    
    if layer_stats:
        # Layer-wise improvement plot
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        layers = sorted(layer_stats.keys(), key=int)
        layer_avg_improvements = [np.mean(layer_stats[layer]['improvements']) for layer in layers]
        
        bars = ax.bar(layers, layer_avg_improvements, alpha=0.7, color='C1')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Average Improvement (%)')
        ax.set_title('Layer-wise Average Improvement')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, layer_avg_improvements):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.savefig(os.path.join(out_dir, 'layer_wise_improvement.png'), dpi=160)
        plt.close(fig)
    
    # Save summary statistics
    summary_stats = {
        'total_modules': len(module_info),
        'avg_mse_before': float(np.mean(all_mse_before)),
        'avg_mse_after': float(np.mean(all_mse_after)),
        'avg_improvement': float(np.mean(all_improvements)),
        'max_improvement': float(np.max(all_improvements)),
        'min_improvement': float(np.min(all_improvements)),
        'modules_with_improvement': int(np.sum(np.array(all_improvements) > 0))
    }
    
    with open(os.path.join(out_dir, 'summary_stats.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description='Visualize RPTQ/NVFP4 CSV outputs')
    ap.add_argument('--csv-dir', type=str, required=True, help='directory that contains per-module CSV folders')
    ap.add_argument('--out-dir', type=str, required=True, help='directory to save generated figures')
    ap.add_argument('--block-size', type=int, default=16)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--maxpoints', type=int, default=4096, help='sampling cap for large-D plots')
    ap.add_argument('--style', type=str, default='default', choices=['default','darkgrid','seaborn','ggplot'])
    ap.add_argument('--dpi', type=int, default=160)
    ap.add_argument('--group-by-layer', action='store_true', default=False, help='Group modules by layer for better organization')
    ap.add_argument('--max-modules', type=int, default=50, help='Maximum number of modules to process (for all-layers mode)')
    ap.add_argument('--create-summary', action='store_true', default=False, help='Create summary plots across all modules')
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

    # Limit modules for all-layers mode
    if len(modules) > args.max_modules:
        print(f"Warning: Found {len(modules)} modules, limiting to {args.max_modules}")
        modules = modules[:args.max_modules]

    if args.group_by_layer:
        # Group modules by layer
        layer_groups = {}
        for mod in modules:
            if mod.startswith('layer_'):
                layer_idx = mod.split('_')[1]
                if layer_idx not in layer_groups:
                    layer_groups[layer_idx] = []
                layer_groups[layer_idx].append(mod)
            else:
                # Handle modules without layer prefix
                if 'other' not in layer_groups:
                    layer_groups['other'] = []
                layer_groups['other'].append(mod)
        
        # Process each layer group
        any_ok = False
        for layer_idx, layer_modules in layer_groups.items():
            layer_out_dir = os.path.join(args.out_dir, f'layer_{layer_idx}')
            _maybe_mkdir(layer_out_dir)
            
            print(f"Processing layer {layer_idx} with {len(layer_modules)} modules...")
            for mod in layer_modules:
                ok = process_one_module(mod, os.path.join(args.csv_dir, mod), layer_out_dir, args.block_size, args.topk, args.maxpoints)
                any_ok = any_ok or ok
    else:
        # Original processing without grouping
        any_ok = False
        for mod in modules:
            ok = process_one_module(mod, os.path.join(args.csv_dir, mod), args.out_dir, args.block_size, args.topk, args.maxpoints)
            any_ok = any_ok or ok

    if not any_ok:
        # no modules processed; create a sentinel empty file
        open(os.path.join(args.out_dir, 'EMPTY.txt'), 'w').close()
    
    # Create summary plots if requested
    if args.create_summary and any_ok:
        print("Creating summary plots...")
        create_summary_plots(modules, args.csv_dir, args.out_dir, args.block_size)


if __name__ == '__main__':
    main()
