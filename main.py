from __future__ import annotations
import os
import json
import argparse
import torch
from model_utils import collect_block_stats_and_samples, pick_device, str2dtype
from reorder_algorithms import compute_reorders
from evaluation import evaluate_nvfp4_mse
from utils import visualize_modules


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
    ap.add_argument('--reorder-method', type=str, default='hybrid', choices=['hybrid','hybrid_plus','abs_hybrid_plus','kmeans','interval','peg'])
    ap.add_argument('--interval-key', type=str, default='center', choices=['center','lexi'])
    ap.add_argument('--hybrid-top-pct', type=float, default=0.10)
    ap.add_argument('--partial-quant', action='store_true', default=False)
    ap.add_argument('--partial-quant-pct', type=float, default=0.10, help='Percentage of channels to keep unquantized in partial quantization mode')
    ap.add_argument('--eval-layers', type=str, default='all', choices=['all', 'single'], help='Evaluate all layers or single layer')
    ap.add_argument('--layer-idx', type=int, default=3, help='Layer index to evaluate when eval-layers=single')
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
    
    layer_suffix = f"layer{args.layer_idx}" if args.eval_layers == 'single' else "all_layers"
    stats_path = os.path.join(args.cache_dir, f'llama31_8b_{layer_suffix}_all_acts.pt')
    perm_path = os.path.join(args.cache_dir, f'rptq_{layer_suffix}_block{args.block_size}_perms.pt')
    device = pick_device(args.device)
    dtype = str2dtype(args.dtype)
    
    oc_stats, ic_stats, acts = collect_block_stats_and_samples(
        model_id=args.model_id, 
        cache_path=stats_path, 
        device=device, 
        dtype=dtype, 
        seq_len=args.seq_len, 
        max_samples=args.max_samples, 
        batch_size=args.batch_size, 
        dataset=args.dataset, 
        dataset_name=args.dataset_name, 
        split=args.split, 
        max_act_rows=args.max_act_rows,
        eval_layers=args.eval_layers,
        layer_idx=args.layer_idx
    )
    
    perms = compute_reorders(
        oc_stats, 
        block_size=args.block_size, 
        method=args.reorder_method, 
        interval_key=args.interval_key, 
        acts_for_mse=acts, 
        sls_iters=0, 
        sls_block_axis='feature', 
        sls_scale_format=args.scale_format, 
        hybrid_top_pct=args.hybrid_top_pct
    )
    torch.save(perms, perm_path)
    
    csv_dir = args.csv_dir or os.path.join(args.cache_dir, f'csv_{layer_suffix}')
    results = evaluate_nvfp4_mse(
        acts, 
        perms, 
        block_size=args.block_size, 
        scale_format=args.scale_format, 
        block_axis=args.block_axis, 
        csv_dir=csv_dir, 
        save_csv=args.csv,
        partial_quant=args.partial_quant,
        top_pct=args.partial_quant_pct
    )
    
    if args.viz:
        viz_dir = args.viz_dir or os.path.join(args.cache_dir, f'viz_{layer_suffix}')
        visualize_modules(
            acts, 
            perms, 
            viz_dir, 
            block_size=args.block_size, 
            max_rows=args.viz_max_rows, 
            bins=args.viz_bins, 
            clip_pct=args.viz_clip_pct, 
            logy=args.viz_logy, 
            block_axis=args.block_axis
        )
    
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