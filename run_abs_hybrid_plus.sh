#!/bin/bash

# 测试 abs_hybrid_plus 方法
echo "Testing abs_hybrid_plus method..."

cd /Users/superone77/Code/reorderquant/PermutationGame

python main.py \
    --model-name "microsoft/DialoGPT-small" \
    --cache-dir "cache_abs_hybrid" \
    --perm-dir "perm_abs_hybrid" \
    --reorder-method "abs_hybrid_plus" \
    --hybrid-top-pct 0.10 \
    --partial-quant \
    --partial-quant-pct 0.10 \
    --eval-layers single \
    --layer-idx 3 \
    --block-size 16 \
    --max-samples 16 \
    --batch-size 1 \
    --device auto \
    --dtype float16 \
    --scale-format e4m3 \
    --max-act-rows 32768 \
    --interval-key center \
    --seed 0 \
    --viz \
    --viz-dir "viz_abs_hybrid" \
    --viz-max-rows 1024 \
    --viz-bins 200 \
    --viz-clip-pct 99.5 \
    --viz-logy \
    --block-axis token \
    --csv \
    --csv-dir "csv_abs_hybrid"

echo "Running visualizer..."
python visualizer.py \
    --csv-dir "csv_abs_hybrid" \
    --out-dir "viz_abs_hybrid" \
    --group-by-layer \
    --create-summary \
    --max-modules 50

echo "✓ abs_hybrid_plus test completed!"
