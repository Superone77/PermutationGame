export CUDA_VISIBLE_DEVICES="0"

torchrun --nnodes=1 --nproc_per_node=1 main.py \
  --model-id /local/mnt/workspace/wanqi/llama/LLM-Research/Meta-Llama-3.1-8B \
  --cache-dir ./cache \
  --dataset wikitext --dataset-name wikitext-2-raw-v1 --split validation \
  --seq-len 2048 --max-samples 32 --batch-size 1 \
  --device auto --dtype float16 \
  --scale-format e4m3 --block-size 16 \
  --reorder-method hybrid_plus --hybrid-top-pct 0.10 --interval-key center \
  --partial-quant --partial-quant-pct 0.10 \
  --block-axis feature --viz

python visualizer.py \
  --csv-dir ./cache/csv_layer4 \
  --out-dir ./cache/viz_from_csv \
  --block-size 16 --topk 20 --maxpoints 4096 --style darkgrid
