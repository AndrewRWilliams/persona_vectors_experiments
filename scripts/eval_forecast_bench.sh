#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-0}
model=${2:-"Qwen/Qwen2.5-7B-Instruct"}
dataset_path=${3:?"Usage: $0 [gpu] [model] <forecast_bench_json_or_jsonl_or_dir> [output_path]"}
output_path=${4:-"results/forecast_bench_results.csv"}

CUDA_VISIBLE_DEVICES="$gpu" python -m eval.eval_persona \
    --model "$model" \
    --trait forecast_bench \
    --benchmark forecast_bench \
    --dataset_path "$dataset_path" \
    --output_path "$output_path" \
    --n_per_question 1 \
    --max_tokens 200 \
    --overwrite True
