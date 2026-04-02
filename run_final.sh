#!/bin/bash
# Final ideology cosine experiment — agree template, 200 per ideology, 4 models
# Run this from the political_biases_meetings_gianluca directory on your GPU machine.
#
# Output: outputs/ideology_cosine_final/
#   - ideology_cosine_summary.csv          (all models, all methods, max_separation_layer / min_lr_cosine)
#   - <model>/cosine_results_long.csv      (per-layer L-L, R-R, L-R per method)
#   - comparison/<template>_cosine_comparison.png   (per-model plots)

set -e
cd "$(dirname "$0")"

python3 run_ideology_cosine_pilot.py \
    --input-csv  data/ideoinst_clean/ideoinst_sampled_rows.csv \
    --output-dir outputs/ideology_cosine_final \
    --templates  agree \
    --per-ideology 200 \
    --models \
        Qwen/Qwen2.5-7B \
        Qwen/Qwen2.5-7B-Instruct \
        mistralai/Mistral-7B-v0.1 \
        mistralai/Mistral-7B-Instruct-v0.2 \
    --random-rounds 500 \
    --seed 42
