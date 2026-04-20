#!/usr/bin/env bash
# =============================================================================
# run_multilingual_colab.sh
# Cross-lingual steering transfer experiment — Colab launcher
#
# Prerequisite: run inside /content/Playground/political-bias-representation-analysis
# with sibling neural_controllers_official/ checked out.
#
# Recommended GPU: Colab A100 (40 GB). Runs in ~45-75 min.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LANGUAGES="${LANGUAGES:-en it}"
REPEATS="${REPEATS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/multilingual_compass_7b}"

# Window steering (cosine-guided, window=9 is best from screening)
WINDOW="${WINDOW:-9}"
LEFT_WINDOW_COEF="${LEFT_WINDOW_COEF:-1.6}"   # centered/window_9: best_delta=1.57
RIGHT_WINDOW_COEF="${RIGHT_WINDOW_COEF:-1.6}" # centered/window_9: best_delta=0.68

# Full-range steering (all layers from -8 downward, 20 layers total)
FULL_RANGE_START="${FULL_RANGE_START:--8}"
LEFT_FULL_COEF="${LEFT_FULL_COEF:-1.5}"   # full-range baseline: best_delta=1.68
RIGHT_FULL_COEF="${RIGHT_FULL_COEF:-4.0}" # full-range baseline: best_delta=5.90

# Set any coef to "" to trigger auto-selection via sweep (adds ~5 min per direction)

echo "================================================================"
echo "  Cross-Lingual Steering Transfer Experiment"
echo "  Model:            $MODEL"
echo "  Languages:        $LANGUAGES"
echo "  Window size:      $WINDOW  (coefs: L=${LEFT_WINDOW_COEF:-AUTO} R=${RIGHT_WINDOW_COEF:-AUTO})"
echo "  Full-range start: $FULL_RANGE_START  (coefs: L=${LEFT_FULL_COEF:-AUTO} R=${RIGHT_FULL_COEF:-AUTO})"
echo "  Output:           $OUTPUT_DIR"
echo "================================================================"

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(
    python run_multilingual_compass_eval.py
    --model "$MODEL"
    --device auto
    --window-size "$WINDOW"
    --full-range-start "$FULL_RANGE_START"
    --languages $LANGUAGES
    --repeats "$REPEATS"
    --output-dir "$OUTPUT_DIR"
)

if [ -n "${LEFT_WINDOW_COEF:-}" ];  then CMD+=(--left-window-coef  "$LEFT_WINDOW_COEF");  fi
if [ -n "${RIGHT_WINDOW_COEF:-}" ]; then CMD+=(--right-window-coef "$RIGHT_WINDOW_COEF"); fi
if [ -n "${LEFT_FULL_COEF:-}" ];    then CMD+=(--left-full-coef    "$LEFT_FULL_COEF");    fi
if [ -n "${RIGHT_FULL_COEF:-}" ];   then CMD+=(--right-full-coef   "$RIGHT_FULL_COEF");   fi

echo ""
echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"

# ── Analysis ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Running analysis..."
echo "================================================================"
python analyze_multilingual_compass.py "$OUTPUT_DIR"

echo ""
echo "================================================================"
echo "  Done! Key output files:"
echo "  $OUTPUT_DIR/master_summary.json"
echo "  $OUTPUT_DIR/cross_language_comparison.csv"
echo "  $OUTPUT_DIR/en/compass_answer_sheet.csv"
echo "  $OUTPUT_DIR/it/compass_answer_sheet.csv"
echo "================================================================"
