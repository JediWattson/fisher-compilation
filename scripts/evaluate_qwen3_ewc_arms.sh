#!/usr/bin/env bash
# Evaluate each EWC-refined artifact on the frozen task panel and the
# 512-token text panel, reusing the expanded-eval reference data.
# Usage: scripts/evaluate_qwen3_ewc_arms.sh <run-dir> [eval-dir]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR="${1:-.local-runs/qwen3-ewc-refine}"
EVAL_DIR="${2:-$RUN_DIR/eval}"
mkdir -p "$EVAL_DIR"

TASKS_REF=.local-runs/qwen3-mlx-expanded-eval/exact-tasks.json
TEXT_CORPUS=.local-runs/qwen3-mlx-expanded-eval/corpus/assessment_512.jsonl
TEXT_REF=.local-runs/qwen3-mlx-expanded-eval/reference-512
CALIB=.local-runs/qwen3-mlx-first-fit/corpus/fit.jsonl

for artifact in "$RUN_DIR"/*/; do
  name=$(basename "$artifact")
  [ -f "$artifact/weights.safetensors" ] || continue
  task_out="$EVAL_DIR/$name-tasks.json"
  text_out="$EVAL_DIR/$name-512.json"
  if [ ! -f "$task_out" ]; then
    echo "=== tasks: $name"
    MLX_ENABLE_TF32=0 .venv/bin/python scripts/evaluate_qwen3_mlx_tasks.py evaluate \
      --artifact "$artifact" --reference "$TASKS_REF" --output "$task_out"
  fi
  if [ ! -f "$text_out" ]; then
    echo "=== text512: $name"
    MLX_ENABLE_TF32=0 .venv/bin/python scripts/evaluate_qwen3_mlx_fit.py \
      --phase evaluate --artifact "$artifact" --corpus "$TEXT_CORPUS" \
      --reference-dir "$TEXT_REF" --reference-mode compact --head-chunk-size 128 \
      --max-examples 64 --max-length 512 --max-tokens 32768 \
      --calibration-corpus "$CALIB" --report "$text_out"
  fi
done
echo "=== done; reports in $EVAL_DIR"
