#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

TRACE_RUN="norules-o4mini-contrastive-none-selected-full"
QWEN4B_RUN="qwen4b-norules-o4mini-contrastive-none-selected-full-h100-lora"
QWEN8B_RUN="qwen8b-norules-o4mini-contrastive-none-selected-full-h100-lora"
# TRACE_RUN="norules-ungrounded-o4mini"
# QWEN4B_RUN="qwen4b-norules-ungrounded-o4mini-h100-lora"
# QWEN8B_RUN="qwen8b-norules-ungrounded-o4mini-h100-lora"
SUBMIT_DIR="../output/submissions"

mkdir -p "$SUBMIT_DIR" "$HOME/nairr"

# echo "[$(date)] 1/4 train Qwen 4B LoRA"
# pixi run -e h100 python train_sft_distill.py \
#   --config default \
#   --traces "$TRACE_RUN" \
#   --out "$QWEN4B_RUN"

# echo "[$(date)] 2/4 train Qwen 8B LoRA"
# pixi run -e h100 python train_sft_distill.py \
#   --config h100_8b \
#   --traces "$TRACE_RUN" \
#   --out "$QWEN8B_RUN"

echo "[$(date)] 3/4 make Qwen 4B submission"
pixi run -e h100 python make_submission.py \
  --stage sft \
  --run "$QWEN4B_RUN" \
  --batch 16 \
  --checkpoints 10 20 40 --n-ckpt-parallel 3 \
  --out "$SUBMIT_DIR/$QWEN4B_RUN.csv"

echo "[$(date)] 4/4 make Qwen 8B submission"
pixi run -e h100 python make_submission.py \
  --stage sft \
  --run "$QWEN8B_RUN" \
  --batch 16 \
  --checkpoints 10 20 40 --n-ckpt-parallel 3 \
  --out "$SUBMIT_DIR/$QWEN8B_RUN.csv"

echo "[$(date)] done"
