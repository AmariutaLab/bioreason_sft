#!/bin/bash
# =============================================================================
# Run one or more Track C experiments on H100s via a SLURM array.
#
#   sbatch slurm/run_experiment.sbatch                 # runs the whole EXPERIMENTS list
#   sbatch --array=0 slurm/run_experiment.sbatch       # just the first one
#   sbatch --array=1-2 slurm/run_experiment.sbatch     # a subset
#
# Each array task = one independent experiment (its own configs + run names), so
# ablations run in parallel rather than serially.
#
# CPU-only prerequisites (run ONCE on the login node — they need internet, not GPU):
#   pixi install
#   pixi run data
#   pixi run grounding
#   TEACHER_API_KEY=... pixi run traces-smoke     # then READ the traces
#   TEACHER_API_KEY=... pixi run traces
# =============================================================================
#SBATCH --job-name=bioreason
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH -t 12:00:00
#SBATCH --output=../output/slurm/%x-%A_%a.out
#SBATCH --error=../output/slurm/%x-%A_%a.err
#SBATCH --partition=nairr-gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=sdp147
#SBATCH --no-requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=i3gupta@ucsd.edu
#SBATCH --array=0-3                     # index into EXPERIMENTS below

set -euo pipefail

# ── experiment matrix ────────────────────────────────────────────────────────
# One line per experiment: "STAGE|SFT_CONFIG|SFT_RUN|GRPO_CONFIG|GRPO_RUN|EXTRA"
#   STAGE       : sft | grpo | sft+grpo
#   EXTRA       : optional "--set key=val" overrides, space separated
# Edit this list; do NOT edit the python source to change an experiment.
EXPERIMENTS=(
  # main run: Qwen3-8B (8.2B dense, most knowledge capacity under the 10B cap)
  "sft+grpo|h100_8b|qwen8b|h100|qwen8b-grpo|"
  # base-model A/B: Gemma 4 E4B (Apache 2.0, native thinking, but edge-tier)
  "sft+grpo|gemma4_e4b|gemma4b|gemma4_e4b|gemma4b-grpo|"
  # the SynthPert control: how much do the traces actually buy?
  "sft|label_only_ablation|abl-labelonly||| "
  # hyperparameter sweep without touching source
  "sft+grpo|h100_8b|qwen8b-r64|h100|qwen8b-r64-grpo|--set lora.r=64 --set lora.alpha=128"
)

TRACES_RUN="${TRACES_RUN:-default}"     # override: TRACES_RUN=v2 sbatch ...

IDX=${SLURM_ARRAY_TASK_ID:-0}
IFS='|' read -r STAGE SFT_CFG SFT_RUN GRPO_CFG GRPO_RUN EXTRA <<< "${EXPERIMENTS[$IDX]}"

echo "=================================================="
echo "array task : $IDX"
echo "stage      : $STAGE"
echo "traces     : $TRACES_RUN"
echo "sft        : config=$SFT_CFG run=$SFT_RUN"
echo "grpo       : config=$GRPO_CFG run=$GRPO_RUN"
echo "extra      : $EXTRA"
echo "node       : $(hostname)   gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=================================================="

# ── environment ──────────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"
mkdir -p ../output/slurm

# CHANGE to match your cluster
module load cuda/12.4 || true

# pixi provides the env. `pixi shell-hook` activates it into THIS shell, so the
# srun'd python below is the project interpreter — no conda, no PATH surprises,
# and the exact same pixi.lock the login node used for trace generation.
export PIXI_HOME="${PIXI_HOME:-$HOME/.pixi}"
export PATH="$PIXI_HOME/bin:$PATH"
eval "$(pixi shell-hook -e default)"

# pixi's [activation.env] already sets TOKENIZERS_PARALLELISM and HF_HOME.
# For wandb use: -e wandb, and export WANDB_API_KEY (or add it as a cluster secret).
# eval "$(pixi shell-hook -e wandb)"

pixi run check-gpu

# ── run ──────────────────────────────────────────────────────────────────────
# --resume is safe on a fresh run (no checkpoint -> starts from scratch), so
# requeued jobs pick up where they left off automatically.
if [[ "$STAGE" == *"sft"* ]]; then
  echo ">>> SFT: $SFT_RUN"
  srun python train_sft_distill.py \
      --config "$SFT_CFG" \
      --traces "$TRACES_RUN" \
      --out    "$SFT_RUN" \
      --resume \
      $EXTRA
fi

if [[ "$STAGE" == *"grpo"* ]]; then
  echo ">>> GRPO: $GRPO_RUN"
  srun python train_grpo.py \
      --config "$GRPO_CFG" \
      --sft    "$SFT_RUN" \
      --out    "$GRPO_RUN" \
      $EXTRA
fi

echo "=================================================="
echo "done. artifacts:"
echo "  ../output/sft/$SFT_RUN/"
[[ -n "${GRPO_RUN// }" ]] && echo "  ../output/grpo/$GRPO_RUN/"
echo "scores:"
grep -h blocked_val "../output/sft/$SFT_RUN/metrics.json" 2>/dev/null || true
echo "=================================================="
