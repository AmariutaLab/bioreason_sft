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
# CPU-only prerequisites (run ONCE on the login node, they need internet, not GPU):
#   python download_data.py
#   python build_grounding.py --out default
#   TEACHER_API_KEY=... python build_traces.py --out default
# =============================================================================
#SBATCH --job-name=bioreason
#SBATCH --partition=gpu                 # <-- CHANGE to your H100 partition
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=../output/slurm/%x-%A_%a.out
#SBATCH --error=../output/slurm/%x-%A_%a.err
#SBATCH --array=0-2                     # index into EXPERIMENTS below

set -euo pipefail

# ── experiment matrix ────────────────────────────────────────────────────────
# One line per experiment: "STAGE|SFT_CONFIG|SFT_RUN|GRPO_CONFIG|GRPO_RUN|EXTRA"
#   STAGE       : sft | grpo | sft+grpo
#   EXTRA       : optional "--set key=val" overrides, space separated
# Edit this list; do NOT edit the python source to change an experiment.
EXPERIMENTS=(
  "sft+grpo|h100_8b|qwen8b|h100|qwen8b-grpo|"
  "sft|label_only_ablation|abl-labelonly||| "
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

# CHANGE these to match your cluster
module load cuda/12.4 || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bioreason

export MLGENX_ROOT="$(cd .. && pwd)"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$MLGENX_ROOT/.hf_cache}"     # keep HF cache off $HOME
# export WANDB_API_KEY=...   # or add --set wandb.enabled=true and log in first

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

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
