#!/bin/bash
# Validate that the persistent bioreason env can see a GPU.
#
# Submit after setup succeeds:
#   sbatch -A csd832 -p gpu-debug --gpus=1 -N 1 -n 1 -c 4 --mem=32G \
#     --dependency=afterok:<setup_job_id> scripts/gpu_smoke.sbatch

#SBATCH --job-name=br-gpucheck
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=../output/slurm/%x-%j.out
#SBATCH --error=../output/slurm/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/expanse/lustre/projects/ddp412/zxu6/mlgenx/bioreason_sft}"
MLGENX_ROOT="$(cd "$REPO_DIR/.." && pwd)"
ENV_PREFIX="${BIOREASON_ENV_PREFIX:-$MLGENX_ROOT/envs/bioreason-gpu}"

cd "$REPO_DIR"

module load gpu/0.17.3b || true
module load cuda/12.6.3 || module load cuda/11.8.0 || true
source /home/zxu6/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

export MLGENX_ROOT
export HF_HOME="${HF_HOME:-$MLGENX_ROOT/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

echo "=================================================="
echo "job        : ${SLURM_JOB_ID:-manual}"
echo "node       : $(hostname)"
echo "env prefix : $ENV_PREFIX"
echo "started    : $(date)"
echo "=================================================="

nvidia-smi

python - <<'PY'
import importlib

mods = ["yaml", "pandas", "numpy", "sklearn", "torch", "transformers", "peft", "trl"]
for name in mods:
    importlib.import_module(name)

import torch

print("imports: ok")
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible from torch")
print("gpu:", torch.cuda.get_device_name(0))
PY

python -m py_compile \
  common.py config.py paths.py task.py modeling.py \
  build_grounding.py build_traces.py train_sft_distill.py train_grpo.py \
  make_submission.py validate_submission.py

echo "finished   : $(date)"
