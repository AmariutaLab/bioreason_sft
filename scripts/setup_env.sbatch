#!/bin/bash
# Create or update the persistent GPU-capable Conda environment for Track C.
#
# Submit example:
#   sbatch -A csd832 -p compute -t 04:00:00 -c 4 --mem=32G scripts/setup_env.sbatch
#
# The package cache is kept on Slurm scratch to avoid the corrupted/slow default
# cache under $HOME. The env itself is persistent under ../envs.
# Override with:
#   BIOREASON_ENV_PREFIX=/path/to/env sbatch ...

#SBATCH --job-name=br-env
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=../output/slurm/%x-%j.out
#SBATCH --error=../output/slurm/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/expanse/lustre/projects/ddp412/zxu6/mlgenx/bioreason_sft}"
MLGENX_ROOT="$(cd "$REPO_DIR/.." && pwd)"
ENV_PREFIX="${BIOREASON_ENV_PREFIX:-$MLGENX_ROOT/envs/bioreason-gpu}"
SCRATCH_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/scratch/$USER/job_${SLURM_JOB_ID:-manual}}}"
PKGS_DIR="$SCRATCH_ROOT/conda_pkgs"

mkdir -p "$PKGS_DIR" "$(dirname "$ENV_PREFIX")"
cd "$REPO_DIR"

echo "=================================================="
echo "job        : ${SLURM_JOB_ID:-manual}"
echo "node       : $(hostname)"
echo "repo       : $REPO_DIR"
echo "env prefix : $ENV_PREFIX"
echo "pkg cache  : $PKGS_DIR"
echo "started    : $(date)"
echo "=================================================="

export CONDA_PKGS_DIRS="$PKGS_DIR"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/zxu6/miniforge3}"

if /home/zxu6/miniforge3/bin/mamba --version >/dev/null 2>&1; then
  SOLVER=(/home/zxu6/miniforge3/bin/mamba)
  echo "solver     : /home/zxu6/miniforge3/bin/mamba"
else
  source /home/zxu6/miniconda3/etc/profile.d/conda.sh
  SOLVER=(conda)
  echo "solver     : conda"
fi

if [[ -x "$ENV_PREFIX/bin/python" ]]; then
  echo "Updating existing env"
  "${SOLVER[@]}" env update -p "$ENV_PREFIX" -f environment.yml --prune
else
  echo "Creating new env"
  "${SOLVER[@]}" env create -p "$ENV_PREFIX" -f environment.yml
fi

"$ENV_PREFIX/bin/python" - <<'PY'
import importlib

mods = ["yaml", "pandas", "numpy", "sklearn", "torch", "transformers", "peft", "trl"]
for name in mods:
    importlib.import_module(name)

import torch

print("imports: ok")
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available on this node:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "finished   : $(date)"
