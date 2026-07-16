"""Filesystem layout.

    mlgenx/
      bioreason_sft/          <- code + configs + prompts (this package)
      data/                   <- Kaggle competition CSVs
      output/
        grounding/<run>/grounding.json
        traces/<run>/traces.jsonl
        sft/<run>/            (adapter + checkpoints + metrics)
        grpo/<run>/

Every script refers to artifacts by RUN NAME, never by path:
    --traces my-traces   ->  output/traces/my-traces/traces.jsonl

Override the root with MLGENX_ROOT if you run from somewhere else.
"""
from __future__ import annotations

import os
from pathlib import Path

PKG  = Path(__file__).resolve().parent          # .../mlgenx/bioreason_sft
ROOT = Path(os.environ.get("MLGENX_ROOT", PKG.parent))   # .../mlgenx

DATA    = ROOT / "data"
OUTPUT  = ROOT / "output"
CONFIGS = PKG / "configs"
PROMPTS = PKG / "prompts"

COMPETITION = "ml-gen-x-bioreasoning-challenge-track-c"


# ── run directories ─────────────────────────────────────────────────────────
def run_dir(stage: str, name: str, create: bool = False) -> Path:
    """output/<stage>/<name>/ — the home of one run's artifacts."""
    d = OUTPUT / stage / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def grounding_json(name: str) -> Path:
    return run_dir("grounding", name) / "grounding.json"


def traces_jsonl(name: str) -> Path:
    return run_dir("traces", name) / "traces.jsonl"


def sft_dir(name: str, create: bool = False) -> Path:
    return run_dir("sft", name, create)


def sft_adapter(name: str) -> Path:
    return sft_dir(name) / "adapter"


def sft_checkpoints(name: str) -> Path:
    return sft_dir(name) / "checkpoints"


def grpo_dir(name: str, create: bool = False) -> Path:
    return run_dir("grpo", name, create)


def require(p: Path, hint: str = "") -> Path:
    if not p.exists():
        raise FileNotFoundError(f"missing: {p}\n{hint}")
    return p


# ── competition data ────────────────────────────────────────────────────────
def train_csv() -> Path:
    return require(DATA / "train.csv", "run: python download_data.py")


def test_csv() -> Path:
    return require(DATA / "test.csv", "run: python download_data.py")


def describe():
    print(f"ROOT   {ROOT}")
    print(f"DATA   {DATA}      exists={DATA.exists()}")
    print(f"OUTPUT {OUTPUT}    exists={OUTPUT.exists()}")
    for stage in ("grounding", "traces", "sft", "grpo"):
        d = OUTPUT / stage
        runs = sorted(x.name for x in d.iterdir() if x.is_dir()) if d.exists() else []
        print(f"  {stage:<10} runs: {runs or '(none)'}")


if __name__ == "__main__":
    describe()
