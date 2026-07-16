"""Data, split, metric, checkpoint and logging utilities shared by all stages.

The split and metric live here so they CANNOT drift between trace generation,
SFT and GRPO. Drift is silent and fatal: a different split leaks the val set and
every number afterwards is fiction.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import paths
from task import LABEL2LETTER


# ── data ────────────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(paths.train_csv())
    test = pd.read_csv(paths.test_csv())
    for df in (train, test):
        df.rename(columns={"pert": "perturb_gene", "gene": "target_gene"},
                  inplace=True)
        for c in ("perturb_gene", "target_gene"):
            df[c] = df[c].astype(str).str.strip()
    norm = {"up": "up", "down": "down", "none": "none", "no-change": "none"}
    train["label"] = train["label"].map(norm)
    assert train["label"].notna().all(), "unmapped label in train.csv"
    train["letter"] = train["label"].map(LABEL2LETTER)
    return train, test


def two_axis_split(train_df, seed=42, val_pert_frac=0.20, val_gene_frac=0.25,
                   verbose=True):
    """Blocked split disjoint on BOTH axes, mimicking the official test design.

    The real competition split is disjoint on the perturbation axis (80/10/10)
    AND the gene axis (60/20/20) — no gene appears in more than one split. So:

        val   = rows where BOTH the perturbation and the gene are held out
        train = rows where NEITHER is
        off-diagonal rows (one axis seen) are DISCARDED — they would leak

    Blocking on two axes is expensive (~30% of rows discarded). That is the
    correct price; a random split flatters the score and you ship a worse model.
    """
    rng = np.random.default_rng(seed)
    perts = train_df["perturb_gene"].unique()
    genes = train_df["target_gene"].unique()
    val_perts = set(rng.choice(perts, int(len(perts) * val_pert_frac), replace=False))
    val_genes = set(rng.choice(genes, int(len(genes) * val_gene_frac), replace=False))

    p = train_df["perturb_gene"].isin(val_perts)
    g = train_df["target_gene"].isin(val_genes)
    va = train_df[p & g].copy()
    tr = train_df[~p & ~g].copy()

    assert set(tr["perturb_gene"]) & set(va["perturb_gene"]) == set()
    assert set(tr["target_gene"]) & set(va["target_gene"]) == set()
    if verbose:
        print(f"[split] train={len(tr)} val={len(va)} "
              f"discarded={len(train_df) - len(tr) - len(va)}")
        print(f"[split] val labels: {va['label'].value_counts().to_dict()}")
        if len(va) < 300:
            print("[split] WARNING val < 300 rows — raise val_*_frac")
    return tr, va


def split_from_cfg(train_df, cfg, verbose=True):
    s = cfg.split
    return two_axis_split(train_df, seed=s.seed, val_pert_frac=s.val_pert_frac,
                          val_gene_frac=s.val_gene_frac, verbose=verbose)


# ── metric ──────────────────────────────────────────────────────────────────
def softmax_T(logits, T=1.0):
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def score(labels, p_up, p_down):
    """Official metric: mean of DE-AUROC and DIR-AUROC.

    DE : (up or down) vs none, ranked by p_up + p_down
    DIR: up vs down among TRUE DE rows, ranked by p_up / (p_up + p_down)

    Both are RANK based, so constant/tied scores give exactly 0.5.
    """
    labels = np.asarray(labels)
    p_up, p_down = np.asarray(p_up), np.asarray(p_down)
    de = roc_auc_score((labels != "none").astype(int), p_up + p_down)
    m = labels != "none"
    den = p_up[m] + p_down[m]
    dir_ = roc_auc_score((labels[m] == "up").astype(int),
                         np.where(den > 0, p_up[m] / den, 0.5))
    return {"de": de, "dir": dir_, "score": (de + dir_) / 2}


def tune_temperature(logits, labels, grid=(0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0)):
    """Sharpen/soften logits to maximise the rank metric on val.

    Saturated logits (1.0 vs 1e-12) tie rows together and cost AUROC; T>1
    spreads them back out. The chosen T is applied to test.
    """
    best_T, best = 1.0, -1.0
    for T in grid:
        p = softmax_T(logits, T)
        s = score(labels, p[:, 0], p[:, 1])
        print(f"  T={T:<4} DE={s['de']:.4f} DIR={s['dir']:.4f} SCORE={s['score']:.4f}")
        if s["score"] > best:
            best, best_T = s["score"], T
    return best_T, best


# ── checkpoints / resume ────────────────────────────────────────────────────
# HF Trainer checkpoints carry optimizer state, LR-scheduler position, RNG state
# and global_step/epoch (trainer_state.json). wandb is NOT the resume mechanism.

def find_last_checkpoint(output_dir):
    p = Path(output_dir)
    if not p.exists():
        return None
    cks = [c for c in p.glob("checkpoint-*")
           if c.is_dir() and (c / "trainer_state.json").exists()]
    if not cks:
        return None
    return max(cks, key=lambda c: int(re.findall(r"\d+", c.name)[-1]))


def checkpoint_progress(ckpt):
    if ckpt is None:
        return {}
    st = Path(ckpt) / "trainer_state.json"
    if not st.exists():
        return {}
    j = json.loads(st.read_text())
    return {"global_step": j.get("global_step"), "epoch": j.get("epoch"),
            "max_steps": j.get("max_steps")}


def describe_resume(ckpt):
    pr = checkpoint_progress(ckpt)
    if not pr:
        print("[resume] no checkpoint found — starting from scratch")
        return {}
    print(f"[resume] {Path(ckpt).name}: stopped at epoch {pr.get('epoch', 0):.2f}, "
          f"step {pr.get('global_step')}/{pr.get('max_steps')}")
    print("[resume] Trainer restores optimizer, LR schedule, RNG and step count.")
    return pr


def init_wandb(cfg, run_name, extra_config=None, resume_id=None):
    """Optional logging. Degrades to a no-op without a key or the package."""
    w = cfg.get("wandb", {})
    if not w.get("enabled"):
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed — skipping")
        return None
    import os
    if not (os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").exists()):
        print("[wandb] no WANDB_API_KEY — skipping")
        return None
    run = wandb.init(project=w.get("project", "bioreason"),
                     entity=w.get("entity"), name=run_name,
                     config={**dict(cfg), **(extra_config or {})},
                     id=resume_id, resume="allow" if resume_id else None,
                     tags=w.get("tags"))
    print(f"[wandb] run id={run.id} — resume logging with wandb.id={run.id}")
    return run.id
