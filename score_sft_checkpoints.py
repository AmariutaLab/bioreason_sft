"""Score blocked validation for saved SFT checkpoints.

Examples:
    python score_sft_checkpoints.py --run qwen8b-strict-medium-h100-qlora
    python score_sft_checkpoints.py --run qwen8b --checkpoints 10 20 checkpoint-30
    python score_sft_checkpoints.py --run qwen8b -i my_val.csv --batch 16
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace


def checkpoint_step(path: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", path.name)
    if not m:
        raise ValueError(f"checkpoint path must end with checkpoint-<step>: {path}")
    return int(m.group(1))


def complete_checkpoints(run: str) -> list[Path]:
    import paths

    ckpt_dir = paths.sft_checkpoints(run)
    paths.require(ckpt_dir, f"missing checkpoints for sft/{run}")
    cks = [p for p in ckpt_dir.glob("checkpoint-*")
           if p.is_dir() and (p / "trainer_state.json").exists()]
    return sorted(cks, key=checkpoint_step)


def resolve_checkpoint(run: str, spec: str) -> Path:
    import paths

    ckpt_dir = paths.sft_checkpoints(run)
    if spec.isdigit():
        p = ckpt_dir / f"checkpoint-{spec}"
    elif spec.startswith("checkpoint-"):
        p = ckpt_dir / spec
    else:
        p = Path(spec)
        if not p.is_absolute() and not p.exists():
            p = ckpt_dir / spec
    paths.require(p, f"missing checkpoint {spec} for sft/{run}")
    paths.require(p / "trainer_state.json", f"checkpoint is incomplete: {p}")
    return p


def read_checkpoint_state(path: Path) -> dict:
    try:
        return json.loads((path / "trainer_state.json").read_text())
    except Exception:
        return {}


def normalize_input_rows(df):
    df = df.copy()
    df.rename(columns={"pert": "perturb_gene", "gene": "target_gene"}, inplace=True)
    required = {"perturb_gene", "target_gene"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"input CSV missing columns: {missing}")
    if "label" in df.columns:
        norm = {"up": "up", "down": "down", "none": "none", "no-change": "none"}
        df["label"] = df["label"].map(norm)
        if df["label"].isna().any():
            raise SystemExit("input CSV has unmapped labels")
    return df


def validation_rows(args, cfg):
    import pandas as pd
    import common

    if args.input_csv:
        va = normalize_input_rows(pd.read_csv(args.input_csv))
        source = str(args.input_csv)
    else:
        train, _ = common.load_data()
        seed = cfg.split.seed if args.split_seed is None else args.split_seed
        val_pert_frac = cfg.split.val_pert_frac if args.val_pert_frac is None else args.val_pert_frac
        val_gene_frac = cfg.split.val_gene_frac if args.val_gene_frac is None else args.val_gene_frac
        _, va = common.two_axis_split(train, seed=seed, val_pert_frac=val_pert_frac,
                                      val_gene_frac=val_gene_frac, verbose=True)
        source = f"blocked(seed={seed},val_pert_frac={val_pert_frac},val_gene_frac={val_gene_frac})"

    max_rows = args.max_rows
    if max_rows is None:
        max_rows = int(cfg.eval.get("max_rows", 0) or 0)
    if max_rows and len(va) > max_rows:
        va = va.sample(max_rows, random_state=cfg.split.seed)
        source += f",sample={max_rows}"
    msg = f"[val] {source} rows={len(va)}"
    if "label" in va.columns:
        msg += f" labels={va['label'].value_counts().to_dict()}"
    else:
        msg += " labels=absent"
    print(msg)
    return va


def tune_temperature_quiet(logits, labels, grid):
    import common

    best_T, best_metrics = None, None
    for T in grid:
        probs = common.softmax_T(logits, T)
        metrics = common.score(labels, probs[:, 0], probs[:, 1])
        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            best_T, best_metrics = T, metrics
    return best_T, best_metrics


def write_trace_sample(outdir: Path, ckpt: Path, n: int, df, probs, traces, tokenizer,
                       model_name: str):
    import pandas as pd

    if n <= 0:
        return
    sample = df.head(n).copy()
    trace_sample = traces[:len(sample)]
    token_counts = [
        len(tokenizer(t, add_special_tokens=False)["input_ids"])
        for t in trace_sample
    ]
    ids = sample["id"].values if "id" in sample.columns else sample.index.astype(str).values
    out = pd.DataFrame({
        "id": ids,
        "prediction_up": probs[:len(sample), 0],
        "prediction_down": probs[:len(sample), 1],
        "reasoning_trace": trace_sample,
        "tokens_used": token_counts,
        "model_name": [model_name] * len(sample),
    })
    if "label" in sample.columns:
        out["real_label"] = sample["label"].values
    ckpt_out = outdir / ckpt.name
    ckpt_out.mkdir(parents=True, exist_ok=True)
    path = ckpt_out / f"out_top{n}.csv"
    out.to_csv(path, index=False)
    print(f"  wrote traces -> {path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="SFT run name under output/sft/<run>/")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="checkpoint specs: steps, checkpoint names, or paths. Default: all complete checkpoints.")
    ap.add_argument("-i", "--input-csv", type=Path, default=None,
                    help="Optional validation CSV with pert/gene or perturb_gene/target_gene, plus optional label.")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="Override blocked validation split seed when --input-csv is not used.")
    ap.add_argument("--val-pert-frac", type=float, default=None,
                    help="Override blocked validation perturbation fraction when --input-csv is not used.")
    ap.add_argument("--val-gene-frac", type=float, default=None,
                    help="Override blocked validation target-gene fraction when --input-csv is not used.")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Validation row cap. Default: eval.max_rows from run config, or all rows.")
    ap.add_argument("--gen-max-new", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--temperature-grid", default=None,
                    help="Comma-separated temperatures. Default: eval.temperature_grid from run config.")
    ap.add_argument("--save-n-traces", type=int, default=10,
                    help="Save the first N validation outputs per checkpoint. Use 0 to disable.")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Output directory. Default: output/sft/<run>/checkpoint_val_outputs")
    ap.add_argument("--out", type=Path, default=None,
                    help="Score TSV path. Default: <outdir>/checkpoint_val_scores.tsv")
    return ap.parse_args()


def main():
    args = parse_args()

    import pandas as pd
    import torch
    from peft import PeftModel

    import common
    import config as cfgmod
    import modeling
    import paths
    from task import Task
    from train_sft_distill import score_rows

    run_dir = paths.sft_dir(args.run)
    paths.require(run_dir / "resolved_config.json", f"missing SFT run metadata for {args.run}")
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    cfg = cfgmod.Cfg(resolved)

    checkpoints = ([resolve_checkpoint(args.run, x) for x in args.checkpoints]
                   if args.checkpoints else complete_checkpoints(args.run))
    if not checkpoints:
        raise SystemExit(f"no complete checkpoints found for sft/{args.run}")
    print("[checkpoints] " + ", ".join(p.name for p in checkpoints))

    va = validation_rows(args, cfg)
    gen_max_new = args.gen_max_new or int(cfg.eval.get("gen_max_new", 320))
    batch = args.batch or int(cfg.eval.get("infer_batch", 8))
    if args.temperature_grid:
        temperature_grid = tuple(float(x) for x in args.temperature_grid.split(",") if x.strip())
    else:
        temperature_grid = tuple(cfg.eval.get("temperature_grid", [1.0]))

    task = Task.from_prompts(cfg.get("prompts", "student/default"))
    model_cfg = SimpleNamespace(**cfg.model)
    if args.save_n_traces < 0:
        raise SystemExit("--save-n-traces must be >= 0")
    outdir = args.outdir or (run_dir / "checkpoint_val_outputs")
    out_path = args.out or (outdir / "checkpoint_val_scores.tsv")
    rows = []

    with torch.no_grad():
        for ckpt in checkpoints:
            step = checkpoint_step(ckpt)
            state = read_checkpoint_state(ckpt)
            print(f"\n[score] {ckpt.name} step={step} epoch={state.get('epoch')}")
            model, tokenizer, backend = modeling.load_model_and_tokenizer(model_cfg)
            model = PeftModel.from_pretrained(model, str(ckpt), is_trainable=False)
            letter_ids = task.resolve_letter_ids(tokenizer)
            modeling.prepare_for_inference(model, backend)
            model.eval()

            logits, traces = score_rows(model, tokenizer, task, va, letter_ids, gen_max_new, batch)
            if "label" in va.columns:
                best_T, metrics = tune_temperature_quiet(logits, va["label"].values, temperature_grid)
            else:
                best_T = temperature_grid[0] if temperature_grid else 1.0
                metrics = {"de": None, "dir": None, "score": None}
                print("  labels absent; skipping blocked-validation metrics")
            probs = common.softmax_T(logits, best_T)
            write_trace_sample(outdir, ckpt, args.save_n_traces, va, probs, traces,
                               tokenizer, model_cfg.name)
            rec = {
                "checkpoint": ckpt.name,
                "step": step,
                "epoch": state.get("epoch"),
                "n_val": len(va),
                "best_temperature": best_T,
                "de": metrics["de"],
                "dir": metrics["dir"],
                "blocked_val_score": metrics["score"],
            }
            rows.append(rec)
            if metrics["score"] is None:
                print(f"  T={best_T} SCORE=n/a")
            else:
                print(f"  best T={best_T} DE={rec['de']:.4f} DIR={rec['dir']:.4f} SCORE={rec['blocked_val_score']:.4f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("step")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n[wrote] {out_path}")
    if df["blocked_val_score"].notna().any():
        best = df.loc[df["blocked_val_score"].idxmax()]
        print(f"[best] {best['checkpoint']} SCORE={best['blocked_val_score']:.4f} T={best['best_temperature']}")


if __name__ == "__main__":
    main()
