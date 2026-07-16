"""Create a Track C submission CSV and zip from an SFT or GRPO adapter.

Examples:
    python make_submission.py --stage sft --run qwen4b
    python make_submission.py --stage grpo --run qwen4b-grpo --checkpoint best
    python make_submission.py --stage sft --run qwen4b --temperature 1.6
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from peft import PeftModel

import common
import modeling
import paths
from task import Task
from train_sft_distill import score_rows


def _run_paths(stage: str, run: str, checkpoint: str):
    if stage == "sft":
        run_dir = paths.sft_dir(run)
        adapter = paths.sft_adapter(run)
    elif stage == "grpo":
        run_dir = paths.grpo_dir(run)
        adapter = run_dir / checkpoint
    else:
        raise ValueError(f"unknown stage: {stage}")
    paths.require(run_dir / "resolved_config.json",
                  f"missing run metadata for {stage}/{run}")
    paths.require(adapter, f"missing adapter for {stage}/{run}")
    return run_dir, adapter


def _default_temperature(run_dir: Path) -> float:
    metrics = run_dir / "metrics.json"
    if not metrics.exists():
        return 1.0
    try:
        return float(json.loads(metrics.read_text()).get("best_temperature", 1.0))
    except Exception:
        return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sft", "grpo"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best",
                    help="GRPO adapter directory under output/grpo/<run>/")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--out", default=None,
                    help="output CSV path; default output/submissions/<stage>-<run>.csv")
    ap.add_argument("--gen-max-new", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    run_dir, adapter = _run_paths(args.stage, args.run, args.checkpoint)
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    cfg = resolved
    model_cfg = cfg["model"]
    eval_cfg = cfg.get("eval", {})
    temperature = args.temperature if args.temperature is not None else _default_temperature(run_dir)
    gen_max_new = args.gen_max_new or int(eval_cfg.get("gen_max_new", 320))
    batch = args.batch or int(eval_cfg.get("infer_batch", 8))

    task = Task.from_prompts(cfg.get("prompts", "student/default"))
    print(f"[submission] stage={args.stage} run={args.run} adapter={adapter}")
    print(f"[submission] T={temperature} gen_max_new={gen_max_new} batch={batch}")

    model_cfg = SimpleNamespace(**model_cfg)
    model, tokenizer, backend = modeling.load_model_and_tokenizer(model_cfg)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    letter_ids = task.resolve_letter_ids(tokenizer)
    modeling.prepare_for_inference(model, backend)
    model.eval()

    _, test = common.load_data()
    logits, traces = score_rows(model, tokenizer, task, test, letter_ids,
                                gen_max_new, batch)
    probs = common.softmax_T(logits, temperature)
    pred_up = probs[:, 0]
    pred_down = probs[:, 1]

    if np.any(pred_up < 0) or np.any(pred_down < 0):
        raise RuntimeError("negative probabilities generated")
    if np.any(pred_up + pred_down > 1.000001):
        raise RuntimeError("prediction_up + prediction_down exceeds 1")

    token_counts = [
        len(tokenizer(t, add_special_tokens=False)["input_ids"])
        for t in traces
    ]
    sub = pd.DataFrame({
        "id": test["id"],
        "prediction_up": pred_up,
        "prediction_down": pred_down,
        "reasoning_trace": traces,
        "tokens_used": token_counts,
        "model_name": [model_cfg["name"]] * len(test),
    })

    out = Path(args.out) if args.out else paths.OUTPUT / "submissions" / f"{args.stage}-{args.run}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out, arcname="submission.csv")
    print(f"[submission] wrote {out}")
    print(f"[submission] wrote {zip_path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
