"""Create Track C submission CSV/zip files from SFT or GRPO adapters.

Examples:
    python make_submission.py --stage sft --run qwen4b
    python make_submission.py --stage sft --run qwen4b --checkpoint checkpoint-63
    python make_submission.py --stage sft --run qwen4b --checkpoints all --n-ckpt-parallel 2
    python make_submission.py --stage grpo --run qwen4b-grpo --checkpoint best
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
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


SFT_FINAL_ALIASES = {"adapter", "best", "final", "last"}


@torch.no_grad()
def score_rows(model, tokenizer, task, df, letter_ids, gen_max_new, batch_size,
               verbose=True):
    """Generate reasoning, then score A/B/C answer logits for submission."""
    order = task.id_order(letter_ids)
    tokenizer.padding_side = "left"
    out = np.zeros((len(df), 3), dtype=np.float32)
    traces = []
    rows = list(df.itertuples(index=False))
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        prompts = [task.think_prompt(tokenizer, r.perturb_gene, r.target_gene)
                   for r in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=gen_max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        reasons = [t.split(task.think_close.strip())[0].strip()[:2500] for t in texts]

        prefixes = [task.answer_prefix(tokenizer, r.perturb_gene, r.target_gene, rs)
                    for r, rs in zip(chunk, reasons)]
        enc2 = tokenizer(prefixes, return_tensors="pt", padding=True,
                         truncation=True, add_special_tokens=False).to(model.device)
        out[i:i + len(chunk)] = model(**enc2).logits[:, -1, order].float().cpu().numpy()
        traces.extend(reasons)
        if verbose:
            print(f"  scored {min(i+batch_size, len(rows))}/{len(rows)}", end="\r")
    if verbose:
        print()
    return out, traces


def _checkpoint_step(path: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", path.name)
    if not m:
        raise ValueError(f"checkpoint path must end with checkpoint-<step>: {path}")
    return int(m.group(1))


def _complete_sft_checkpoints(run: str) -> list[str]:
    ckpt_dir = paths.sft_checkpoints(run)
    paths.require(ckpt_dir, f"missing checkpoints for sft/{run}")
    cks = [p for p in ckpt_dir.glob("checkpoint-*")
           if p.is_dir() and (p / "trainer_state.json").exists()]
    return [p.name for p in sorted(cks, key=_checkpoint_step)]


def _resolve_sft_adapter(run: str, checkpoint: str) -> Path:
    run_dir = paths.sft_dir(run)
    spec = str(checkpoint or "adapter")
    if spec in SFT_FINAL_ALIASES:
        return paths.sft_adapter(run)
    if spec.isdigit():
        return paths.sft_checkpoints(run) / f"checkpoint-{spec}"
    if spec.startswith("checkpoint-"):
        return paths.sft_checkpoints(run) / spec

    p = Path(spec)
    if p.is_absolute() or p.exists():
        return p
    for candidate in (run_dir / spec, paths.sft_checkpoints(run) / spec):
        if candidate.exists():
            return candidate
    return paths.sft_checkpoints(run) / spec




def _require_adapter_readable(adapter: Path, stage: str, run: str, checkpoint: str):
    config = paths.require(adapter / "adapter_config.json",
                           f"missing PEFT adapter_config.json for {stage}/{run}: {checkpoint}")
    candidates = [
        adapter / "adapter_model.safetensors",
        adapter / "adapter_model.bin",
        adapter / "adapter_model.safetensors.index.json",
        adapter / "adapter_model.bin.index.json",
    ]
    candidates += sorted(adapter.glob("adapter_model-*.safetensors"))
    candidates += sorted(adapter.glob("adapter_model-*.bin"))
    weights = [x for x in candidates if x.exists()]
    if not weights:
        raise FileNotFoundError(f"missing PEFT adapter weights in {adapter}")

    unreadable = [x for x in [config, *weights] if not os.access(x, os.R_OK)]
    if unreadable:
        paths_txt = "\n  ".join(str(x) for x in unreadable[:8])
        raise PermissionError(
            f"adapter files exist but are not readable for {stage}/{run} "
            f"checkpoint={checkpoint}:\n  {paths_txt}\n"
            "Ask the file owner to grant read access, e.g. chmod -R g+rX "
            f"{adapter.parent if adapter.name.startswith('checkpoint-') else adapter}"
        )

def _run_paths(stage: str, run: str, checkpoint: str):
    if stage == "sft":
        run_dir = paths.sft_dir(run)
        adapter = _resolve_sft_adapter(run, checkpoint)
    elif stage == "grpo":
        run_dir = paths.grpo_dir(run)
        adapter = run_dir / checkpoint
    else:
        raise ValueError(f"unknown stage: {stage}")
    paths.require(run_dir / "resolved_config.json",
                  f"missing run metadata for {stage}/{run}")
    paths.require(adapter, f"missing adapter/checkpoint for {stage}/{run}: {checkpoint}")
    _require_adapter_readable(adapter, stage, run, checkpoint)
    return run_dir, adapter


def _default_temperature(run_dir: Path) -> float:
    metrics = run_dir / "metrics.json"
    if not metrics.exists():
        return 1.0
    try:
        return float(json.loads(metrics.read_text()).get("best_temperature", 1.0))
    except Exception:
        return 1.0


def _checkpoint_label(stage: str, checkpoint: str) -> str:
    if stage == "sft" and str(checkpoint or "adapter") in SFT_FINAL_ALIASES:
        return "adapter"
    return Path(str(checkpoint)).name.replace("/", "-")


def _default_out(stage: str, run: str, checkpoint: str, multiple: bool) -> Path:
    if multiple:
        return paths.OUTPUT / "submissions" / f"{stage}-{run}-{_checkpoint_label(stage, checkpoint)}.csv"
    return paths.OUTPUT / "submissions" / f"{stage}-{run}.csv"


def _resolve_out(out_arg: str | None, stage: str, run: str, checkpoint: str,
                 multiple: bool) -> Path:
    if not out_arg:
        return _default_out(stage, run, checkpoint, multiple)
    out = Path(out_arg)
    if not multiple:
        return out
    label = _checkpoint_label(stage, checkpoint)
    if out.suffix:
        return out.with_name(f"{out.stem}-{label}{out.suffix}")
    return out / f"{stage}-{run}-{label}.csv"


def _expand_checkpoints(stage: str, run: str, checkpoint: str,
                        checkpoints: list[str] | None) -> list[str]:
    if checkpoints is None:
        return [checkpoint]
    if stage != "sft":
        raise SystemExit("--checkpoints is currently only supported for --stage sft")
    if len(checkpoints) == 1 and checkpoints[0] == "all":
        cks = _complete_sft_checkpoints(run)
        if not cks:
            raise SystemExit(f"no complete checkpoints found for sft/{run}")
        return cks
    return checkpoints


def create_submission(stage: str, run: str, checkpoint: str, out_arg: str | None,
                      temperature_arg: float | None, gen_max_new_arg: int | None,
                      batch_arg: int | None, multiple: bool) -> str:
    run_dir, adapter = _run_paths(stage, run, checkpoint)
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    cfg = resolved
    model_cfg = cfg["model"]
    eval_cfg = cfg.get("eval", {})
    temperature = temperature_arg if temperature_arg is not None else _default_temperature(run_dir)
    gen_max_new = gen_max_new_arg or int(eval_cfg.get("gen_max_new", 320))
    batch = batch_arg or int(eval_cfg.get("infer_batch", 8))

    task = Task.from_prompts(cfg.get("prompts", "student/default"))
    out = _resolve_out(out_arg, stage, run, checkpoint, multiple)
    print(f"[submission] stage={stage} run={run} checkpoint={checkpoint} adapter={adapter}")
    print(f"[submission] T={temperature} gen_max_new={gen_max_new} batch={batch} out={out}")

    model_cfg = SimpleNamespace(**model_cfg)
    model, tokenizer, backend = modeling.load_model_and_tokenizer(model_cfg)
    modeling.disable_incompatible_torchao_for_peft()
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
        "model_name": [model_cfg.name] * len(test),
    })

    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out, arcname="submission.csv")
    print(f"[submission] wrote {out}")
    print(f"[submission] wrote {zip_path}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(out)


def _worker(job: dict) -> str:
    with torch.no_grad():
        return create_submission(**job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sft", "grpo"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best",
                    help="Single adapter/checkpoint. SFT accepts adapter/final, N, checkpoint-N, or a path.")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="SFT checkpoint list, e.g. --checkpoints 10 20 checkpoint-30, or --checkpoints all")
    ap.add_argument("--n-ckpt-parallel", type=int, default=1,
                    help="Number of checkpoint submissions to run concurrently when --checkpoints has multiple entries.")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--out", default=None,
                    help="Output CSV path. With multiple checkpoints, a checkpoint suffix is added, or this can be an output directory.")
    ap.add_argument("--gen-max-new", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    if args.n_ckpt_parallel < 1:
        raise SystemExit("--n-ckpt-parallel must be >= 1")

    checkpoints = _expand_checkpoints(args.stage, args.run, args.checkpoint,
                                      args.checkpoints)
    multiple = len(checkpoints) > 1
    if multiple:
        print(f"[submission] checkpoints={checkpoints} n_parallel={args.n_ckpt_parallel}")

    jobs = [{
        "stage": args.stage,
        "run": args.run,
        "checkpoint": ck,
        "out_arg": args.out,
        "temperature_arg": args.temperature,
        "gen_max_new_arg": args.gen_max_new,
        "batch_arg": args.batch,
        "multiple": multiple,
    } for ck in checkpoints]

    if len(jobs) == 1 or args.n_ckpt_parallel == 1:
        for job in jobs:
            _worker(job)
        return

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.n_ckpt_parallel, mp_context=ctx) as ex:
        futs = {ex.submit(_worker, job): job["checkpoint"] for job in jobs}
        for fut in as_completed(futs):
            ck = futs[fut]
            try:
                print(f"[submission] complete checkpoint={ck} out={fut.result()}")
            except Exception as e:
                print(f"[submission] failed checkpoint={ck}: {e}")
                raise


if __name__ == "__main__":
    with torch.no_grad():
        main()
