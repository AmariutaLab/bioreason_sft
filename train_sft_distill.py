"""Stage 2: QLoRA + reasoning distillation.

    python train_sft_distill.py --traces default --out qwen4b
    python train_sft_distill.py --traces default --out qwen8b --config h100_8b
    python train_sft_distill.py --traces default --out qwen4b --resume
    python train_sft_distill.py --traces default --out r64 --set lora.r=64

Reads  output/traces/<--traces>/traces.jsonl
Writes output/sft/<--out>/{adapter, checkpoints, metrics.json, resolved_config.json}

--traces / --out override traces_run / the run name from the config.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig

import common
import config as cfgmod
import paths
from task import Task


@torch.no_grad()
def score_rows(model, tokenizer, task, df, letter_ids, gen_max_new, batch_size,
               verbose=True):
    """(N,3) raw logits for [A,B,C] AFTER the model's own reasoning.

    Two passes per batch, which is what keeps the scores continuous:
      1. generate the <think> block  (the reasoning does the actual work)
      2. append <answer> and read the letter logits in ONE forward pass
    The official baseline instead takes one sample -> one letter -> a fixed
    probability pair, so its scores have few distinct values and rank-based
    AUROC mostly ties at ~0.5.
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="sft run name -> output/sft/<out>/")
    ap.add_argument("--traces", default=None, help="traces run name (overrides config)")
    ap.add_argument("--resume", action="store_true",
                    help="auto-resume from the newest checkpoint of THIS run")
    ap.add_argument("--resume-from", default=None, help="explicit checkpoint path")
    ap.add_argument("--wandb-id", default=None, help="continue an existing wandb run")
    cfgmod.add_config_args(ap, "sft")
    args = ap.parse_args()
    cfg = cfgmod.resolve(args, "sft")
    traces_run = args.traces or cfg.traces_run

    out_dir = paths.sft_dir(args.out, create=True)
    ckpt_dir = paths.sft_checkpoints(args.out)
    task = Task.from_prompts(cfg.prompts)
    print(f"[prompts] {cfg.prompts}   [traces] {traces_run}   [out] {out_dir}")

    # ── data ────────────────────────────────────────────────────────────────
    train, _ = common.load_data()
    tr, va = common.split_from_cfg(train, cfg)

    tp = paths.traces_jsonl(traces_run)
    paths.require(tp, f"run: python build_traces.py --out {traces_run}")
    tdf = pd.DataFrame([json.loads(l) for l in tp.read_text().splitlines() if l])
    print(f"Loaded {len(tdf)} traces "
          f"(critic: {tdf['critic_score'].value_counts().to_dict()})")

    # Guard: traces must not touch the val block on EITHER axis.
    bad = tdf[tdf["pert"].isin(set(va["perturb_gene"]))
              | tdf["gene"].isin(set(va["target_gene"]))]
    if len(bad):
        print(f"WARNING dropping {len(bad)} traces that touch the val block "
              f"(traces/split config mismatch?)")
        tdf = tdf.drop(bad.index)
    print(f"Training on {len(tdf)} traces | labels {tdf['label'].value_counts().to_dict()}")

    # ── model ───────────────────────────────────────────────────────────────
    m = cfg.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m.name, max_seq_length=m.max_seq_length,
        load_in_4bit=m.load_in_4bit, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora.r, lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout, target_modules=list(cfg.lora.target_modules),
        use_gradient_checkpointing="unsloth", random_state=cfg.split.seed)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_ids = task.resolve_letter_ids(tokenizer)

    # ── dataset ─────────────────────────────────────────────────────────────
    empty = bool(cfg.get_path("ablation.empty_reasoning", False))
    if empty:
        print("ABLATION: empty reasoning (the SynthPert label-only control)")
    texts = [task.train_text(tokenizer, r.pert, r.gene,
                             "" if empty else r.reasoning, r.letter)
             for r in tdf.itertuples(index=False)]
    ds = Dataset.from_dict({"text": texts})
    print("\n--- sample target (tail) ---\n" + texts[0][-380:] + "\n---")

    t = cfg.train
    steps = math.ceil(len(ds) / (t.batch * t.grad_accum)) * t.epochs
    warm = max(1, int(steps * t.warmup_frac))
    print(f"steps={steps} warmup={warm}")

    # ── resume ──────────────────────────────────────────────────────────────
    ckpt = args.resume_from or (common.find_last_checkpoint(ckpt_dir)
                                if args.resume else None)
    prog = common.describe_resume(ckpt) if ckpt else {}
    if prog and prog.get("max_steps") not in (None, steps):
        print(f"WARNING checkpoint max_steps={prog['max_steps']} != {steps}. "
              f"Resume needs IDENTICAL data and args or the LR schedule will not "
              f"line up — consider starting fresh.")

    bf16 = (torch.cuda.is_bf16_supported() if t.bf16 == "auto" else bool(t.bf16))
    run_id = common.init_wandb(cfg, run_name=f"sft-{args.out}",
                               extra_config={"traces_run": traces_run,
                                             "n_traces": len(tdf), "steps": steps},
                               resume_id=args.wandb_id)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=m.max_seq_length,
            per_device_train_batch_size=t.batch,
            gradient_accumulation_steps=t.grad_accum,
            num_train_epochs=t.epochs, learning_rate=t.lr,
            warmup_steps=warm, lr_scheduler_type="cosine",
            logging_steps=t.logging_steps, optim=t.optim,
            weight_decay=t.weight_decay, seed=cfg.split.seed,
            output_dir=str(ckpt_dir), save_strategy=t.save_strategy,
            save_total_limit=t.save_total_limit,
            bf16=bf16, fp16=not bf16,
            report_to="wandb" if run_id else "none"),
    )
    # loss only on the assistant turn (trace + answer), not the long prompt
    trainer = train_on_responses_only(trainer,
                                      instruction_part=m.instruction_part,
                                      response_part=m.response_part)
    trainer.train(resume_from_checkpoint=str(ckpt) if ckpt else None)

    adapter = paths.sft_adapter(args.out)
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    print(f"\nAdapter -> {adapter}")

    metrics = {"traces_run": traces_run, "n_traces": len(tdf), "steps": steps,
               "ablation_empty_reasoning": empty}
    if cfg.eval.enabled:
        FastLanguageModel.for_inference(model)
        model.eval()
        print("\nScoring blocked val (generate reasoning -> read letter logits) ...")
        vl, vtr = score_rows(model, tokenizer, task, va, letter_ids,
                             cfg.eval.gen_max_new, cfg.eval.infer_batch)
        best_T, best = common.tune_temperature(vl, va["label"].values,
                                               tuple(cfg.eval.temperature_grid))
        print(f"\nBLOCKED-VAL SCORE = {best:.4f} (T={best_T})")
        print("\n--- example generated reasoning ---\n" + vtr[0][:600])
        metrics |= {"blocked_val_score": best, "best_temperature": best_T}
        (out_dir / "val_reasoning_samples.txt").write_text("\n\n====\n\n".join(vtr[:20]))

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cfgmod.snapshot(cfg, out_dir, {"run": args.out, "metrics": metrics,
                                   "wandb_id": run_id})
    print(f"\nNEXT: python train_grpo.py --sft {args.out} --out {args.out}-grpo")


if __name__ == "__main__":
    main()
