"""Stage 3: GRPO on top of the SFT-distilled adapter.

    python train_grpo.py --sft qwen4b --out qwen4b-grpo
    python train_grpo.py --sft qwen8b --out qwen8b-grpo --config h100
    python train_grpo.py --sft qwen4b --out abl-noverif --config no_verifier
    python train_grpo.py --sft qwen4b --out beta10 --set grpo.beta=0.1

Reads  output/sft/<--sft>/adapter
Writes output/grpo/<--out>/{adapter, best/, metrics.json, resolved_config.json}

Warm start is REQUIRED: RL alone does not add new reasoning priors to a student;
SFT on teacher traces adds the primitives that RL then explores.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainerCallback
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

import common
import config as cfgmod
import paths
from task import Task, LABEL2LETTER

ANSWER_RE = re.compile(r"<answer>\s*([ABC])\s*</answer>", re.I)


def _text(c):
    return c if isinstance(c, str) else c[0]["content"]


def parse_letter(t):
    m = ANSWER_RE.search(t or "")
    return m.group(1).upper() if m else None


# ── reward 1: format ────────────────────────────────────────────────────────
def make_format_reward(weight, think_close):
    def fn(completions, **kw):
        out = []
        for c in completions:
            t = _text(c)
            out.append(weight * (0.7 * (parse_letter(t) is not None)
                                 + 0.3 * (think_close.strip() in t)))
        return out
    fn.__name__ = "format_reward"
    return fn


# ── reward 2: metric-shaped correctness (primary) ──────────────────────────
def make_correctness_reward(weight, partial_de, class_weight):
    """Mirror the DE/DIR decomposition of the official metric.

    You cannot optimize a rank metric per-sample, but you can split credit the
    way the metric splits it:
        true == C : all credit rides on the DE call
        true != C : partial_de for getting DE right + the rest for direction
                    -> "moved, wrong sign" earns partial credit, which is exactly
                       what DIR-AUROC measures separately from DE-AUROC
    """
    def fn(completions, letter=None, **kw):
        out = []
        for c, truth in zip(completions, letter):
            pred = parse_letter(_text(c))
            if pred is None:
                out.append(0.0)
                continue
            if truth == "C":
                r = 1.0 if pred == "C" else 0.0
            else:
                r = (partial_de if pred != "C" else 0.0) \
                    + ((1.0 - partial_de) if pred == truth else 0.0)
            out.append(weight * class_weight.get(truth, 1.0) * r)
        return out
    fn.__name__ = "correctness_reward"
    return fn


# ── reward 3: CollecTRI soft verifier (rbio1-style) ────────────────────────
def make_verifier_reward(grounding, weight, wrong_penalty):
    """Reward reasoning consistent with curated SIGNED regulatory edges.

    You HAVE labels, so this is not needed for correctness — rbio1's soft
    verifiers exist because they lacked labels. Its value here is grounding, and
    it abstains on rows without a curated edge. Kept at low weight on purpose: a
    heavy verifier caps the student at the verifier's own accuracy and invites
    reward hacking.
    """
    def fn(completions, row_id=None, **kw):
        out = []
        for c, rid in zip(completions, row_id):
            pred = parse_letter(_text(c))
            f = grounding.get(rid, {}).get("features", {})
            exp = f.get("expected_letter")
            if pred is None or not f.get("has_edge") or exp is None:
                out.append(0.0)                      # abstain
            else:
                out.append(weight * (1.0 if pred == exp else wrong_penalty))
        return out
    fn.__name__ = "verifier_reward"
    return fn


def validate_verifier(grounding, df, min_acc):
    """Never trust a verifier you have not measured."""
    hits = [(grounding[f"{r.perturb_gene}_{r.target_gene}"]["features"], r.letter)
            for r in df.itertuples(index=False)
            if f"{r.perturb_gene}_{r.target_gene}" in grounding]
    cov = [(f, l) for f, l in hits if f.get("has_edge") and f.get("expected_letter")]
    if not cov:
        print("[verifier] no covered rows -> disabling")
        return 0.0, False
    de = [(f, l) for f, l in cov if l != "C"]
    acc = float(np.mean([f["expected_letter"] == l for f, l in de])) if de else 0.0
    print(f"[verifier] CollecTRI covers {len(cov)}/{len(hits)} rows "
          f"({100*len(cov)/max(1,len(hits)):.1f}%); direction accuracy on covered "
          f"DE rows = {acc:.3f} (n={len(de)})")
    ok = acc >= min_acc
    if not ok:
        print(f"[verifier] BELOW min_direction_acc={min_acc} -> DISABLING. "
              f"A noisy reward is worse than none.")
    return acc, ok


# ── blocked-val callback: the REAL stopping signal ─────────────────────────
class ValAUROCCallback(TrainerCallback):
    """Reward can climb while AUROC falls (calibration collapse): correctness-only
    RL sharpens toward argmax, scores tie, and a rank metric craters. Watch this.
    """

    def __init__(self, model, tokenizer, task, va, letter_ids, cfg, out_dir):
        from train_sft_distill import score_rows
        self._score_rows = score_rows
        self.model, self.tok, self.task = model, tokenizer, task
        self.letter_ids, self.cfg, self.out_dir = letter_ids, cfg, out_dir
        self.va = va.sample(min(cfg.eval.n, len(va)), random_state=cfg.split.seed)
        self.best, self.history = -1.0, []

    def on_step_end(self, args, state, control, **kw):
        if state.global_step == 0 or state.global_step % self.cfg.eval.every:
            return
        training = self.model.training
        self.model.eval()
        logits, _ = self._score_rows(self.model, self.tok, self.task, self.va,
                                     self.letter_ids, self.cfg.eval.gen_max_new,
                                     self.cfg.eval.infer_batch, verbose=False)
        p = common.softmax_T(logits, 1.0)
        s = common.score(self.va["label"].values, p[:, 0], p[:, 1])
        self.history.append({"step": state.global_step, **s})
        flag = ""
        if s["score"] > self.best and self.cfg.eval.save_best:
            self.best = s["score"]
            self.model.save_pretrained(str(self.out_dir / "best"))
            flag = "  <-- best, saved"
        print(f"\n[val {state.global_step}] DE={s['de']:.4f} DIR={s['dir']:.4f} "
              f"SCORE={s['score']:.4f}{flag}")
        try:
            import wandb
            if wandb.run:
                wandb.log({f"val/{k}": v for k, v in s.items()},
                          step=state.global_step)
        except Exception:
            pass
        if training:
            self.model.train()
        return control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="grpo run name -> output/grpo/<out>/")
    ap.add_argument("--sft", default=None, help="sft run name (overrides config)")
    ap.add_argument("--wandb-id", default=None)
    cfgmod.add_config_args(ap, "grpo")
    args = ap.parse_args()
    cfg = cfgmod.resolve(args, "grpo")
    sft_run = args.sft or cfg.sft_run

    out_dir = paths.grpo_dir(args.out, create=True)
    adapter = paths.sft_adapter(sft_run)
    paths.require(adapter, f"run: python train_sft_distill.py --out {sft_run}")
    task = Task.from_prompts(cfg.prompts)
    print(f"[prompts] {cfg.prompts}  [sft] {sft_run}  [out] {out_dir}")

    # The SFT run's prompts MUST match, or the warm start is meaningless.
    sft_cfg_p = paths.sft_dir(sft_run) / "resolved_config.json"
    if sft_cfg_p.exists():
        sp = json.loads(sft_cfg_p.read_text()).get("prompts")
        if sp and sp != cfg.prompts:
            print(f"WARNING SFT run used prompts '{sp}' but this GRPO config uses "
                  f"'{cfg.prompts}'. The answer format/positions must match.")

    train, _ = common.load_data()
    tr, va = common.split_from_cfg(train, cfg)

    grounding = {}
    gp = paths.grounding_json(cfg.grounding_run)
    if gp.exists():
        grounding = json.loads(gp.read_text())["rows"]

    # ── model: base + SFT adapter ───────────────────────────────────────────
    m = cfg.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m.name, max_seq_length=m.max_seq_length,
        load_in_4bit=m.load_in_4bit, dtype=None)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_ids = task.resolve_letter_ids(tokenizer)
    print(f"Warm-started from {adapter}")

    # ── class-balanced prompt buffer ───────────────────────────────────────
    R = cfg.rewards
    counts = tr["label"].value_counts()
    if R.correctness.class_balance:
        n_per = int(counts.min())
        bal = pd.concat([tr[tr["label"] == l].sample(n_per, random_state=cfg.split.seed)
                         for l in ("up", "down", "none")])
        print(f"Balanced buffer: {len(bal)} rows ({n_per}/class) — prevents the "
              f"'always answer C' collapse ('none' is {100*counts['none']/len(tr):.0f}% "
              f"of train)")
        cw = {LABEL2LETTER[l]: float(len(tr) / (3 * counts[l])) for l in counts.index}
        cw = {k: v / max(cw.values()) for k, v in cw.items()}
    else:
        bal, cw = tr, {}
    bal = bal.sample(frac=1, random_state=cfg.split.seed)
    print(f"Class reward weights: { {k: round(v,3) for k,v in cw.items()} }")

    ds = Dataset.from_dict({
        "prompt": [task.think_prompt(tokenizer, r.perturb_gene, r.target_gene)
                   for r in bal.itertuples(index=False)],
        "letter": list(bal["letter"]),
        "row_id": [f"{r.perturb_gene}_{r.target_gene}"
                   for r in bal.itertuples(index=False)]})

    # ── rewards ─────────────────────────────────────────────────────────────
    rewards, verif_acc = [], None
    if R.format.enabled:
        rewards.append(make_format_reward(R.format.weight, task.think_close))
    if R.correctness.enabled:
        rewards.append(make_correctness_reward(R.correctness.weight,
                                               R.correctness.partial_de_credit, cw))
    if R.verifier.enabled and grounding:
        verif_acc, ok = validate_verifier(grounding, tr, R.verifier.min_direction_acc)
        if ok:
            rewards.append(make_verifier_reward(grounding, R.verifier.weight,
                                                R.verifier.wrong_penalty))
    print(f"Rewards: {[f.__name__ for f in rewards]}")

    g = cfg.grpo
    bf16 = torch.cuda.is_bf16_supported()
    run_id = common.init_wandb(cfg, run_name=f"grpo-{args.out}",
                              extra_config={"sft_run": sft_run,
                                            "verifier_acc": verif_acc},
                              resume_id=args.wandb_id)
    cb = ValAUROCCallback(model, tokenizer, task, va, letter_ids, cfg, out_dir)

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, reward_funcs=rewards,
        train_dataset=ds,
        args=GRPOConfig(
            output_dir=str(out_dir / "trainer"),
            learning_rate=g.lr,
            per_device_train_batch_size=g.num_generations,
            gradient_accumulation_steps=g.grad_accum,
            num_generations=g.num_generations,
            max_prompt_length=g.max_prompt_length,
            max_completion_length=g.max_completion_length,
            max_steps=g.max_steps,
            beta=g.beta,                 # KL to reference: the calibration guard
            temperature=g.temperature,
            logging_steps=g.logging_steps,
            save_strategy="no",          # the callback saves the best by val AUROC
            optim="adamw_8bit", bf16=bf16, fp16=not bf16,
            report_to="wandb" if run_id else "none", seed=cfg.split.seed),
        callbacks=[cb])
    trainer.train()

    model.save_pretrained(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))
    metrics = {"sft_run": sft_run, "best_blocked_val": cb.best,
               "history": cb.history, "verifier_direction_acc": verif_acc,
               "rewards": [f.__name__ for f in rewards]}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cfgmod.snapshot(cfg, out_dir, {"run": args.out, "metrics": metrics,
                                   "wandb_id": run_id})
    print(f"\nBest blocked-val during GRPO: {cb.best:.4f}")
    print(f"  best  -> {out_dir/'best'}   (use this)")
    print(f"  final -> {out_dir/'adapter'}")
    if cb.history and cb.history[-1]["score"] < cb.best - 0.01:
        print("NOTE the final model is worse than the best checkpoint — likely "
              "calibration collapse. Raise grpo.beta or lower grpo.max_steps.")


if __name__ == "__main__":
    main()
