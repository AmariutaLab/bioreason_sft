"""Build deterministic label-grounded traces without a teacher API.

This is a fallback baseline, not a replacement for high-quality teacher traces.
It creates the same JSONL schema as build_traces.py from the blocked training
split only, so SFT can run before API credentials or paid teacher traces exist.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import common
import config as cfgmod
import paths


REASONING = {
    "up": (
        "The CRISPRi perturbation is associated with increased expression of "
        "the target gene in the training screen. I treat this pair as a "
        "differential-expression case and assign the up-regulated direction."
    ),
    "down": (
        "The CRISPRi perturbation is associated with decreased expression of "
        "the target gene in the training screen. I treat this pair as a "
        "differential-expression case and assign the down-regulated direction."
    ),
    "none": (
        "The CRISPRi perturbation is not associated with a significant target "
        "gene expression change in the training screen. I therefore assign the "
        "no-change class rather than an up or down direction."
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="traces run name")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="optional balanced cap per class; 0 keeps all rows")
    cfgmod.add_config_args(ap, "sft")
    args = ap.parse_args()
    cfg = cfgmod.resolve(args, "sft")

    train, _ = common.load_data()
    tr, _ = common.split_from_cfg(train, cfg)

    if args.max_per_class > 0:
        rows = []
        for label in ("up", "down", "none"):
            sub = tr[tr["label"] == label]
            rows.append(sub.sample(min(args.max_per_class, len(sub)),
                                   random_state=cfg.split.seed))
        tr = pd.concat(rows).sample(frac=1, random_state=cfg.split.seed)

    out_dir = paths.run_dir("traces", args.out, create=True)
    out_file = out_dir / "traces.jsonl"
    with out_file.open("w") as fh:
        for r in tr.itertuples(index=False):
            rid = f"{r.perturb_gene}_{r.target_gene}"
            rec = {
                "id": rid,
                "pert": r.perturb_gene,
                "gene": r.target_gene,
                "label": r.label,
                "letter": r.letter,
                "reasoning": REASONING[r.label],
                "critic_score": 5,
                "critic_reason": "deterministic label fallback",
                "teacher": "label_fallback",
                "prompts": "label_fallback",
            }
            fh.write(json.dumps(rec) + "\n")

    stats = {
        "kept": int(len(tr)),
        "label_counts": {k: int(v) for k, v in tr["label"].value_counts().items()},
        "max_per_class": int(args.max_per_class),
        "source": "blocked_train_labels",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    cfgmod.snapshot(cfg, out_dir, {"stats": stats})
    print(f"Wrote {len(tr)} fallback traces -> {out_file}")
    print("NEXT: python train_sft_distill.py --config v100_4b "
          f"--traces {args.out} --out qwen4b-v100")


if __name__ == "__main__":
    main()
