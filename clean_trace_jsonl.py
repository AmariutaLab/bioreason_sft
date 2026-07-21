"""Build a clean trainable trace JSONL from one or more trace/reject files.

Use this after interrupted or extended teacher runs. It drops reject records,
requires usable reasoning text, optionally requires a minimum critic score, and
deduplicates by row id.

Examples:
    python clean_trace_jsonl.py \
      --input /home/i3gupta/lustre/tools/mlgenx/output/traces/norules-ungrounded-o4mini-noval-fulltrain/traces.jsonl \
      --out-run norules-ungrounded-o4mini-noval-fulltrain-clean \
      --min-critic-score 4

    python clean_trace_jsonl.py \
      --input /home/i3gupta/lustre/tools/mlgenx/output/traces/norules-ungrounded-o4mini/all_traces.jsonl \
      --input /home/i3gupta/lustre/tools/mlgenx/output/traces/norules-ungrounded-o4mini-noval-fulltrain/traces.jsonl \
      --out-run norules-ungrounded-o4mini-noval-fulltrain-clean \
      --min-critic-score 4
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import paths


REQUIRED = ("id", "pert", "gene", "label", "letter")


def read_jsonl(path: Path):
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield lineno, json.loads(line)
            except Exception as e:
                yield lineno, {"_bad_json": str(e), "_raw": line[:500]}


def output_path(args) -> Path:
    if args.out_file:
        out = Path(args.out_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    if args.out_run:
        out_dir = paths.run_dir("traces", args.out_run, create=True)
        return out_dir / "traces.jsonl"
    raise SystemExit("provide --out-run or --out-file")


def normalize_reasoning(row: dict, allow_legacy: bool) -> str:
    reasoning = row.get("reasoning")
    if reasoning is None and allow_legacy:
        reasoning = row.get("response") or row.get("trace")
    return str(reasoning or "").strip()


def score_ok(row: dict, min_score: int | None) -> bool:
    if min_score is None:
        return True
    try:
        return int(row.get("critic_score")) >= min_score
    except (TypeError, ValueError):
        return False


def source_rank(source: str) -> int:
    # Prefer actual accepted trace files over combined/reject-derived files.
    if source.endswith("/traces.jsonl"):
        return 3
    if source.endswith("/all_traces.jsonl"):
        return 2
    return 1


def choose_existing(existing: dict, candidate: dict, prefer: str) -> dict:
    if prefer == "first":
        return existing
    if prefer == "later":
        return candidate
    # auto: prefer rows from traces.jsonl, then higher critic score, then later.
    es = source_rank(existing.get("_source_file", ""))
    cs = source_rank(candidate.get("_source_file", ""))
    if cs != es:
        return candidate if cs > es else existing
    try:
        esc = int(existing.get("critic_score", -1))
    except (TypeError, ValueError):
        esc = -1
    try:
        csc = int(candidate.get("critic_score", -1))
    except (TypeError, ValueError):
        csc = -1
    if csc != esc:
        return candidate if csc > esc else existing
    return candidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="JSONL file to include; may be repeated")
    ap.add_argument("--out-run", default=None,
                    help="write output/traces/<out-run>/traces.jsonl")
    ap.add_argument("--out-file", default=None,
                    help="write to an explicit JSONL path")
    ap.add_argument("--min-critic-score", type=int, default=4,
                    help="drop rows with critic_score below this; use -1 to disable")
    ap.add_argument("--allow-legacy-text", action="store_true",
                    help="use response/trace when reasoning is missing")
    ap.add_argument("--prefer", choices=["auto", "first", "later"], default="auto",
                    help="duplicate id policy")
    args = ap.parse_args()

    min_score = None if args.min_critic_score < 0 else args.min_critic_score
    selected: dict[str, dict] = {}
    stats = Counter()
    drop_reasons = Counter()

    for raw in args.input:
        path = paths.require(Path(raw), "missing input JSONL")
        for lineno, row in read_jsonl(path):
            stats["input_rows"] += 1
            if "_bad_json" in row:
                drop_reasons["bad_json"] += 1
                continue
            # Reject records from reject files carry `reason`; accepted rows carry
            # `reasoning` and sometimes `critic_reason`.
            if row.get("reason"):
                drop_reasons["reject_record_reason_field"] += 1
                continue
            missing = [k for k in REQUIRED if not row.get(k)]
            if missing:
                drop_reasons[f"missing_{','.join(missing)}"] += 1
                continue
            reasoning = normalize_reasoning(row, args.allow_legacy_text)
            if not reasoning:
                drop_reasons["missing_reasoning"] += 1
                continue
            if not score_ok(row, min_score):
                drop_reasons["critic_score_below_min"] += 1
                continue

            rec = dict(row)
            rec["reasoning"] = reasoning
            rec["_source_file"] = str(path)
            rec["_source_lineno"] = lineno
            rid = str(rec["id"])
            if rid in selected:
                stats["duplicate_ids"] += 1
                chosen = choose_existing(selected[rid], rec, args.prefer)
                if chosen is rec:
                    stats["duplicates_replaced"] += 1
                selected[rid] = chosen
            else:
                selected[rid] = rec

    out = output_path(args)
    labels = Counter()
    sources = Counter()
    with out.open("w") as fh:
        for rid in sorted(selected):
            rec = dict(selected[rid])
            labels[rec.get("label", "?")] += 1
            sources[rec.get("_source_file", "?")] += 1
            rec.pop("_source_file", None)
            rec.pop("_source_lineno", None)
            fh.write(json.dumps(rec) + "\n")

    summary = {
        "inputs": args.input,
        "output": str(out),
        "min_critic_score": min_score,
        "allow_legacy_text": args.allow_legacy_text,
        "prefer": args.prefer,
        "stats": dict(stats),
        "drop_reasons": dict(drop_reasons),
        "kept_rows": len(selected),
        "labels": dict(labels),
        "selected_sources": dict(sources),
    }
    (out.parent / "clean_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
