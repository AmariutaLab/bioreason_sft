"""Pairwise preference audit for two trace files.

The judge compares original vs candidate rationales for the same gene pair and
association. It does not decide whether the association is experimentally true;
it decides which rationale is better as an SFT target.

Example:
    python audit_trace_pairs.py \
      --original-file /home/i3gupta/lustre/tools/mlgenx/output/traces/norules-ungrounded-o4mini/traces.jsonl \
      --candidate-run norules-ungrounded-o4mini-refined-cleanctx-gpt4omini-fallback \
      --out refined-vs-original-gpt4omini-n100 \
      --limit 100 --workers 2 --selected-out refined-selected-gpt4omini-n100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import paths


LETTER_MEANING = {
    "A": "UP-REGULATED",
    "B": "DOWN-REGULATED",
    "C": "NOT differentially expressed",
}
LABEL_TO_LETTER = {"up": "A", "down": "B", "none": "C"}


class Chat:
    def __init__(self, model, base_url, api_key, timeout=180, max_retries=3):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def __call__(self, prompt, temperature, max_tokens):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    self.url,
                    timeout=self.timeout,
                    headers={
                        "Authorization": f"Bearer {self.key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"].get("content") or ""
            except Exception as e:
                if attempt == self.max_retries:
                    print(f"    api fail: {str(e)[:120]}")
                    return None
                time.sleep(2 ** attempt)


def read_jsonl(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def trace_path(run=None, file=None):
    if file:
        return paths.require(Path(file), "expected trace JSONL")
    if run:
        return paths.require(paths.traces_jsonl(run), f"missing trace run {run}")
    raise SystemExit("provide a trace run or file")


def row_letter(row):
    return row.get("letter") or LABEL_TO_LETTER[row["label"]]


def stable_swap(seed, rid):
    h = hashlib.sha256(f"{seed}:{rid}".encode()).hexdigest()
    return int(h[:8], 16) % 2 == 1


def judge_prompt(row, a_text, b_text):
    letter = row_letter(row)
    meaning = LETTER_MEANING[letter]
    return f"""You are an expert molecular and cellular biologist auditing synthetic rationales for supervised fine-tuning.

Task: compare two rationales for the same gene pair and requested association. Do NOT decide whether the requested association is experimentally correct. Choose which rationale is the better training target.

Perturbed gene: {row["pert"]}
Target gene: {row["gene"]}
Association to rationalize: {meaning}

Criteria:
- accurate, specific functions for both genes
- concrete sign-producing mechanism for up/down, or specific functional separation for no-change
- clear distinction between established biology and hypotheses
- compact SynthPert-style reasoning without unnecessary bullets
- no invented direct regulation, promoter binding, named response elements, or macrophage specificity unless well supported
- no source/statistical leakage such as context, databases, GO terms, labels, observed effects, thresholds, FDR, fold changes, or statistical significance

Rationale A:
{a_text}

Rationale B:
{b_text}

Reply only with valid JSON:
{{"winner": "A" | "B" | "tie", "score_a": <0-5>, "score_b": <0-5>, "reason": "<20 words or fewer>"}}"""


def parse_judgment(text):
    if not text:
        return {"winner": "tie", "score_a": 0, "score_b": 0,
                "reason": "api fail", "raw": text or ""}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"winner": "tie", "score_a": 0, "score_b": 0,
                "reason": "unparseable", "raw": text[:500]}
    try:
        obj = json.loads(m.group(0))
        winner = str(obj.get("winner", "tie")).strip()
        if winner not in {"A", "B", "tie"}:
            winner = "tie"
        return {
            "winner": winner,
            "score_a": int(obj.get("score_a", 0)),
            "score_b": int(obj.get("score_b", 0)),
            "reason": str(obj.get("reason", ""))[:120],
            "raw": text[:500],
        }
    except Exception:
        return {"winner": "tie", "score_a": 0, "score_b": 0,
                "reason": "unparseable", "raw": text[:500]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-run", default=None)
    ap.add_argument("--original-file", default=None)
    ap.add_argument("--candidate-run", default=None)
    ap.add_argument("--candidate-file", default=None)
    ap.add_argument("--out", required=True, help="output/trace_audits/<out>/")
    ap.add_argument("--selected-out", default=None,
                    help="optional output/traces/<selected-out>/traces.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--tie-policy", choices=["original", "candidate"],
                    default="original")
    args = ap.parse_args()

    key = os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(f"set ${args.api_key_env}")

    original_path = trace_path(args.original_run, args.original_file)
    candidate_path = trace_path(args.candidate_run, args.candidate_file)
    original = {r["id"]: r for r in read_jsonl(original_path)}
    candidate = {r["id"]: r for r in read_jsonl(candidate_path)}
    ids = sorted(set(original) & set(candidate))
    if args.offset:
        ids = ids[args.offset:]
    if args.limit:
        rng = random.Random(args.seed)
        rng.shuffle(ids)
        ids = sorted(ids[:args.limit])
    print(f"Auditing {len(ids)} common pairs")
    print(f"original:  {original_path}")
    print(f"candidate: {candidate_path}")

    audit_dir = paths.run_dir("trace_audits", args.out, create=True)
    audit_file = audit_dir / "judgments.jsonl"
    selected_file = None
    if args.selected_out:
        selected_dir = paths.run_dir("traces", args.selected_out, create=True)
        selected_file = selected_dir / "traces.jsonl"

    done = set()
    if audit_file.exists():
        for line in audit_file.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
        print(f"Resuming: {len(done)} already judged")

    chat = Chat(args.model, args.base_url, key)
    lock = threading.Lock()
    stats = {"original": 0, "candidate": 0, "tie": 0, "fail": 0}
    selected_stats = {"original": 0, "candidate": 0}
    af = audit_file.open("a")
    sf = selected_file.open("a") if selected_file else None

    def write_selected(rid, chosen_source, judgment):
        row = dict(candidate[rid] if chosen_source == "candidate" else original[rid])
        row.update({
            "pairwise_selected_from": chosen_source,
            "pairwise_audit_out": args.out,
            "pairwise_judge_model": args.model,
            "pairwise_winner": judgment["selected_winner"],
            "pairwise_score_original": judgment["score_original"],
            "pairwise_score_candidate": judgment["score_candidate"],
            "pairwise_reason": judgment["reason"],
        })
        sf.write(json.dumps(row) + "\n")

    def audit_one(rid):
        if rid in done:
            return
        orig = original[rid]
        cand = candidate[rid]
        swap = stable_swap(args.seed, rid)
        a_source = "candidate" if swap else "original"
        b_source = "original" if swap else "candidate"
        a_text = cand["reasoning"] if a_source == "candidate" else orig["reasoning"]
        b_text = orig["reasoning"] if b_source == "original" else cand["reasoning"]
        raw = chat(judge_prompt(orig, a_text, b_text), args.temperature,
                   args.max_tokens)
        parsed = parse_judgment(raw)
        winner = parsed["winner"]
        if winner == "A":
            selected_winner = a_source
        elif winner == "B":
            selected_winner = b_source
        else:
            selected_winner = "tie"
        if selected_winner == "candidate":
            chosen = "candidate"
        elif selected_winner == "original":
            chosen = "original"
        else:
            chosen = args.tie_policy

        score_original = parsed["score_b"] if b_source == "original" else parsed["score_a"]
        score_candidate = parsed["score_a"] if a_source == "candidate" else parsed["score_b"]
        rec = {
            "id": rid,
            "pert": orig["pert"],
            "gene": orig["gene"],
            "label": orig["label"],
            "letter": row_letter(orig),
            "a_source": a_source,
            "b_source": b_source,
            "winner": winner,
            "selected_winner": selected_winner,
            "chosen_source": chosen,
            "score_original": score_original,
            "score_candidate": score_candidate,
            "reason": parsed["reason"],
            "judge_model": args.model,
        }
        with lock:
            af.write(json.dumps(rec) + "\n")
            af.flush()
            if sf:
                write_selected(rid, chosen, rec)
                sf.flush()
                selected_stats[chosen] += 1
            if selected_winner in stats:
                stats[selected_winner] += 1
            else:
                stats["fail"] += 1
            n = sum(stats.values())
            if n % 25 == 0:
                print(f"  {n}/{len(ids)} original={stats['original']} "
                      f"candidate={stats['candidate']} tie={stats['tie']} "
                      f"fail={stats['fail']}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(audit_one, ids))
    af.close()
    if sf:
        sf.close()

    total = max(1, sum(stats.values()))
    summary = {
        "original_path": str(original_path),
        "candidate_path": str(candidate_path),
        "audited": len(ids),
        "stats": stats,
        "selected_stats": selected_stats,
        "candidate_win_rate": stats["candidate"] / total,
        "original_win_rate": stats["original"] / total,
        "tie_rate": stats["tie"] / total,
        "tie_policy": args.tie_policy,
        "selected_out": args.selected_out,
        "judge_model": args.model,
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
