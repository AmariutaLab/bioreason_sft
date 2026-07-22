"""Rewrite an existing trace set with a stronger teacher prompt.

This is for improving a known-good trace distribution rather than sampling new
rows. The input can be an output/traces/<run>/traces.jsonl run name or an
absolute JSONL path.

Example:
    python refine_traces.py \
      --source-file /home/i3gupta/lustre/tools/mlgenx/output/traces/norules-ungrounded-o4mini/traces.jsonl \
      --out norules-ungrounded-o4mini-refined-cleanctx-smoke \
      --config refine_synthpert_cleanctx_gpt4o_mini \
      --limit 40 --set workers=2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import config as cfgmod
import paths

LETTER_MEANING = {
    "A": "UP-REGULATED",
    "B": "DOWN-REGULATED",
    "C": "NOT differentially expressed",
}
LABEL_TO_LETTER = {"up": "A", "down": "B", "none": "C"}


class Chat:
    """Minimal OpenAI-compatible client."""

    def __init__(self, model, base_url, api_key, timeout=180, max_retries=3,
                 reasoning_effort=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort

    def __call__(self, prompt, temperature, max_tokens):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        reasoning_model = bool(self.reasoning_effort) or re.match(r"^o[0-9]", self.model)
        if reasoning_model:
            payload["max_completion_tokens"] = max_tokens
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["temperature"] = temperature
            payload["max_tokens"] = max_tokens

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
                txt = r.json()["choices"][0]["message"].get("content") or ""
                return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
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


def source_path(args):
    if args.source_file:
        return paths.require(Path(args.source_file), "expected source trace JSONL")
    if args.source_run:
        return paths.require(paths.traces_jsonl(args.source_run),
                             f"missing source trace run {args.source_run}")
    raise SystemExit("provide --source-run or --source-file")


def row_letter(row):
    letter = row.get("letter")
    if letter:
        return letter
    return LABEL_TO_LETTER[row["label"]]


def quality_prefilter(trace, pert, gene, cfg):
    if cfg.get_path("filters.require_gene_mentions", False):
        for sym in (pert, gene):
            if not re.search(rf"\b{re.escape(str(sym))}\b", trace, re.I):
                return False, f"missing {sym}"
    for pat in cfg.get_path("filters.reject_patterns", []) or []:
        if re.search(pat, trace, re.I):
            return False, f"reject pattern: {pat[:35]}"
    return True, ""


def filter_context(ctx, cfg):
    line_pats = cfg.get_path("grounding.drop_context_lines", []) or []
    part_pats = cfg.get_path("grounding.drop_context_parts", []) or []
    go_term_pats = cfg.get_path("grounding.drop_go_terms", []) or []
    if not ctx or (not line_pats and not part_pats and not go_term_pats):
        return ctx
    kept = []
    for line in ctx.splitlines():
        if any(re.search(pat, line, re.I) for pat in line_pats):
            continue
        if go_term_pats and "| GO:" in line:
            prefix, _, terms = line.partition("| GO:")
            kept_terms = []
            for term in terms.split(";"):
                term = term.strip()
                if term and not any(re.search(pat, term, re.I) for pat in go_term_pats):
                    kept_terms.append(term)
            line = prefix.rstrip()
            if kept_terms:
                line = f"{line} | GO: {'; '.join(kept_terms)}"
        for pat in part_pats:
            line = re.sub(pat, "", line, flags=re.I).rstrip()
        if line:
            kept.append(line)
    return "\n".join(kept)


def critic_score(chat, prompt_tmpl, row, meaning, trace, cfg, context):
    out = chat(
        prompt_tmpl.format(
            pert=row["pert"],
            gene=row["gene"],
            meaning=meaning,
            trace=trace,
            context=context,
        ),
        cfg.critic.temperature,
        cfg.critic.max_tokens,
    )
    if not out:
        return 0, "api fail"
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return 0, "unparseable"
    try:
        obj = json.loads(m.group(0))
        return int(obj.get("score", 0)), str(obj.get("reason", ""))[:80]
    except Exception:
        return 0, "unparseable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output/traces/<out>/")
    ap.add_argument("--source-run", default=None,
                    help="input output/traces/<source-run>/traces.jsonl")
    ap.add_argument("--source-file", default=None,
                    help="absolute input traces.jsonl path")
    ap.add_argument("--limit", type=int, default=0,
                    help="rewrite at most this many source traces")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many source traces before --limit")
    ap.add_argument("--fallback-original-on-reject", action="store_true",
                    help="write the original trace if the rewrite is rejected")
    cfgmod.add_config_args(ap, "traces")
    args = ap.parse_args()
    cfg = cfgmod.resolve(args, "traces")
    P = cfgmod.load_prompts(cfg.prompts)
    if "refine" not in P:
        raise SystemExit(f"prompt {cfg.prompts} must define a refine template")
    print(f"[prompts] {cfg.prompts}")

    key = os.environ.get(cfg.teacher.api_key_env)
    if not key:
        raise SystemExit(f"set ${cfg.teacher.api_key_env}")
    teacher = Chat(cfg.teacher.model, cfg.teacher.base_url, key,
                   cfg.teacher.timeout, cfg.teacher.max_retries,
                   cfg.teacher.get("reasoning_effort"))
    critic = teacher if not cfg.get_path("critic.model") else Chat(
        cfg.critic.model,
        cfg.get_path("critic.base_url") or cfg.teacher.base_url,
        key,
        cfg.teacher.timeout,
        cfg.teacher.max_retries,
        cfg.critic.get("reasoning_effort"),
    )

    grounding_enabled = bool(cfg.get_path("grounding.enabled", True))
    grounding_run = cfg.get_path("grounding.run", cfg.get("grounding_run", "default"))
    if grounding_enabled:
        gpath = paths.grounding_json(grounding_run)
        paths.require(gpath, f"run: python build_grounding.py --out {grounding_run}")
        grounding = json.loads(gpath.read_text())["rows"]
        print(f"[grounding] {grounding_run} enabled")
    else:
        grounding = {}
        print("[grounding] disabled")

    src_path = source_path(args)
    rows = read_jsonl(src_path)
    if args.offset:
        rows = rows[args.offset:]
    if args.limit:
        rows = rows[:args.limit]
    print(f"Rewriting {len(rows)} traces from {src_path}")

    out_dir = paths.run_dir("traces", args.out, create=True)
    out_file = out_dir / "traces.jsonl"
    reject_file = out_dir / "rejects.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
        print(f"Resuming: {len(done)} rewritten traces already written")

    leak_re = re.compile("|".join(P.leak_patterns), re.I) \
        if cfg.filters.leak_filter else None
    lock = threading.Lock()
    stats = {"kept": 0, "fallback": 0, "leak": 0, "prefilter": 0,
             "lowscore": 0, "fail": 0}
    fh = out_file.open("a")
    rfh = reject_file.open("a")

    def processed_count():
        return stats["kept"] + stats["leak"] + stats["prefilter"] \
            + stats["lowscore"] + stats["fail"]

    def reject(reason, row, revised="", critic_score_value=None, critic_reason=""):
        rec = dict(row)
        rec.update({
            "reason": reason,
            "trace": revised,
            "original_reasoning": row.get("reasoning", ""),
            "critic_score": critic_score_value,
            "critic_reason": critic_reason,
            "refiner": cfg.teacher.model,
            "prompts": cfg.prompts,
            "grounding_enabled": grounding_enabled,
            "grounding_run": grounding_run if grounding_enabled else None,
        })
        rfh.write(json.dumps(rec) + "\n")
        rfh.flush()

    def write_original_fallback(row, reject_reason, revised="",
                                critic_score_value=None, critic_reason=""):
        rec = dict(row)
        rec.update({
            "reasoning": row.get("reasoning", ""),
            "original_reasoning": row.get("reasoning", ""),
            "original_teacher": row.get("teacher"),
            "original_prompts": row.get("prompts"),
            "original_critic_score": row.get("critic_score"),
            "original_critic_reason": row.get("critic_reason"),
            "refine_status": "fallback_original",
            "refine_reject_reason": reject_reason,
            "refine_reject_trace": revised,
            "refine_reject_critic_score": critic_score_value,
            "refine_reject_critic_reason": critic_reason,
            "refiner": cfg.teacher.model,
            "refine_prompts": cfg.prompts,
            "grounding_enabled": grounding_enabled,
            "grounding_run": grounding_run if grounding_enabled else None,
        })
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        stats["fallback"] += 1

    def work(row):
        rid = row["id"]
        if rid in done:
            return
        letter = row_letter(row)
        meaning = LETTER_MEANING[letter]
        ctx = filter_context(grounding.get(rid, {}).get("context", ""), cfg)
        prompt = P.refine.format(
            context=ctx,
            pert=row["pert"],
            gene=row["gene"],
            meaning=meaning,
            label=row["label"],
            letter=letter,
            original_trace=row.get("reasoning", ""),
            rules=P.rules,
        )
        revised = teacher(prompt, cfg.teacher.temperature, cfg.teacher.max_tokens)
        if not revised or len(revised) < cfg.filters.min_chars:
            with lock:
                stats["fail"] += 1
                reject("fail_or_short", row, revised or "")
                if args.fallback_original_on_reject:
                    write_original_fallback(row, "fail_or_short", revised or "")
            return
        ok, why_prefilter = quality_prefilter(revised, row["pert"], row["gene"], cfg)
        if not ok:
            with lock:
                stats["prefilter"] += 1
                reason = f"prefilter: {why_prefilter}"
                reject(reason, row, revised)
                if args.fallback_original_on_reject:
                    write_original_fallback(row, reason, revised)
            return
        if leak_re and leak_re.search(revised):
            with lock:
                stats["leak"] += 1
                reject("leak", row, revised)
                if args.fallback_original_on_reject:
                    write_original_fallback(row, "leak", revised)
            return
        if cfg.critic.enabled:
            sc, why = critic_score(critic, P.critic, row, meaning, revised, cfg, ctx)
        else:
            sc, why = 5, "critic disabled"
        if sc < cfg.critic.min_score:
            with lock:
                stats["lowscore"] += 1
                reject("lowscore", row, revised, sc, why)
                if args.fallback_original_on_reject:
                    write_original_fallback(row, "lowscore", revised, sc, why)
            return
        rec = dict(row)
        rec.update({
            "reasoning": revised,
            "original_reasoning": row.get("reasoning", ""),
            "original_teacher": row.get("teacher"),
            "original_prompts": row.get("prompts"),
            "original_critic_score": row.get("critic_score"),
            "original_critic_reason": row.get("critic_reason"),
            "critic_score": sc,
            "critic_reason": why,
            "refine_status": "refined",
            "refiner": cfg.teacher.model,
            "prompts": cfg.prompts,
            "grounding_enabled": grounding_enabled,
            "grounding_run": grounding_run if grounding_enabled else None,
        })
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            stats["kept"] += 1
            n = processed_count()
            if n % 25 == 0:
                print(f"  {n}/{len(rows)} kept={stats['kept']} "
                      f"fallback={stats['fallback']} leak={stats['leak']} "
                      f"prefilter={stats['prefilter']} low={stats['lowscore']} "
                      f"fail={stats['fail']}")

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        list(ex.map(work, rows))
    fh.close()
    rfh.close()

    total = max(1, processed_count())
    output = stats["kept"] + stats["fallback"]
    print(f"\nkept={stats['kept']} fallback={stats['fallback']} leak={stats['leak']} "
          f"prefilter={stats['prefilter']} lowscore={stats['lowscore']} "
          f"fail={stats['fail']}  "
          f"(refined keep rate {100*stats['kept']/total:.1f}%, "
          f"output rate {100*output/total:.1f}%)")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    cfgmod.snapshot(cfg, out_dir, {
        "stats": stats,
        "attempted": len(rows),
        "source": str(src_path),
        "grounding_enabled": grounding_enabled,
        "grounding_run": grounding_run if grounding_enabled else None,
    })
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
