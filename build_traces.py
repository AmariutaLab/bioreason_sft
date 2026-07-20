"""Stage 1: generate + critic-filter synthetic reasoning traces.

    export OPENAI_API_KEY=...
    python build_traces.py --out smoke   --config smoke     # ALWAYS smoke first
    python build_traces.py --out default --config default

Writes output/traces/<out>/traces.jsonl (+ resolved_config.json, stats.json).
Resumable: re-running skips ids already in traces.jsonl.

Strategy = SynthPert "Approach 2": the teacher is GIVEN the label and asked to
RATIONALIZE it. The teacher's own accuracy stops mattering (o4-mini: ~52% on this
task; its traces trained an 8B student to ~89%).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

import common
import config as cfgmod
import paths

LETTER_MEANING = {"A": "UP-REGULATED", "B": "DOWN-REGULATED",
                  "C": "NOT significantly changed"}


class Chat:
    """Minimal OpenAI-compatible client (OpenAI / DeepSeek / OpenRouter / vLLM)."""

    def __init__(self, model, base_url, api_key, timeout=180, max_retries=3,
                 reasoning_effort=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key = model, api_key
        self.timeout, self.max_retries = timeout, max_retries
        self.reasoning_effort = reasoning_effort

    def __call__(self, prompt, temperature, max_tokens):
        import time
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
                    self.url, timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json"},
                    json=payload)
                r.raise_for_status()
                txt = r.json()["choices"][0]["message"].get("content") or ""
                # a reasoning teacher may emit its own <think>; strip it
                return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
            except Exception as e:
                if attempt == self.max_retries:
                    print(f"    api fail: {str(e)[:120]}")
                    return None
                time.sleep(2 ** attempt)


def critic_score(chat, prompt_tmpl, pert, gene, meaning, trace, cfg, context=""):
    out = chat(prompt_tmpl.format(pert=pert, gene=gene, meaning=meaning,
                                  trace=trace, context=context),
               cfg.critic.temperature, cfg.critic.max_tokens)
    if not out:
        return 0, "api fail"
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return 0, "unparseable"
    try:
        j = json.loads(m.group(0))
        return int(j.get("score", 0)), str(j.get("reason", ""))[:60]
    except Exception:
        return 0, "unparseable"


def quality_prefilter(trace, pert, gene, cfg):
    """Cheap deterministic QA before spending critic calls.

    The LLM critic can be over-generous. These checks catch failure modes that
    are unambiguously bad training data: the rationale does not mention the
    requested genes, or it uses banned generic/meta phrases.
    """
    if cfg.get_path("filters.require_gene_mentions", False):
        for sym in (pert, gene):
            if not re.search(rf"\b{re.escape(str(sym))}\b", trace, re.I):
                return False, f"missing {sym}"
    for pat in cfg.get_path("filters.reject_patterns", []) or []:
        if re.search(pat, trace, re.I):
            return False, f"reject pattern: {pat[:35]}"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="traces run name")
    ap.add_argument("--no-grounding", action="store_true",
                    help="do not load or inject grounding context into teacher prompts")
    cfgmod.add_config_args(ap, "traces")
    args = ap.parse_args()
    cfg = cfgmod.resolve(args, "traces")
    P = cfgmod.load_prompts(cfg.prompts)
    print(f"[prompts] {cfg.prompts}")

    key = os.environ.get(cfg.teacher.api_key_env)
    if not key:
        raise SystemExit(f"set ${cfg.teacher.api_key_env}")
    teacher = Chat(cfg.teacher.model, cfg.teacher.base_url, key,
                   cfg.teacher.timeout, cfg.teacher.max_retries,
                   cfg.teacher.get("reasoning_effort"))
    critic = teacher if not cfg.get_path("critic.model") else Chat(
        cfg.critic.model, cfg.get_path("critic.base_url") or cfg.teacher.base_url,
        key, cfg.teacher.timeout, cfg.teacher.max_retries,
        cfg.critic.get("reasoning_effort"))

    leak_re = re.compile("|".join(P.leak_patterns), re.I) \
        if cfg.filters.leak_filter else None

    grounding_enabled = bool(cfg.get_path("grounding.enabled", True)) and not args.no_grounding
    grounding_run = cfg.get_path("grounding.run", cfg.get("grounding_run", "default"))
    if grounding_enabled:
        gpath = paths.grounding_json(grounding_run)
        paths.require(gpath, f"run: python build_grounding.py --out {grounding_run}")
        grounding = json.loads(gpath.read_text())["rows"]
        print(f"[grounding] {grounding_run} enabled")
    else:
        grounding = {}
        print("[grounding] disabled; teacher sees no retrieved context")

    train, _ = common.load_data()
    tr, va = common.split_from_cfg(train, cfg)     # SAME split as the trainers

    # Balanced sampling: 'none' is 55%, 'down' only 14%. Proportional sampling
    # wastes teacher budget on the easy majority and starves 'down'.
    s = cfg.sampling
    if s.balance_classes:
        per = s.n // 3
        rows = pd.concat([tr[tr["label"] == l].sample(min(per, (tr["label"] == l).sum()),
                                                      random_state=s.seed)
                          for l in ("up", "down", "none")])
    else:
        rows = tr.sample(min(s.n, len(tr)), random_state=s.seed)
    rows = rows.sample(frac=1, random_state=s.seed)
    print(f"Attempting {len(rows)} traces: {rows['label'].value_counts().to_dict()}")

    out_dir = paths.run_dir("traces", args.out, create=True)
    out_file = out_dir / "traces.jsonl"
    reject_file = out_dir / "rejects.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
        print(f"Resuming: {len(done)} traces already written")

    lock = threading.Lock()
    stats = {"kept": 0, "leak": 0, "prefilter": 0, "lowscore": 0, "fail": 0}
    fh = out_file.open("a")
    rfh = reject_file.open("a")

    def reject(reason, r, trace="", critic_score=None, critic_reason=""):
        rec = {"id": f"{r.perturb_gene}_{r.target_gene}",
               "pert": r.perturb_gene, "gene": r.target_gene,
               "label": r.label, "letter": r.letter,
               "reason": reason, "trace": trace,
               "critic_score": critic_score, "critic_reason": critic_reason,
               "grounding_enabled": grounding_enabled,
               "grounding_run": grounding_run if grounding_enabled else None}
        rfh.write(json.dumps(rec) + "\n")
        rfh.flush()

    def work(r):
        rid = f"{r.perturb_gene}_{r.target_gene}"
        if rid in done:
            return
        ctx = grounding.get(rid, {}).get("context", "")
        meaning = LETTER_MEANING[r.letter]
        tmpl = P.prompt_none if r.label == "none" else P.prompt_de
        trace = teacher(tmpl.format(context=ctx, pert=r.perturb_gene,
                                    gene=r.target_gene, meaning=meaning,
                                    rules=P.rules),
                        cfg.teacher.temperature, cfg.teacher.max_tokens)
        if not trace or len(trace) < cfg.filters.min_chars:
            with lock:
                stats["fail"] += 1
                reject("fail_or_short", r, trace or "")
            return
        ok, why_prefilter = quality_prefilter(trace, r.perturb_gene,
                                              r.target_gene, cfg)
        if not ok:
            with lock:
                stats["prefilter"] += 1
                reject(f"prefilter: {why_prefilter}", r, trace)
            return
        if leak_re and leak_re.search(trace):
            with lock:
                stats["leak"] += 1
                reject("leak", r, trace)
            return
        if cfg.critic.enabled:
            sc, why = critic_score(critic, P.critic, r.perturb_gene, r.target_gene,
                                   meaning, trace, cfg, ctx)
        else:
            sc, why = 5, "critic disabled"
        if sc < cfg.critic.min_score:
            with lock:
                stats["lowscore"] += 1
                reject("lowscore", r, trace, sc, why)
            return
        rec = {"id": rid, "pert": r.perturb_gene, "gene": r.target_gene,
               "label": r.label, "letter": r.letter, "reasoning": trace,
               "critic_score": sc, "critic_reason": why,
               "teacher": cfg.teacher.model, "prompts": cfg.prompts,
               "grounding_enabled": grounding_enabled,
               "grounding_run": grounding_run if grounding_enabled else None}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            stats["kept"] += 1
            n = sum(stats.values())
            if n % 25 == 0:
                print(f"  {n}/{len(rows)} kept={stats['kept']} leak={stats['leak']} "
                      f"prefilter={stats['prefilter']} low={stats['lowscore']} "
                      f"fail={stats['fail']}")

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        list(ex.map(work, list(rows.itertuples(index=False))))
    fh.close()
    rfh.close()

    total = max(1, sum(stats.values()))
    print(f"\nkept={stats['kept']} leak={stats['leak']} "
          f"prefilter={stats['prefilter']} lowscore={stats['lowscore']} "
          f"fail={stats['fail']}  (keep rate {100*stats['kept']/total:.1f}%)")
    print("SynthPert kept ~2% after filtering and still beat full-data training — "
          "a low keep rate is not a bug.")
    if stats["leak"] > 0.2 * total:
        print("HIGH LEAK RATE: the teacher is reasoning backward from the answer "
              "you handed it. Tighten prompts/teacher/<name>.yaml rules.")
    if stats["lowscore"] > 0.9 * total:
        print("NEARLY EVERYTHING FILTERED: loosen --set critic.min_score=4 or use "
              "a stronger teacher.")

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    cfgmod.snapshot(cfg, out_dir, {"stats": stats, "attempted": len(rows),
                                   "grounding_enabled": grounding_enabled,
                                   "grounding_run": grounding_run if grounding_enabled else None})
    print(f"\nWrote {out_file}")
    print("NEXT: read 5-10 traces by hand. Do they reason about SPECIFIC gene "
          "function, or hand-wave about pathways? The former generalizes.")


if __name__ == "__main__":
    main()
