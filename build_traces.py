"""Stage 1: generate + critic-filter synthetic reasoning traces.

    export TEACHER_API_KEY=...
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

    def __init__(self, model, base_url, api_key, timeout=180, max_retries=3):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key = model, api_key
        self.timeout, self.max_retries = timeout, max_retries

    def __call__(self, prompt, temperature, max_tokens):
        import time
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    self.url, timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json"},
                    json={"model": self.model,
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": temperature, "max_tokens": max_tokens})
                r.raise_for_status()
                txt = r.json()["choices"][0]["message"].get("content") or ""
                # a reasoning teacher may emit its own <think>; strip it
                return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
            except Exception as e:
                if attempt == self.max_retries:
                    print(f"    api fail: {str(e)[:120]}")
                    return None
                time.sleep(2 ** attempt)


def critic_score(chat, prompt_tmpl, pert, gene, meaning, trace, cfg):
    out = chat(prompt_tmpl.format(pert=pert, gene=gene, meaning=meaning,
                                  trace=trace),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="traces run name")
    cfgmod.add_config_args(ap, "traces")
    args = ap.parse_args()
    cfg = cfgmod.resolve(args, "traces")
    P = cfgmod.load_prompts(cfg.prompts)
    print(f"[prompts] {cfg.prompts}")

    key = os.environ.get(cfg.teacher.api_key_env)
    if not key:
        raise SystemExit(f"set ${cfg.teacher.api_key_env}")
    teacher = Chat(cfg.teacher.model, cfg.teacher.base_url, key,
                   cfg.teacher.timeout, cfg.teacher.max_retries)
    critic = teacher if not cfg.get_path("critic.model") else Chat(
        cfg.critic.model, cfg.get_path("critic.base_url") or cfg.teacher.base_url,
        key, cfg.teacher.timeout, cfg.teacher.max_retries)

    leak_re = re.compile("|".join(P.leak_patterns), re.I) \
        if cfg.filters.leak_filter else None

    gpath = paths.grounding_json(cfg.grounding_run)
    paths.require(gpath, f"run: python build_grounding.py --out {cfg.grounding_run}")
    grounding = json.loads(gpath.read_text())["rows"]

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
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
        print(f"Resuming: {len(done)} traces already written")

    lock = threading.Lock()
    stats = {"kept": 0, "leak": 0, "lowscore": 0, "fail": 0}
    fh = out_file.open("a")

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
            return
        if leak_re and leak_re.search(trace):
            with lock:
                stats["leak"] += 1
            return
        if cfg.critic.enabled:
            sc, why = critic_score(critic, P.critic, r.perturb_gene, r.target_gene,
                                   meaning, trace, cfg)
        else:
            sc, why = 5, "critic disabled"
        if sc < cfg.critic.min_score:
            with lock:
                stats["lowscore"] += 1
            return
        rec = {"id": rid, "pert": r.perturb_gene, "gene": r.target_gene,
               "label": r.label, "letter": r.letter, "reasoning": trace,
               "critic_score": sc, "critic_reason": why,
               "teacher": cfg.teacher.model, "prompts": cfg.prompts}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            stats["kept"] += 1
            n = sum(stats.values())
            if n % 25 == 0:
                print(f"  {n}/{len(rows)} kept={stats['kept']} leak={stats['leak']} "
                      f"low={stats['lowscore']} fail={stats['fail']}")

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        list(ex.map(work, list(rows.itertuples(index=False))))
    fh.close()

    total = max(1, sum(stats.values()))
    print(f"\nkept={stats['kept']} leak={stats['leak']} lowscore={stats['lowscore']} "
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
    cfgmod.snapshot(cfg, out_dir, {"stats": stats, "attempted": len(rows)})
    print(f"\nWrote {out_file}")
    print("NEXT: read 5-10 traces by hand. Do they reason about SPECIFIC gene "
          "function, or hand-wave about pathways? The former generalizes.")


if __name__ == "__main__":
    main()
