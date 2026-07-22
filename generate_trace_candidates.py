"""Generate independent candidate rationales for an existing trace set.

Unlike refine_traces.py, this script does NOT show the original rationale to the
teacher. It keeps the same rows/labels, adds cleaned gene grounding plus
train-split neighborhood signatures, and asks the teacher for a fresh
SynthPert-style rationale. Use pairwise audit to keep a candidate only when it
beats the original.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
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
    return row.get("letter") or LABEL_TO_LETTER[row["label"]]


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


MODULE_PATTERNS = [
    ("IFN/JAK/STAT/ISG antiviral",
     r"\b(ifn|interferon|isg|stat|irf|ifi|ifit|ifitm|oas|mx|rsad|viperin|cmpk2|irgm|gbp|h2-|histocompatibility|antiviral|response to virus)\b"),
    ("TLR/TNF/NF-kB inflammatory signaling",
     r"\b(tlr|tnf|traf|irak|myd88|nf-?kb|rela|ikb|mapk|cytokine|chemokine|ccl|cxcl|interleukin|socs|malt1|acod1|inflamm)\b"),
    ("ribosome/translation/nucleolar stress",
     r"\b(rpl|rps|ribosom\w*|rrna|nucleol\w*|translation|trna|eif|dph|hars|odc1|polyamine)\b"),
    ("spliceosome/RNA processing/export",
     r"\b(prpf|snr|srsf|cwc|splice|splicing|spliceosom|rna helicase|mrna processing|rna export|dhx|ddx)\b"),
    ("chromatin/transcriptional regulation",
     r"\b(arid|smarc|kdm|brd|ncor|med|histone|chromatin|nucleosome|transcription factor|polycomb|hdac|coactivator|corepressor)\b"),
    ("lysosome/autophagy/vesicle trafficking",
     r"\b(lysosom\w*|endosom\w*|autophag\w*|golgi|vesicle|v-?atpase|atp6v|chmp|escrt|exoc|cop|gosr|bloc|dock|tmem|mcoln|tfeb|tfe3)\b"),
    ("mitochondrial/metabolic/redox",
     r"\b(mitochond\w*|oxid\w*|redox|respiration|nad|mthfd|aldh|prdx|selen\w*|nampt|lipid|lipase|fatty|folate|one-carbon|hif|hypoxi\w*)\b"),
    ("proteostasis/chaperone/ER stress",
     r"\b(chaperon\w*|proteostasis|unfolded protein|upr|er stress|hsp|cct|pfdn|dnaj|sec|ddost|ero1|proteasom\w*|ubiquitin)\b"),
    ("cytoskeleton/adhesion/phagocytosis",
     r"\b(actin|cytoskeleton|microtubule|cdc42|rac|integrin|fermt|adhesion|phagocyt|motility|arpc|arp2/3|tubulin)\b"),
    ("cell cycle/DNA repair",
     r"\b(cell cycle|mitotic|dna repair|replication|checkpoint|mms|cdc|hira|npat|slbp|p53|tp53)\b"),
]


def parse_gene_contexts(grounding):
    gene_ctx = defaultdict(list)
    for row in grounding.values():
        for line in row.get("context", "").splitlines():
            m = re.match(r"^(?:PERTURBED|TARGET):\s+([^=\s]+).*", line)
            if m:
                sym = m.group(1).strip()
                if line not in gene_ctx[sym]:
                    gene_ctx[sym].append(line)
    return {k: " ".join(v[:3]) for k, v in gene_ctx.items()}


def gene_modules(sym, gene_ctx):
    text = f"{sym} {gene_ctx.get(sym, '')}".lower()
    mods = [name for name, pat in MODULE_PATTERNS if re.search(pat, text, re.I)]
    return mods[:4] or ["poorly characterized / no strong module"]


def summarize_label_rows(rows, gene_col, gene_ctx, max_genes):
    lines = []
    counts = Counter(r["label"] for r in rows)
    lines.append(f"counts: up={counts.get('up', 0)}, down={counts.get('down', 0)}, none={counts.get('none', 0)}")
    for label in ("up", "down", "none"):
        genes = [r[gene_col] for r in rows if r["label"] == label]
        if not genes:
            continue
        shown = genes[:max_genes]
        module_counts = Counter()
        for g in genes:
            module_counts.update(gene_modules(g, gene_ctx))
        top_mods = ", ".join(f"{m} ({n})" for m, n in module_counts.most_common(4))
        lines.append(f"{label}: {', '.join(shown)}; modules: {top_mods}")
    return "\n".join(lines)


def build_program_signature(row, train_rows, gene_ctx, max_genes=8):
    pert = row["pert"]
    gene = row["gene"]
    same_pert = [r for r in train_rows
                 if r.perturb_gene == pert and r.target_gene != gene]
    same_gene = [r for r in train_rows
                 if r.target_gene == gene and r.perturb_gene != pert]

    lines = [
        "PROGRAM SIGNATURE HINTS (weak train-split neighborhood summaries; do not cite them):",
        f"Perturbed gene modules for {pert}: {', '.join(gene_modules(pert, gene_ctx))}",
        f"Target gene modules for {gene}: {', '.join(gene_modules(gene, gene_ctx))}",
    ]
    if same_pert:
        pert_rows = [
            {"label": r.label, "target_gene": r.target_gene}
            for r in same_pert
        ]
        lines.append(f"Other target responses when {pert} is perturbed:")
        lines.append(summarize_label_rows(pert_rows, "target_gene", gene_ctx, max_genes))
    else:
        lines.append(f"No other train-split target responses are available for {pert}.")
    if same_gene:
        gene_rows = [
            {"label": r.label, "perturb_gene": r.perturb_gene}
            for r in same_gene
        ]
        lines.append(f"Other perturbations affecting {gene}:")
        lines.append(summarize_label_rows(gene_rows, "perturb_gene", gene_ctx, max_genes))
    else:
        lines.append(f"No other train-split perturbation responses are available for {gene}.")
    lines.append("Use these hints only to choose a cellular program or canonical module; do not mention neighbor genes as evidence unless they are mechanistically relevant.")
    return "\n".join(lines)


def quality_prefilter(trace, pert, gene, cfg):
    if cfg.get_path("filters.require_gene_mentions", False):
        for sym in (pert, gene):
            if not re.search(rf"\b{re.escape(str(sym))}\b", trace, re.I):
                return False, f"missing {sym}"
    for pat in cfg.get_path("filters.reject_patterns", []) or []:
        if re.search(pat, trace, re.I):
            return False, f"reject pattern: {pat[:35]}"
    return True, ""


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
    ap.add_argument("--source-run", default=None)
    ap.add_argument("--source-file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--labels", default=None,
                    help="comma-separated labels to generate, e.g. none,down")
    ap.add_argument("--seed", type=int, default=83)
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
        cfg.critic.model,
        cfg.get_path("critic.base_url") or cfg.teacher.base_url,
        key,
        cfg.teacher.timeout,
        cfg.teacher.max_retries,
        cfg.critic.get("reasoning_effort"),
    )

    grounding_run = cfg.get_path("grounding.run", cfg.get("grounding_run", "default"))
    grounding = {}
    gene_ctx = {}
    if cfg.get_path("grounding.enabled", True):
        gpath = paths.grounding_json(grounding_run)
        paths.require(gpath, f"run: python build_grounding.py --out {grounding_run}")
        grounding = json.loads(gpath.read_text())["rows"]
        cleaned = {rid: {"context": filter_context(row.get("context", ""), cfg)}
                   for rid, row in grounding.items()}
        gene_ctx = parse_gene_contexts(cleaned)
        print(f"[grounding] {grounding_run} enabled")
    else:
        print("[grounding] disabled")

    import common
    train, _ = common.load_data()
    tr, _ = common.split_from_cfg(train, cfg, verbose=True)
    train_rows = list(tr.itertuples(index=False))

    src = source_path(args)
    rows = read_jsonl(src)
    if args.labels:
        wanted = {x.strip() for x in args.labels.split(",") if x.strip()}
        rows = [r for r in rows if r["label"] in wanted]
    if args.offset:
        rows = rows[args.offset:]
    if args.limit:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[:args.limit]
        rows = sorted(rows, key=lambda r: r["id"])
    print(f"Generating {len(rows)} candidate traces from {src}")

    out_dir = paths.run_dir("traces", args.out, create=True)
    out_file = out_dir / "traces.jsonl"
    reject_file = out_dir / "rejects.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
        print(f"Resuming: {len(done)} candidate traces already written")

    leak_re = re.compile("|".join(P.leak_patterns), re.I) \
        if cfg.filters.leak_filter else None
    lock = threading.Lock()
    stats = {"kept": 0, "leak": 0, "prefilter": 0, "lowscore": 0, "fail": 0}
    fh = out_file.open("a")
    rfh = reject_file.open("a")

    def reject(reason, row, trace="", critic_score_value=None, critic_reason=""):
        rec = dict(row)
        rec.update({
            "reason": reason,
            "trace": trace,
            "critic_score": critic_score_value,
            "critic_reason": critic_reason,
            "candidate_teacher": cfg.teacher.model,
            "prompts": cfg.prompts,
            "source_reasoning": row.get("reasoning", ""),
        })
        rfh.write(json.dumps(rec) + "\n")
        rfh.flush()

    def work(row):
        rid = row["id"]
        if rid in done:
            return
        letter = row_letter(row)
        meaning = LETTER_MEANING[letter]
        gene_context = filter_context(grounding.get(rid, {}).get("context", ""), cfg)
        signature_context = build_program_signature(
            row, train_rows, gene_ctx, cfg.get_path("signature.max_genes", 8))
        context = "\n\n".join(x for x in (gene_context, signature_context) if x)
        tmpl = P.prompt_none if row["label"] == "none" else P.prompt_de
        prompt = tmpl.format(
            context=context,
            pert=row["pert"],
            gene=row["gene"],
            meaning=meaning,
            label=row["label"],
            letter=letter,
            rules=P.rules,
        )
        trace = teacher(prompt, cfg.teacher.temperature, cfg.teacher.max_tokens)
        if not trace or len(trace) < cfg.filters.min_chars:
            with lock:
                stats["fail"] += 1
                reject("fail_or_short", row, trace or "")
            return
        ok, why_prefilter = quality_prefilter(trace, row["pert"], row["gene"], cfg)
        if not ok:
            with lock:
                stats["prefilter"] += 1
                reject(f"prefilter: {why_prefilter}", row, trace)
            return
        if leak_re and leak_re.search(trace):
            with lock:
                stats["leak"] += 1
                reject("leak", row, trace)
            return
        if cfg.critic.enabled:
            sc, why = critic_score(critic, P.critic, row, meaning, trace, cfg, context)
        else:
            sc, why = 5, "critic disabled"
        if sc < cfg.critic.min_score:
            with lock:
                stats["lowscore"] += 1
                reject("lowscore", row, trace, sc, why)
            return
        rec = dict(row)
        rec.update({
            "reasoning": trace,
            "source_reasoning": row.get("reasoning", ""),
            "source_teacher": row.get("teacher"),
            "source_prompts": row.get("prompts"),
            "source_critic_score": row.get("critic_score"),
            "critic_score": sc,
            "critic_reason": why,
            "teacher": cfg.teacher.model,
            "candidate_teacher": cfg.teacher.model,
            "prompts": cfg.prompts,
            "grounding_enabled": bool(grounding),
            "grounding_run": grounding_run if grounding else None,
        })
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
        list(ex.map(work, rows))
    fh.close()
    rfh.close()

    total = max(1, sum(stats.values()))
    print(f"\nkept={stats['kept']} leak={stats['leak']} "
          f"prefilter={stats['prefilter']} lowscore={stats['lowscore']} "
          f"fail={stats['fail']}  (keep rate {100*stats['kept']/total:.1f}%)")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    cfgmod.snapshot(cfg, out_dir, {
        "stats": stats,
        "attempted": len(rows),
        "source": str(src),
        "labels_filter": args.labels,
    })
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
