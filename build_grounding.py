"""Stage 0: biological grounding for the TEACHER prompt and the GRPO verifier.

    python build_grounding.py --out default
    python build_grounding.py --out no_collectri --set sources.collectri.enabled=false

Writes output/grounding/<out>/grounding.json + resolved_config.json.

Runs OFFLINE at data-prep time. Track C forbids tools at INFERENCE, not during
data prep — grounding gets distilled into weights via traces, never injected as
student context.
"""
from __future__ import annotations

import argparse
import json
import time

import requests

import common
import config as cfgmod
import paths

MYGENE_URL = "https://mygene.info/v3/query"


# ── mygene.info: gene name + RefSeq summary (function) + GO ─────────────────
def fetch_annotations(symbols, cfg):
    m = cfg.sources.mygene
    if not m.enabled:
        return {}
    out, symbols = {}, sorted(set(symbols))
    for i in range(0, len(symbols), m.batch_size):
        chunk = symbols[i:i + m.batch_size]
        try:
            r = requests.post(MYGENE_URL, timeout=60, data={
                "q": ",".join(chunk), "scopes": "symbol,alias",
                "species": cfg.species,
                "fields": "symbol,name,summary,go.MF,go.BP"})
            r.raise_for_status()
            for hit in r.json():
                sym = hit.get("symbol")
                if not sym or hit.get("notfound"):
                    continue
                go = hit.get("go", {}) or {}
                terms = []
                for k in ("MF", "BP"):
                    v = go.get(k) or []
                    v = v if isinstance(v, list) else [v]
                    terms += [t.get("term") for t in v if isinstance(t, dict)][:4]
                cand = {"name": hit.get("name", ""),
                        "summary": (hit.get("summary") or "")[:m.max_summary_chars],
                        "go": terms[:m.max_go_terms]}
                prev = out.get(sym.upper())
                if not prev or len(cand["summary"]) > len(prev["summary"]):
                    out[sym.upper()] = cand
        except Exception as e:
            print(f"  [mygene] batch {i} failed: {e}")
        print(f"  [mygene] {min(i + m.batch_size, len(symbols))}/{len(symbols)}",
              end="\r")
        time.sleep(m.sleep)
    print()
    return out


# ── CollecTRI: SIGNED TF->target regulons ──────────────────────────────────
def fetch_collectri(cfg):
    if not cfg.sources.collectri.enabled:
        return {}, set()
    try:
        import decoupler as dc
    except ImportError:
        print("  [collectri] decoupler not installed — pip install decoupler")
        return {}, set()
    net = None
    for fn in (lambda: dc.op.collectri(organism=cfg.species),
               lambda: dc.get_collectri(organism=cfg.species,
                                        split_complexes=False)):
        try:
            net = fn()
            break
        except Exception as e:
            print(f"  [collectri] {type(e).__name__}: {e}")
    if net is None:
        return {}, set()
    cols = {c.lower(): c for c in net.columns}
    src, tgt = cols.get("source"), cols.get("target")
    mor = cols.get("mor") or cols.get("weight")
    edges, tfs = {}, set()
    for r in net.itertuples(index=False):
        s = str(getattr(r, src)).upper()
        t = str(getattr(r, tgt)).upper()
        edges[(s, t)] = float(getattr(r, mor)) if mor else 1.0
        tfs.add(s)
    print(f"  [collectri] {len(edges)} signed edges, {len(tfs)} TFs")
    return edges, tfs


# ── per-row features + teacher context ─────────────────────────────────────
def edge_features(pert, gene, edges, tfs, targets_of):
    P, G = pert.upper(), gene.upper()
    mor = edges.get((P, G))
    return {
        "pert_is_tf": int(P in tfs),
        "has_edge": int(mor is not None),
        "mor": float(mor) if mor is not None else 0.0,
        # CRISPRi knocks the TF DOWN, so the target moves OPPOSITE to mor:
        #   activator (mor>0) knocked down -> target down -> letter B
        #   repressor (mor<0) knocked down -> target up   -> letter A
        "expected_letter": (("B" if mor > 0 else "A") if mor is not None else None),
        "shared_regulators": len(targets_of.get(P, set())
                                 & targets_of.get(G, set())),
    }


def context_block(pert, gene, ann, f, cfg):
    c = cfg.context

    def desc(sym):
        a = ann.get(sym.upper())
        if not a:
            return f"{sym}: no annotation found (poorly characterized gene)"
        bits = [f"{sym} = {a['name']}"]
        if c.include_summary and a["summary"]:
            bits.append(f"Function: {a['summary']}")
        if c.include_go and a["go"]:
            bits.append(f"GO: {'; '.join(a['go'])}")
        return " | ".join(bits)

    lines = [f"PERTURBED: {desc(pert)}", f"TARGET:    {desc(gene)}"]
    if c.include_edge:
        if f["has_edge"]:
            kind = "ACTIVATOR of" if f["mor"] > 0 else "REPRESSOR of"
            implies = "down" if f["mor"] > 0 else "up"
            lines.append(f"KNOWN REGULATORY EDGE: {pert} is a curated {kind} "
                         f"{gene}. CRISPRi knockdown of {pert} therefore pushes "
                         f"{gene} {implies}.")
        elif f["pert_is_tf"]:
            lines.append(f"REGULATORY EDGE: {pert} is a known TF but no curated "
                         f"{pert}->{gene} edge exists.")
        else:
            lines.append(f"REGULATORY EDGE: none curated; {pert} is not a known TF.")
    if c.include_shared_regulators and f["shared_regulators"]:
        lines.append(f"Shared upstream regulators: {f['shared_regulators']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="grounding run name")
    cfgmod.add_config_args(ap, "grounding")
    args = ap.parse_args()
    cfg = cfgmod.resolve(args, "grounding")

    out_dir = paths.run_dir("grounding", args.out, create=True)
    train, test = common.load_data()
    symbols = sorted(set(train["perturb_gene"]) | set(train["target_gene"])
                     | set(test["perturb_gene"]) | set(test["target_gene"]))

    print(f"Annotating {len(symbols)} unique symbols ({cfg.species}) ...")
    ann = fetch_annotations(symbols, cfg)
    cov_ann = 100 * len(ann) / max(1, len(symbols))
    print(f"  annotated {len(ann)}/{len(symbols)} ({cov_ann:.1f}%)")
    if cov_ann < 60:
        print("  NOTE low coverage -> the 'lantern effect' will bite: the teacher "
              "has little to reason from on unannotated genes and may confabulate.")

    print("Fetching CollecTRI signed regulons ...")
    edges, tfs = fetch_collectri(cfg)
    targets_of = {}
    for (s, t) in edges:
        targets_of.setdefault(t, set()).add(s)

    rows = {}
    for df in (train, test):
        for r in df.itertuples(index=False):
            f = edge_features(r.perturb_gene, r.target_gene, edges, tfs, targets_of)
            rows[f"{r.perturb_gene}_{r.target_gene}"] = {
                "pert": r.perturb_gene, "gene": r.target_gene, "features": f,
                "context": context_block(r.perturb_gene, r.target_gene, ann, f, cfg)}

    cov_edge = sum(v["features"]["has_edge"] for v in rows.values())
    (out_dir / "grounding.json").write_text(json.dumps(
        {"annotations": ann, "rows": rows,
         "stats": {"n_symbols": len(symbols), "n_annotated": len(ann),
                   "n_edges": len(edges), "n_tfs": len(tfs),
                   "n_rows": len(rows), "n_rows_with_edge": cov_edge}}, indent=1))
    cfgmod.snapshot(cfg, out_dir, {"annotation_coverage_pct": round(cov_ann, 1),
                                   "edge_coverage_pct": round(100 * cov_edge / len(rows), 1)})

    print(f"\nWrote {out_dir/'grounding.json'}")
    print(f"  rows={len(rows)}  with a curated signed edge: {cov_edge} "
          f"({100*cov_edge/len(rows):.1f}%)")
    print("  Edge coverage is low BY DESIGN (CollecTRI is high-precision, "
          "low-recall). train_grpo.py validates it before using it as a reward.")
    print("\n--- example context block ---")
    print(next(iter(rows.values()))["context"][:700])


if __name__ == "__main__":
    main()
