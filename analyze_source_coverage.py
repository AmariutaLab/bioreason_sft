"""Summarize how much useful grounding exists for a trace set.

This is intentionally read-only and offline. It answers whether a proposed
source is likely to help before we spend API or GPU budget integrating it into
teacher prompts.

Examples:
    python analyze_source_coverage.py \
      --trace-file /home/i3gupta/lustre/tools/mlgenx/output/traces/norules-ungrounded-o4mini/traces.jsonl \
      --out source_coverage_original_o4mini

    python analyze_source_coverage.py \
      --trace-run norules-o4mini-signature-selected-none-full \
      --gmt reactome=/path/to/reactome.gmt \
      --gmt msigdb_mouse=/path/to/mouse_hallmark.gmt \
      --out source_coverage_with_gmts

    python analyze_source_coverage.py \
      --csv-file /path/to/train.csv \
      --out source_coverage_train
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import paths


GENERIC_GO = {
    "biological_process",
    "molecular_function",
    "cellular_component",
    "protein binding",
    "identical protein binding",
    "catalytic activity",
    "binding",
    "enzyme binding",
    "metal ion binding",
    "atp binding",
}


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


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def trace_path(args) -> Path:
    if args.trace_file:
        return paths.require(Path(args.trace_file), "expected trace JSONL")
    if args.trace_run:
        return paths.require(paths.traces_jsonl(args.trace_run),
                             f"missing trace run {args.trace_run}")
    raise SystemExit("provide --trace-run or --trace-file")


def input_path(args) -> Path:
    if args.csv_file:
        return paths.require(Path(args.csv_file), "expected CSV with id,pert,gene")
    return trace_path(args)


def read_rows(args) -> list[dict]:
    if not args.csv_file:
        return read_jsonl(trace_path(args))
    rows = []
    with paths.require(Path(args.csv_file), "expected CSV").open(newline="") as fh:
        for rec in csv.DictReader(fh):
            pert = rec.get("pert") or rec.get("perturb_gene")
            gene = rec.get("gene") or rec.get("target_gene")
            if not pert or not gene:
                continue
            rows.append({
                "id": rec.get("id") or f"{pert}_{gene}",
                "pert": pert.strip(),
                "gene": gene.strip(),
                "label": (rec.get("label") or "unknown").strip() or "unknown",
            })
    return rows


def useful_go_terms(ann: dict) -> list[str]:
    terms = []
    for term in ann.get("go", []) or []:
        t = str(term).strip()
        if t and t.lower() not in GENERIC_GO:
            terms.append(t)
    return terms


def has_annotation(ann: dict) -> bool:
    return bool(ann.get("name") or ann.get("summary") or useful_go_terms(ann))


def gene_text(sym: str, annotations: dict) -> str:
    ann = annotations.get(sym.upper(), {})
    return " ".join([
        sym,
        str(ann.get("symbol", "")),
        str(ann.get("name", "")),
        str(ann.get("summary", "")),
        " ".join(ann.get("go", []) or []),
    ])


def gene_modules(sym: str, annotations: dict) -> list[str]:
    text = gene_text(sym, annotations)
    mods = [name for name, pat in MODULE_PATTERNS if re.search(pat, text, re.I)]
    return mods


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({(100.0 * n / d if d else 0.0):.1f}%)"


def summarize_bool(rows: list[dict], key: str) -> dict:
    out = {"all": sum(1 for r in rows if r[key])}
    for label in sorted({r["label"] for r in rows}):
        sub = [r for r in rows if r["label"] == label]
        out[label] = sum(1 for r in sub if r[key])
    return out


def parse_named_path(raw: str) -> tuple[str, Path]:
    name, sep, path = raw.partition("=")
    if not sep:
        p = Path(name)
        return p.stem, p
    return name.strip(), Path(path)


def read_gmt(path: Path) -> dict[str, set[str]]:
    gene_to_sets = defaultdict(set)
    with path.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            set_name = parts[0]
            for gene in parts[2:]:
                g = gene.strip().upper()
                if g:
                    gene_to_sets[g].add(set_name)
    return dict(gene_to_sets)


def gmt_coverage(name: str, path: Path, rows: list[dict]) -> dict:
    gene_to_sets = read_gmt(paths.require(path, f"missing GMT for {name}"))
    stats = Counter()
    by_label = defaultdict(Counter)
    for row in rows:
        p_sets = gene_to_sets.get(row["pert"].upper(), set())
        g_sets = gene_to_sets.get(row["gene"].upper(), set())
        vals = {
            "pert_has_set": bool(p_sets),
            "target_has_set": bool(g_sets),
            "both_have_sets": bool(p_sets and g_sets),
            "same_set": bool(p_sets & g_sets),
            "separated_sets": bool(p_sets and g_sets and not (p_sets & g_sets)),
        }
        for k, v in vals.items():
            if v:
                stats[k] += 1
                by_label[row["label"]][k] += 1
    return {
        "name": name,
        "path": str(path),
        "genes_with_sets": len(gene_to_sets),
        "stats": dict(stats),
        "by_label": {k: dict(v) for k, v in by_label.items()},
    }


def load_grounding(run: str) -> dict:
    path = paths.grounding_json(run)
    if not path.exists():
        return {"annotations": {}, "rows": {}}
    return json.loads(path.read_text())


def row_coverage(rows: list[dict], grounding: dict) -> tuple[list[dict], dict]:
    anns = grounding.get("annotations", {})
    grounded_rows = grounding.get("rows", {})
    covered = []
    module_counts = Counter()
    for row in rows:
        rid = row["id"]
        pert, gene = row["pert"], row["gene"]
        p_ann = anns.get(pert.upper(), {})
        g_ann = anns.get(gene.upper(), {})
        p_mods = set(gene_modules(pert, anns))
        g_mods = set(gene_modules(gene, anns))
        feats = (grounded_rows.get(rid) or {}).get("features", {})
        rec = dict(row)
        rec.update({
            "pert_annotated": has_annotation(p_ann),
            "target_annotated": has_annotation(g_ann),
            "both_annotated": has_annotation(p_ann) and has_annotation(g_ann),
            "pert_summary": bool(p_ann.get("summary")),
            "target_summary": bool(g_ann.get("summary")),
            "pert_specific_go": bool(useful_go_terms(p_ann)),
            "target_specific_go": bool(useful_go_terms(g_ann)),
            "pert_modules": sorted(p_mods),
            "target_modules": sorted(g_mods),
            "both_modules": bool(p_mods and g_mods),
            "shared_module": bool(p_mods & g_mods),
            "module_separation": bool(p_mods and g_mods and not (p_mods & g_mods)),
            "direct_signed_edge": bool(feats.get("has_edge") and feats.get("mor")),
            "pert_is_tf": bool(feats.get("pert_is_tf")),
            "shared_regulators": bool(feats.get("shared_regulators")),
            "target_has_upstream_tfs": bool(feats.get("target_upstream_tfs")),
            "pert_has_regulon": bool(feats.get("pert_regulon_genes")),
        })
        for m in p_mods | g_mods:
            module_counts[m] += 1
        covered.append(rec)
    return covered, dict(module_counts.most_common())


def markdown_report(trace: Path, rows: list[dict], covered: list[dict],
                    module_counts: dict, gmt_reports: list[dict]) -> str:
    n = len(rows)
    labels = Counter(r["label"] for r in rows)
    bool_keys = [
        ("both_annotated", "Both genes have local annotation"),
        ("pert_summary", "Perturbed gene has RefSeq/mygene summary"),
        ("target_summary", "Target gene has RefSeq/mygene summary"),
        ("pert_specific_go", "Perturbed gene has specific GO"),
        ("target_specific_go", "Target gene has specific GO"),
        ("both_modules", "Both genes map to heuristic modules"),
        ("shared_module", "Pert/target share a heuristic module"),
        ("module_separation", "Both mapped but module-separated"),
        ("direct_signed_edge", "Direct signed CollecTRI edge"),
        ("pert_is_tf", "Perturbed gene is TF in CollecTRI"),
        ("pert_has_regulon", "Perturbed gene has CollecTRI regulon members"),
        ("target_has_upstream_tfs", "Target has upstream task TFs"),
        ("shared_regulators", "Pert/target share upstream regulators"),
    ]
    label_cols = [x for x in ("down", "none", "up") if x in labels]
    label_cols += [x for x in sorted(labels) if x not in {"down", "none", "up"}]
    header = "| Signal | All | " + " | ".join(label_cols) + " |"
    sep = "|---|---:|" + "|".join("---:" for _ in label_cols) + "|"
    lines = [
        "# Source coverage diagnostic",
        "",
        f"Input file: `{trace}`",
        f"Rows: {n}",
        f"Labels: {dict(labels)}",
        "",
        "## Local grounding coverage",
        "",
        header,
        sep,
    ]
    for key, label in bool_keys:
        vals = summarize_bool(covered, key)
        row_vals = [pct(vals.get("all", 0), n)]
        row_vals.extend(pct(vals.get(col, 0), labels.get(col, 0))
                        for col in label_cols)
        lines.append(f"| {label} | " + " | ".join(row_vals) + " |")

    lines += ["", "## Heuristic module frequency", ""]
    for name, count in list(module_counts.items())[:15]:
        lines.append(f"- {name}: {count} rows")
    if not module_counts:
        lines.append("- No heuristic modules found.")

    if gmt_reports:
        lines += ["", "## Optional GMT source coverage", ""]
        for rep in gmt_reports:
            stats = rep["stats"]
            lines.append(f"### {rep['name']}")
            lines.append("")
            lines.append(f"Path: `{rep['path']}`")
            lines.append(f"Genes with sets: {rep['genes_with_sets']}")
            lines.append(f"- pert has set: {pct(stats.get('pert_has_set', 0), n)}")
            lines.append(f"- target has set: {pct(stats.get('target_has_set', 0), n)}")
            lines.append(f"- both have sets: {pct(stats.get('both_have_sets', 0), n)}")
            lines.append(f"- same set: {pct(stats.get('same_set', 0), n)}")
            lines.append(f"- separated sets: {pct(stats.get('separated_sets', 0), n)}")
            lines.append("")

    lines += [
        "",
        "## Interpretation guide",
        "",
        "- Direct signed edges are high precision but usually too sparse to drive the full method.",
        "- Module separation is most useful for `none` rationales.",
        "- Shared modules or direct/regulon edges are more useful for directional `up/down` traces, but still need a sign-producing mechanism.",
        "- Add a new grounding source only if it increases row-level coverage or gives a more specific module/sign than existing local grounding.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-run", default=None)
    ap.add_argument("--trace-file", default=None)
    ap.add_argument("--csv-file", default=None,
                    help="CSV with id,pert,gene and optional label columns")
    ap.add_argument("--grounding-run", default="default")
    ap.add_argument("--gmt", action="append", default=[],
                    help="optional NAME=/path/source.gmt for source coverage")
    ap.add_argument("--out", default="source_coverage",
                    help="output/notes/<out>.md and <out>.json")
    args = ap.parse_args()

    tpath = input_path(args)
    rows = read_rows(args)
    grounding = load_grounding(args.grounding_run)
    covered, module_counts = row_coverage(rows, grounding)
    gmt_reports = [gmt_coverage(*parse_named_path(raw), rows) for raw in args.gmt]

    report = markdown_report(tpath, rows, covered, module_counts, gmt_reports)
    out_md = paths.OUTPUT / "notes" / f"{args.out}.md"
    out_json = paths.OUTPUT / "notes" / f"{args.out}.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report)
    out_json.write_text(json.dumps({
        "input_file": str(tpath),
        "grounding_run": args.grounding_run,
        "rows": len(rows),
        "labels": Counter(r["label"] for r in rows),
        "module_counts": module_counts,
        "gmt_reports": gmt_reports,
    }, indent=2))
    print(report)
    print(f"\nWrote {out_md}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
