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
import sys
import time

import requests

import common
import config as cfgmod
import paths

MYGENE_URL = "https://mygene.info/v3/query"


# ── mygene.info: gene name + RefSeq summary (function) + GO ─────────────────
def fetch_annotations(symbols, cfg):
    """QUERIED symbol -> {name, summary, go}.

    Keyed by the QUERY, not the hit's symbol. With scopes=symbol,alias a query
    for 'Foo' can return gene 'Bar' (Foo being an alias of Bar). Keying by the
    hit symbol both invents entries we never asked for (coverage >100%) and
    leaves the queried symbol unfindable at lookup time. mygene returns a
    'query' field precisely for this.
    """
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
                q = hit.get("query")
                if not q or hit.get("notfound"):
                    continue
                go = hit.get("go", {}) or {}
                terms = []
                for k in ("MF", "BP"):
                    v = go.get(k) or []
                    v = v if isinstance(v, list) else [v]
                    for t in v:
                        if isinstance(t, dict) and t.get("term"):
                            terms.append(t["term"])
                # dedupe, preserve order (mygene repeats terms across evidence codes)
                terms = list(dict.fromkeys(terms))[:m.max_go_terms]
                cand = {"symbol": hit.get("symbol", q),
                        "name": hit.get("name", ""),
                        "summary": (hit.get("summary") or "")[:m.max_summary_chars],
                        "go": terms}
                prev = out.get(q.upper())
                # mygene can return several hits per query; keep the richest
                if not prev or len(cand["summary"]) > len(prev["summary"]):
                    out[q.upper()] = cand
        except Exception as e:
            print(f"  [mygene] batch {i} failed: {e}")
        print(f"  [mygene] {min(i + m.batch_size, len(symbols))}/{len(symbols)}",
              end="\r")
        time.sleep(m.sleep)
    print()
    return out


# ── CollecTRI: SIGNED TF->target regulons ──────────────────────────────────
# Fetched over plain HTTP from OmniPath's REST API. We deliberately do NOT
# depend on `decoupler` for this: decoupler pulls in numba (and a scanpy-ish
# stack) and its resolver backtracks to numba 0.53.1, which has no Python 3.12
# wheel and fails to build ("only versions >=3.6,<3.10 are supported"). We use
# decoupler for exactly one thing — downloading this table — so we just download
# it. Set sources.collectri.method: decoupler to use the library if you have it.

OMNIPATH_URL = "https://omnipathdb.org/interactions"
# Static CollecTRI dump on a DIFFERENT host from the REST API. When
# omnipathdb.org/interactions returns 502 (which it does), this still serves.
# Schema: source,target,weight,TF.category,resources,PMID,sign.decision
# `weight` is already +1/-1 — the sign, directly, with no is_stimulation /
# is_inhibition reconstruction needed. Human symbols (CollecTRI is human-native).
COLLECTRI_STATIC_URL = "https://rescued.omnipathdb.org/CollecTRI.csv"
NCBI_TAXID = {"mouse": 10090, "human": 9606, "rat": 10116}

# CollecTRI names some TF complexes as single tokens rather than real gene
# symbols. Left alone they simply never match a perturbed gene and silently cost
# coverage. NFKB and AP1 matter a lot here: they are THE macrophage inflammatory
# TFs, so their edges are exactly the ones this task cares about.
COMPLEX_MEMBERS = {
    "NFKB":  ["NFKB1", "NFKB2", "RELA", "RELB", "REL"],
    "AP1":   ["JUN", "JUNB", "JUND", "FOS", "FOSB", "FOSL1", "FOSL2"],
    "NFY":   ["NFYA", "NFYB", "NFYC"],
    "SMAD":  ["SMAD2", "SMAD3", "SMAD4"],
}


def _edges_from_records(records):
    """rows with (source, target, is_stimulation, is_inhibition) -> signed edges.

    mor = +1 activation / -1 repression.

    Two cleanups, both mirroring what decoupler does:

    1. AMBIGUOUS SIGNS are dropped. Rows that are both stimulation and
       inhibition (or neither) carry no usable direction, and direction is the
       entire reason we want CollecTRI. An ambiguous edge is worse than no edge.

    2. COMPLEXES ARE SPLIT. CollecTRI sources include protein complexes written
       as underscore-joined symbols ("FOS_JUN", "FOSL1_JUNB"). Those never match
       a single perturbed gene symbol, so left intact they would silently drop
       coverage. We attribute the edge to each member, like decoupler's
       split_complexes=True.
    """
    edges, tfs, ambiguous, complexes = {}, set(), 0, 0
    for s, t, stim, inhib in records:
        s, t = str(s).upper().strip(), str(t).upper().strip()
        if not s or not t:
            continue
        if bool(stim) == bool(inhib):        # 0/0 or 1/1 -> no usable sign
            ambiguous += 1
            continue
        mor = 1.0 if stim else -1.0
        members = COMPLEX_MEMBERS.get(s)
        if members is None:
            members = s.split("_") if "_" in s else [s]
        if len(members) > 1:
            complexes += 1
        for member in members:
            if not member:
                continue
            # A single-TF edge beats a complex-derived one on conflict.
            if (member, t) in edges and len(members) > 1:
                continue
            edges[(member, t)] = mor
            tfs.add(member)
    return edges, tfs, ambiguous, complexes


def _http_get(url, cache_path, params=None, timeout=300, tries=4):
    label = cache_path.name if cache_path else url
    if cache_path and cache_path.exists() and cache_path.stat().st_size > 1000:
        print(f"  [collectri] using cached {cache_path.name}")
        return cache_path.read_text()
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(r.text)
            return r.text
        except Exception as e:
            wait = 5 * (2 ** attempt)
            if attempt == tries - 1:
                print(f"  [collectri] giving up on {label} after {tries} tries: "
                      f"{str(e)[:120]}")
                return None
            print(f"  [collectri] {label}: {str(e)[:80]} — retry "
                  f"{attempt+1}/{tries-1} in {wait}s")
            time.sleep(wait)
    return None


def _omnipath_get(params, cache_path, tries=4):
    """GET with backoff. 502/503/504 from OmniPath are common and transient —
    the mouse query triggers server-side orthology translation, which is the
    slow path and the one that gateways out. Cache the raw TSV so a rerun (or a
    later stage) never re-fetches."""
    return _http_get(OMNIPATH_URL, cache_path, params=params, timeout=300,
                     tries=tries)


def _parse_omnipath_tsv(text):
    import csv
    import io
    rd = csv.DictReader(io.StringIO(text), delimiter="\t")
    cols = rd.fieldnames or []
    src = "source_genesymbol" if "source_genesymbol" in cols else "source"
    tgt = "target_genesymbol" if "target_genesymbol" in cols else "target"
    if src not in cols or "is_stimulation" not in cols:
        print(f"  [collectri] unexpected columns: {cols[:8]}")
        return None
    return [(row[src], row[tgt], row["is_stimulation"] == "1",
             row["is_inhibition"] == "1") for row in rd]


def _parse_collectri_static_csv(text):
    import csv
    import io
    rd = csv.DictReader(io.StringIO(text))
    cols = rd.fieldnames or []
    needed = {"source", "target", "weight"}
    if not needed.issubset(cols):
        print(f"  [collectri] unexpected static mirror columns: {cols[:8]}")
        return None
    recs = []
    for row in rd:
        try:
            mor = float(row["weight"])
        except (TypeError, ValueError):
            continue
        if mor == 0:
            recs.append((row["source"], row["target"], False, False))
        else:
            recs.append((row["source"], row["target"], mor > 0, mor < 0))
    return recs


def fetch_collectri_static(cfg, cache_dir=None):
    c = cfg.sources.collectri
    if not c.get("static_mirror", True):
        return {}, set()
    cache = (cache_dir / "collectri_static.csv") if cache_dir else None
    print("  [collectri] querying static CollecTRI mirror ...")
    text = _http_get(COLLECTRI_STATIC_URL, cache, timeout=180,
                     tries=c.get("static_mirror_tries", 2))
    if not text:
        return {}, set()
    recs = _parse_collectri_static_csv(text)
    if not recs:
        return {}, set()
    edges, tfs, amb, cplx = _edges_from_records(recs)
    if edges:
        print(f"  [collectri] static mirror: {len(edges)} signed edges, "
              f"{len(tfs)} TFs ({amb} ambiguous dropped, {cplx} complexes split)")
        if cfg.species != "human":
            print("  [collectri] NOTE static CollecTRI is human-native; using "
                  "symbol matching for mouse genes.")
    return edges, tfs


def diagnose_omnipath(cfg):
    """Distinguish 'our request is wrong' from 'the server is slow/unwell'.

    A 502 is a SERVER-side failure: the request reached OmniPath and the backend
    failed or timed out behind the proxy. A malformed request would be a 4xx
    (400 for bad params, 404 for a bad endpoint), not a 502 — so a 502 mostly
    rules out 'we asked wrong'. But it does NOT tell you WHY the backend
    struggled. These probes do:

        human tiny   -> is the service up at all?
        mouse tiny   -> is the mouse path itself broken, or just slow at volume?
        mouse full   -> is the failure specific to the full-size translation?

    Read the table:
      all OK ........................ it was transient; just rerun
      human OK, mouse tiny FAILS .... the mouse/orthology path is genuinely broken
      mouse tiny OK, full FAILS ..... timeout on volume -> retry, or use --limit
      everything FAILS .............. the service is down; not your request
    """
    taxid = NCBI_TAXID.get(cfg.species, 10090)
    base = {"datasets": "collectri", "genesymbols": "yes", "format": "tsv"}
    probes = [
        ("human, limit=10", {**base, "organisms": "9606", "limit": "10"}),
        (f"{cfg.species}, limit=10", {**base, "organisms": str(taxid), "limit": "10"}),
        (f"{cfg.species}, full", {**base, "organisms": str(taxid), "fields": "sources"}),
    ]
    print("Probing OmniPath (a 502 = server-side; a 4xx = our request is wrong)\n")
    results = []
    for label, params in probes:
        t0 = time.time()
        try:
            r = requests.get(OMNIPATH_URL, params=params, timeout=120)
            dt = time.time() - t0
            n = len(r.text.splitlines()) - 1 if r.ok else 0
            print(f"  {label:22} HTTP {r.status_code}  {dt:6.1f}s  rows={n}")
            if not r.ok:
                print(f"    body: {r.text[:150]!r}")
            results.append((label, r.status_code, dt, n))
        except Exception as e:
            dt = time.time() - t0
            print(f"  {label:22} EXC after {dt:.1f}s: {type(e).__name__}: {str(e)[:90]}")
            results.append((label, None, dt, 0))

    ok = {lbl: (code == 200 and rows > 0) for lbl, code, _, rows in results}
    codes = [code for _, code, _, _ in results]
    client_err = [c for c in codes if c is not None and 400 <= c < 500]
    server_err = [c for c in codes if c is not None and c >= 500]

    print("\nVerdict:")
    human_ok = ok.get("human, limit=10")
    mouse_small_ok = ok.get(f"{cfg.species}, limit=10")
    mouse_full_ok = ok.get(f"{cfg.species}, full")

    # Check the ERROR CLASS before anything else: 4xx means WE asked wrong, 5xx
    # means the server failed. Conflating them sends you debugging the wrong
    # system.
    if client_err and not any(ok.values()):
        print(f"  4xx on every probe {client_err} -> OUR REQUEST IS MALFORMED, not "
              f"a server problem. The endpoint/params are wrong (bad `datasets` "
              f"value, bad field name, renamed API). Read the response body above "
              f"and check https://omnipathdb.org/queries/interactions")
    elif client_err:
        print(f"  Some probes returned 4xx {client_err} -> those specific requests "
              f"are malformed (likely the organism or a field name), while others "
              f"work. Compare the params of the failing vs passing probe above.")
    elif all(ok.values()):
        print("  Everything works — the earlier failure was transient. Rerun "
              "`build_grounding.py`; the raw TSV caches on success.")
    elif not human_ok:
        print(f"  The smallest human query fails with {server_err or 'a timeout'} "
              f"(5xx/timeout, NOT 4xx) -> the OmniPath service is down or "
              f"unreachable from this node. Not your request. Check egress "
              f"(curl -I https://omnipathdb.org) and try later.")
    elif human_ok and not mouse_small_ok:
        print(f"  Human works; {cfg.species} fails server-side even at limit=10 -> "
              f"the orthology-translation path is genuinely broken right now (not "
              f"a volume/timeout issue, since 10 rows is trivial). Use "
              f"--set sources.collectri.fallback_to_human=true (symbol matching).")
    elif mouse_small_ok and not mouse_full_ok:
        print(f"  Small {cfg.species} query works, the full one does not -> a "
              f"timeout on volume/translation. This is the ONLY scenario that "
              f"confirms the 'translation is the slow path' hypothesis. Retry "
              f"with backoff (the fetcher does), or fall back to human.")
    else:
        print("  Mixed result — rerun this probe before concluding anything.")
    return results


def fetch_collectri_http(cfg, cache_dir=None):
    """OmniPath REST — no numba, no decoupler, no build step.

    Falls back from mouse to human. That is not a hack: OmniPath is built from
    human data and orthology-translates on request, so the mouse query is the
    expensive server-side path that 502s. Since every symbol in this pipeline is
    upper-cased before lookup, human symbols match mouse ones for the large
    majority of 1:1 orthologs (MYC/Myc, TP53/Trp53 being a known exception) —
    i.e. roughly the same homology mapping, done client-side.
    """
    c = cfg.sources.collectri
    if c.get("prefer_static_mirror", True):
        edges, tfs = fetch_collectri_static(cfg, cache_dir)
        if edges:
            return edges, tfs

    taxid = NCBI_TAXID.get(cfg.species)
    base = {"datasets": "collectri", "genesymbols": "yes",
            "fields": "sources", "format": "tsv"}

    attempts = [(cfg.species, taxid)]
    if c.get("fallback_to_human", True) and cfg.species != "human":
        attempts.append(("human (ortholog-by-symbol fallback)", 9606))

    for label, tid in attempts:
        if tid is None:
            continue
        print(f"  [collectri] querying OmniPath for {label} (taxid {tid}) ...")
        cache = (cache_dir / f"collectri_{tid}.tsv") if cache_dir else None
        text = _omnipath_get({**base, "organisms": str(tid)}, cache)
        if not text:
            continue
        recs = _parse_omnipath_tsv(text)
        if not recs:
            continue
        edges, tfs, amb, cplx = _edges_from_records(recs)
        if edges:
            print(f"  [collectri] {len(edges)} signed edges, {len(tfs)} TFs "
                  f"({amb} ambiguous dropped, {cplx} complexes split)")
            if tid == 9606 and cfg.species != "human":
                print("  [collectri] NOTE using human edges matched by symbol. "
                      "Record this in the paper: it is homology-by-symbol, not "
                      "curated mouse orthology.")
            return edges, tfs
    return fetch_collectri_static(cfg, cache_dir)


def fetch_collectri_decoupler(cfg, cache_dir=None):
    """Optional path if decoupler is installed (pixi: -e grounding-decoupler)."""
    try:
        import decoupler as dc
    except ImportError:
        print("  [collectri] decoupler not installed — falling back to HTTP")
        return fetch_collectri_http(cfg, cache_dir)
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
        return fetch_collectri_http(cfg, cache_dir)
    cols = {c.lower(): c for c in net.columns}
    s_c, t_c = cols.get("source"), cols.get("target")
    m_c = cols.get("mor") or cols.get("weight")
    edges, tfs = {}, set()
    for r in net.itertuples(index=False):
        s = str(getattr(r, s_c)).upper()
        t = str(getattr(r, t_c)).upper()
        edges[(s, t)] = float(getattr(r, m_c)) if m_c else 1.0
        tfs.add(s)
    print(f"  [collectri] decoupler: {len(edges)} signed edges, {len(tfs)} TFs")
    return edges, tfs


def fetch_collectri(cfg, cache_dir=None):
    c = cfg.sources.collectri
    if not c.enabled:
        return {}, set()
    method = c.get("method", "http")
    return (fetch_collectri_decoupler(cfg, cache_dir) if method == "decoupler"
            else fetch_collectri_http(cfg, cache_dir))


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
        # If the query resolved through an alias, name the canonical gene — the
        # teacher reasons better about 'Map3k8/Tpl2' than about 'Tpl2' alone.
        canonical = a.get("symbol", sym)
        head = f"{sym} = {a['name']}" if canonical.upper() == sym.upper() \
            else f"{sym} (canonical: {canonical}) = {a['name']}"
        bits = [head]
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
    ap.add_argument("--diagnose", action="store_true",
                    help="probe OmniPath to tell a bad request (4xx) apart from a "
                         "slow/unwell server (502), then exit")
    ap.add_argument("--allow-no-edges", action="store_true",
                    help="write grounding.json even if CollecTRI returned nothing "
                         "(you lose the direction signal — see the fatal message)")
    cfgmod.add_config_args(ap, "grounding")
    args = ap.parse_args()
    cfg = cfgmod.resolve(args, "grounding")

    if args.diagnose:
        diagnose_omnipath(cfg)
        return

    out_dir = paths.run_dir("grounding", args.out, create=True)
    train, test = common.load_data()
    symbols = sorted(set(train["perturb_gene"]) | set(train["target_gene"])
                     | set(test["perturb_gene"]) | set(test["target_gene"]))

    print(f"Annotating {len(symbols)} unique symbols ({cfg.species}) ...")
    ann = fetch_annotations(symbols, cfg)
    # Coverage against the symbols we ACTUALLY asked for. (Keying by hit symbol
    # instead of query symbol previously produced >100% here — a real bug, since
    # aliases resolve to other genes and queried symbols went missing.)
    hit = {s for s in symbols if s.upper() in ann}
    cov_ann = 100 * len(hit) / max(1, len(symbols))
    print(f"  annotated {len(hit)}/{len(symbols)} ({cov_ann:.1f}%) of queried symbols")
    with_summary = sum(1 for s in hit if ann[s.upper()]["summary"])
    print(f"  with a RefSeq summary (the part that generalises): {with_summary} "
          f"({100*with_summary/max(1,len(symbols)):.1f}%)")
    if cov_ann < 60:
        print("  NOTE low coverage -> the 'lantern effect' will bite: the teacher "
              "has little to reason from on unannotated genes and may confabulate.")

    print("Fetching CollecTRI signed regulons ...")
    edges, tfs = fetch_collectri(cfg, cache_dir=out_dir / "cache")
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

    # FAIL LOUD on zero edges. Writing a grounding.json with no CollecTRI edges
    # silently deletes the single highest-value signal (direction), and you would
    # only find out much later when the verifier validated at chance.
    if cfg.sources.collectri.enabled and not edges and not args.allow_no_edges:
        sys.exit(
            "\nFATAL: CollecTRI returned 0 edges — refusing to write a grounding "
            "file that silently drops the direction signal.\n"
            "  OmniPath 502/503 is usually transient: just rerun (the raw TSV is "
            "cached once it succeeds).\n"
            "  Options:\n"
            "    - rerun in a few minutes\n"
            "    - --set sources.collectri.fallback_to_human=true (default; human "
            "edges matched by symbol)\n"
            "    - pixi run -e grounding-decoupler grounding --set "
            "sources.collectri.method=decoupler\n"
            "    - --allow-no-edges  (proceed WITHOUT direction grounding; the "
            "GRPO verifier will be disabled and teacher prompts lose the edge "
            "statement)\n"
            "    - --set sources.collectri.enabled=false  (explicit ablation)")

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