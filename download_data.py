"""Download the competition data into data/.

Kaggle has TWO credential styles. Both work here:

  NEW (token)      kaggle.com/settings/api -> "Generate New Token"
                     export KAGGLE_API_TOKEN=<token>
                   or  ~/.kaggle/access_token   (chmod 600)

  LEGACY (json)    same page -> "Legacy API Credentials" -> "Create Legacy API Key"
                     ~/.kaggle/kaggle.json      (chmod 600)
                   or  export KAGGLE_USERNAME=... KAGGLE_KEY=...

You must ALSO have accepted the competition rules on the website with the same
account, or every method returns 403.

Usage:
    python download_data.py
    python download_data.py --force
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import paths

KAGGLE_DIR = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
API_BASE = "https://www.kaggle.com/api/v1"


# ── credential discovery ────────────────────────────────────────────────────
def load_credentials() -> dict:
    """Find whatever the user has and normalise it into env vars.

    Returns {"kind": "token"|"legacy"|None, ...}. Also EXPORTS the env vars the
    downstream tools (kagglehub / kaggle CLI) look for, so a file-only setup
    works without the user having exported anything.
    """
    # 1. new-style token in the environment
    token = os.environ.get("KAGGLE_API_TOKEN")
    source = "KAGGLE_API_TOKEN env" if token else None

    # 2. new-style token on disk
    tok_file = KAGGLE_DIR / "access_token"
    if not token and tok_file.exists():
        raw = tok_file.read_text().strip()
        # People sometimes paste the whole kaggle.json into access_token.
        # Detect that and treat it as legacy creds instead of a bearer token.
        if raw.startswith("{"):
            try:
                j = json.loads(raw)
                if "username" in j and "key" in j:
                    print(f"[auth] {tok_file} holds legacy JSON creds — using "
                          f"them as username/key")
                    os.environ["KAGGLE_USERNAME"] = j["username"]
                    os.environ["KAGGLE_KEY"] = j["key"]
                    return {"kind": "legacy", "username": j["username"],
                            "key": j["key"], "source": str(tok_file)}
            except json.JSONDecodeError:
                pass
        else:
            token, source = raw, str(tok_file)

    if token:
        os.environ["KAGGLE_API_TOKEN"] = token       # for kagglehub / CLI
        return {"kind": "token", "token": token, "source": source}

    # 3. legacy env vars
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return {"kind": "legacy", "username": os.environ["KAGGLE_USERNAME"],
                "key": os.environ["KAGGLE_KEY"], "source": "env"}

    # 4. legacy kaggle.json
    kj = KAGGLE_DIR / "kaggle.json"
    if kj.exists():
        try:
            j = json.loads(kj.read_text())
            os.environ["KAGGLE_USERNAME"] = j["username"]
            os.environ["KAGGLE_KEY"] = j["key"]
            return {"kind": "legacy", "username": j["username"], "key": j["key"],
                    "source": str(kj)}
        except Exception as e:
            print(f"[auth] {kj} unreadable: {e}")

    return {"kind": None}


def _unzip_all(dest: Path):
    for z in list(dest.glob("*.zip")):
        print(f"[dl] unzip {z.name}")
        with zipfile.ZipFile(z) as f:
            f.extractall(dest)
        z.unlink()


# ── method 1: kagglehub (reads the env vars we just set) ───────────────────
def via_kagglehub(dest: Path) -> bool:
    try:
        import kagglehub
    except ImportError:
        print("[dl] kagglehub not installed (pip install kagglehub)")
        return False
    try:
        print("[dl] trying kagglehub.competition_download ...")
        src = Path(kagglehub.competition_download(paths.COMPETITION))
        n = 0
        for f in src.rglob("*"):
            if f.is_file():
                shutil.copy2(f, dest / f.name)
                n += 1
        print(f"[dl] kagglehub copied {n} files from {src}")
        return n > 0
    except Exception as e:
        print(f"[dl] kagglehub failed: {str(e)[:200]}")
        return False


# ── method 2: kaggle CLI ───────────────────────────────────────────────────
def via_cli(dest: Path) -> bool:
    if shutil.which("kaggle") is None:
        print("[dl] kaggle CLI not on PATH (pip install kaggle)")
        return False
    print(f"[dl] trying: kaggle competitions download -c {paths.COMPETITION}")
    r = subprocess.run(
        ["kaggle", "competitions", "download", "-c", paths.COMPETITION,
         "-p", str(dest)],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[dl] CLI failed: {(r.stderr or r.stdout).strip()[:300]}")
        return False
    _unzip_all(dest)
    return True


# ── method 3: direct HTTP (works with either credential style) ─────────────
def via_http(dest: Path, creds: dict) -> bool:
    import requests
    url = f"{API_BASE}/competitions/data/download-all/{paths.COMPETITION}"
    kw = {}
    if creds["kind"] == "token":
        kw["headers"] = {"Authorization": f"Bearer {creds['token']}"}
    else:
        kw["auth"] = (creds["username"], creds["key"])
    print(f"[dl] trying direct HTTP ({creds['kind']} auth) ...")
    try:
        with requests.get(url, stream=True, timeout=300, **kw) as r:
            if r.status_code == 403:
                print("[dl] 403 Forbidden — you almost certainly have not ACCEPTED "
                      "THE COMPETITION RULES on the website with this account.")
                return False
            if r.status_code == 401:
                print("[dl] 401 Unauthorized — token wrong or expired.")
                return False
            r.raise_for_status()
            zp = dest / f"{paths.COMPETITION}.zip"
            total = 0
            with zp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    total += len(chunk)
                    print(f"\r[dl] {total/1e6:.1f} MB", end="")
            print()
        _unzip_all(dest)
        return True
    except Exception as e:
        print(f"[dl] HTTP failed: {str(e)[:200]}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dest = paths.DATA
    dest.mkdir(parents=True, exist_ok=True)

    if not args.force and (dest / "train.csv").exists() and (dest / "test.csv").exists():
        print(f"[dl] data already present in {dest} (--force to redownload)")
    else:
        creds = load_credentials()
        if not creds["kind"]:
            sys.exit(
                "No Kaggle credentials found. Either style works:\n\n"
                "  NEW (token) — kaggle.com/settings/api -> 'Generate New Token'\n"
                "    export KAGGLE_API_TOKEN=<token>\n"
                "    # or: mkdir -p ~/.kaggle && echo <token> > ~/.kaggle/access_token\n"
                "    #     chmod 600 ~/.kaggle/access_token\n\n"
                "  LEGACY (json) — same page -> 'Legacy API Credentials'\n"
                "    ~/.kaggle/kaggle.json   (chmod 600)\n"
                "    # or: export KAGGLE_USERNAME=... KAGGLE_KEY=...\n\n"
                "You must also ACCEPT THE COMPETITION RULES on the website.")
        print(f"[auth] using {creds['kind']} credentials (from {creds.get('source')})")

        # kagglehub first (understands the new token), then CLI, then raw HTTP.
        if not (via_kagglehub(dest) or via_cli(dest) or via_http(dest, creds)):
            sys.exit(
                "\n[dl] all methods failed. Most common causes:\n"
                "  1. Competition rules not accepted on the website (403)\n"
                "  2. Token expired or malformed (401)\n"
                "  3. kagglehub/kaggle too old for token auth:\n"
                "       pip install -U kagglehub kaggle\n"
                "  4. Cluster egress blocked — download locally and scp data/ over")

    # ── verify ──────────────────────────────────────────────────────────────
    import pandas as pd
    tr = pd.read_csv(paths.train_csv())
    te = pd.read_csv(paths.test_csv())
    print(f"\ntrain.csv {tr.shape}  cols={list(tr.columns)}")
    print(f"test.csv  {te.shape}  cols={list(te.columns)}")
    print(f"labels: {tr['label'].value_counts().to_dict()}")
    print(f"perturbations: train={tr['pert'].nunique()} test={te['pert'].nunique()}")
    if len(tr) != 7705 or len(te) != 1813:
        print(f"NOTE expected 7705/1813 rows, got {len(tr)}/{len(te)} — the "
              f"competition data may have been updated.")
    print(f"\nOK -> {dest}")


if __name__ == "__main__":
    main()