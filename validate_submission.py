"""Validate a Track C submission CSV before zipping/uploading."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import paths

REQUIRED = [
    "id",
    "prediction_up",
    "prediction_down",
    "reasoning_trace",
    "tokens_used",
    "model_name",
]


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if "submission.csv" not in names:
                raise SystemExit("zip must contain submission.csv")
            with zf.open("submission.csv") as fh:
                return pd.read_csv(fh)
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", help="submission CSV or zip")
    args = ap.parse_args()

    p = Path(args.submission)
    sub = _read(p)
    _, test = pd.read_csv(paths.train_csv()), pd.read_csv(paths.test_csv())

    missing = [c for c in REQUIRED if c not in sub.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    if len(sub) != len(test):
        raise SystemExit(f"wrong row count: got {len(sub)}, expected {len(test)}")
    if set(sub["id"]) != set(test["id"]):
        raise SystemExit("submission ids do not match test ids")
    if sub["id"].duplicated().any():
        raise SystemExit("duplicate ids in submission")

    up = pd.to_numeric(sub["prediction_up"], errors="coerce")
    down = pd.to_numeric(sub["prediction_down"], errors="coerce")
    if up.isna().any() or down.isna().any():
        raise SystemExit("non-numeric prediction values")
    if np.any(up < 0) or np.any(up > 1) or np.any(down < 0) or np.any(down > 1):
        raise SystemExit("prediction values must be in [0, 1]")
    if np.any(up + down > 1.000001):
        raise SystemExit("prediction_up + prediction_down must be <= 1")

    if sub["reasoning_trace"].isna().any():
        raise SystemExit("reasoning_trace contains missing values")
    if sub["tokens_used"].isna().any():
        raise SystemExit("tokens_used contains missing values")
    if sub["model_name"].isna().any():
        raise SystemExit("model_name contains missing values")

    print(f"OK: {p} has {len(sub)} valid Track C rows")


if __name__ == "__main__":
    main()
