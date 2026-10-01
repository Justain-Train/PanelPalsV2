#!/usr/bin/env python3
"""
Combine labelled ML CSVs into one training file, removing duplicates.

Webtoon episodes open by replaying the last panels of the previous episode,
and every episode repeats the same warning/credits pages. Those repeats would
land in both train and test splits and inflate the test score.

A row is dropped when:
  - its normalized text (>= MIN_RECAP_CHARS) already appeared in a different
    episode of the same series (recaps, warnings, credits), or
  - its text and all feature values exactly match an earlier row.

Short texts (SIGH, HUH ?, 0 ...) legitimately recur across panels, so they're
only dropped on an exact feature match.

Usage:
    python -m backend.ml.combine_ml_data backend/ml/ml_data/raw/a.csv backend/ml/ml_data/raw/b.csv ... [--out backend/ml/ml_data/combined/combined.csv]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

MIN_RECAP_CHARS = 8


def series_of(path: Path) -> str:
    """collected_<series>_ep3.csv -> <series>; other files are their own series."""
    return re.sub(r"_ep\d+$", "", path.stem)


def main():
    parser = argparse.ArgumentParser(description="Combine and dedupe ML CSVs")
    parser.add_argument("csvs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("backend/ml/ml_data/combined/combined.csv"))
    args = parser.parse_args()

    frames = []
    for path in args.csvs:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        df["source_sample_id"] = df["sample_id"]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    feature_cols = sorted(c for c in df.columns if c.startswith("feature_"))

    # Boolean features (e.g. feature_is_timestamp, only present in some files)
    # must be numeric, or NaN-filling in prepare_dataset leaves mixed True/0.0
    # strings that sklearn can't convert.
    for col in feature_cols:
        if df[col].dtype == object or df[col].dtype == bool:
            df[col] = df[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)

    series =df["source_file"].map(lambda name: series_of(Path(name)))
    norm = df["text"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()

    exact_key = norm + "|" + df[feature_cols].round(4).astype(str).agg("|".join, axis=1)
    exact_dup = exact_key.duplicated()

    seen = {}  # (series, text) -> first source file
    recap_dup = pd.Series(False, index=df.index)
    for i in df.index:
        if len(norm[i]) < MIN_RECAP_CHARS:
            continue
        key = (series[i], norm[i])
        first = seen.setdefault(key, df.at[i, "source_file"])
        recap_dup[i] = first != df.at[i, "source_file"]

    drop = exact_dup | recap_dup
    dropped = df[drop]
    combined = df[~drop].copy()
    combined["sample_id"] = range(len(combined))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)

    for _, row in dropped.iterrows():
        print(f"  dropped {row['source_file']}#{row['source_sample_id']}: {str(row['text'])[:40]}")
    labels = combined["label"].fillna("<blank>").value_counts().to_dict()
    print(f"{len(df)} rows → {len(combined)} ({len(dropped)} duplicates removed)")
    print(f"labels: {labels}")
    print(f"💾 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
