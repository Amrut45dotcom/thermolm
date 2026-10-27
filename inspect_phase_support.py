#!/usr/bin/env python3
"""Inspect phase-wise positive/negative support before building classifiers.

For each phase, this script uses only that phase's rows:
    positive = NP >= --np-threshold
    negative = NP <  --np-threshold

This is the correct support calculation for a phase-presence classifier. Rows
belonging to other phases are not counted as negatives for this phase because
those rows represent different phase records at the same composition/temperature.

The script also performs a composition-grouped 70/15/15 split and reports support
in every split. Use the printed table to choose the phases for the first TabFM or
conventional-classifier experiment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
COMPOSITION = ["x_B", "x_DY", "x_FE", "x_ND"]
REQUIRED = COMPOSITION + ["phase", "NP"]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", required=True, help="DNFB long-format CSV")
    p.add_argument("--out", default="phase_support_inspection.csv")
    p.add_argument("--np-threshold", type=float, default=0.005)
    p.add_argument("--min-positive-compositions", type=int, default=100)
    p.add_argument("--min-negative-compositions", type=int, default=100)
    p.add_argument("--top", type=int, default=10, help="Number of recommended phases")
    return p.parse_args()


def make_split(df):
    groups = df["composition_id"].to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tv_i, te_i = next(first.split(df, groups=groups))
    tv, test = df.iloc[tv_i].copy(), df.iloc[te_i].copy()
    second = GroupShuffleSplit(n_splits=1, test_size=0.1764705882353, random_state=SEED)
    tr_i, va_i = next(second.split(tv, groups=tv["composition_id"].to_numpy()))
    return tv.iloc[tr_i], tv.iloc[va_i], test


def support_for_phase(frame, phase, threshold):
    z = frame[frame.phase == phase]
    pos = z.NP >= threshold
    return {
        "rows": len(z),
        "positive_rows": int(pos.sum()),
        "negative_rows": int((~pos).sum()),
        "positive_compositions": int(z.loc[pos, "composition_id"].nunique()),
        "negative_compositions": int(z.loc[~pos, "composition_id"].nunique()),
        "positive_fraction": float(pos.mean()) if len(z) else np.nan,
    }


def main():
    a = parse_args()
    usecols = REQUIRED
    df = pd.read_csv(a.data, usecols=usecols)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["composition_id"] = pd.factorize(
        pd.MultiIndex.from_frame(df[COMPOSITION])
    )[0]
    train, valid, test = make_split(df)
    phases = sorted(df.phase.unique())
    rows = []
    for phase in phases:
        overall = support_for_phase(df, phase, a.np_threshold)
        tr = support_for_phase(train, phase, a.np_threshold)
        va = support_for_phase(valid, phase, a.np_threshold)
        te = support_for_phase(test, phase, a.np_threshold)
        row = {"phase": phase, "selected": False}
        for prefix, values in [("overall", overall), ("train", tr), ("valid", va), ("test", te)]:
            row.update({f"{prefix}_{k}": v for k, v in values.items()})
        # Selection is based on composition-level support, not correlated row count.
        row["eligible"] = bool(
            tr["positive_compositions"] >= a.min_positive_compositions
            and tr["negative_compositions"] >= a.min_negative_compositions
            and va["positive_rows"] >= a.min_positive_compositions // 2
            and te["positive_rows"] >= a.min_positive_compositions // 2
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    # Rank eligible phases by positive composition support, then by balanced support.
    result["support_score"] = result[["train_positive_compositions", "train_negative_compositions"]].min(axis=1)
    eligible = result[result.eligible].sort_values(
        ["support_score", "train_positive_compositions"], ascending=False
    ).copy()
    result.loc[eligible.head(a.top).index, "selected"] = True
    result = result.sort_values(["selected", "eligible", "support_score"], ascending=[False, False, False])
    result.to_csv(a.out, index=False)

    print(f"Input rows: {len(df):,}")
    print(f"Unique compositions: {df.composition_id.nunique():,}")
    print(f"NP threshold: {a.np_threshold}")
    print("Split: 70% train / 15% validation / 15% test by exact composition")
    print(f"Results written to: {Path(a.out).resolve()}\n")
    cols = [
        "phase", "selected", "eligible",
        "train_positive_compositions", "train_negative_compositions",
        "valid_positive_rows", "test_positive_rows",
        "overall_positive_rows", "overall_negative_rows",
        "overall_positive_fraction",
    ]
    print(result[cols].to_string(index=False))
    print("\nRecommended phases:")
    print(", ".join(result.loc[result.selected, "phase"].tolist()))


if __name__ == "__main__":
    main()
