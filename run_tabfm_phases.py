#!/usr/bin/env python3
"""Per-phase experiments with Google TabFM (classification and regression), plus gradient-boosting baselines.

--task classification (default)
    Per phase: is the phase present?   label = NP >= --np-threshold
    Rows used: every (composition, temperature) row of that phase.

--task regression
    Per phase: how much of the phase is there?   target = NP
    Rows used: only rows where the phase is present (NP >= --np-threshold).
    (Use classification first to decide presence, then regression for the amount.)

Features: x_B, x_DY, x_FE, x_ND and temperature.

The 70/15/15 composition split is rebuilt exactly as in inspect_phase_support.py (same seed, same calls).
For the 10 phases of your first run it checks the rebuilt train positive-composition counts against your
support table and prints a WARNING if they differ.

Models, all scored on the SAME test compositions:
    tabfm     : TabFM with a context sampled by whole compositions
    gbm_ctx   : gradient boosting on that same context (fair like-for-like)
    gbm_full  : gradient boosting on ALL train rows of the phase (the real bar to beat)

About the context (verified in the tabfm 1.0.1 source code, not in the README):
    * The estimators have no hard row limit. max_num_rows defaults to None, so ALL rows you pass to fit()
      go into the context of every ensemble member.
    * n_estimators defaults to 32. Each member is a separate forward pass over the whole context, so run time
      and GPU memory grow with rows x members. This script defaults to 8 members for a first pass.
    * --max-context caps the rows given to fit() (chosen as whole compositions, all temperatures kept).
    * --max-num-rows is TabFM's own option: each member gets a different random row sample of the context.
      That lets all members together see more data than one member can hold in memory.

Usage (PowerShell):
    python run_tabfm_phases.py --data .\\dataset\\DNFB_12_dataset_long.csv --phases LIQUID,FCC_A1 --no-tabfm
    python run_tabfm_phases.py --data ... --phases FCC_A1 --max-context 8000 --n-estimators 4
    python run_tabfm_phases.py --data ... --task regression --phases LIQUID,HCP_A3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
COMPOSITION = ["x_B", "x_DY", "x_FE", "x_ND"]
TEMP_CANDIDATES = ["T", "T_K", "T_C", "temperature", "Temperature", "temp", "Temp", "TEMP"]

DEFAULT_PHASES = [
    "LIQUID", "FE4NDB4", "FE14ND2B", "HCP_A3", "FCC_A1",
    "DYB2", "BCC_A2", "FE2ND5B6", "DYFE3", "FE17ND2",
]
# train_positive_compositions from your inspect_phase_support.py output (for the split check).
EXPECTED_TRAIN_POS_COMPS = {
    "LIQUID": 2286, "FE4NDB4": 1663, "FE14ND2B": 1445, "HCP_A3": 1219, "FCC_A1": 711,
    "DYB2": 630, "BCC_A2": 570, "FE2ND5B6": 563, "DYFE3": 504, "FE17ND2": 476,
    "NDB4": 473, "DY2FE17": 416, "FE2B": 394, "FE17ND5": 356, "ND2B5": 341,
    "DY6FE23": 227, "DHCP": 159, "DYFE2": 156, "FEB": 100,
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", required=True, help="DNFB long-format CSV")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--out-dir", default="tabfm_runs")
    p.add_argument("--phases", default=",".join(DEFAULT_PHASES), help="comma-separated phase names")
    p.add_argument("--np-threshold", type=float, default=0.005)
    p.add_argument("--temp-col", default=None, help="temperature column name (auto-detected if omitted)")
    p.add_argument("--max-context", type=int, default=20000,
                   help="max training rows given to TabFM / gbm_ctx (whole compositions); 0 = all train rows")
    p.add_argument("--context-strategy", choices=["balanced", "random"], default="balanced",
                   help="classification only. balanced = half compositions with a positive row, half without")
    p.add_argument("--n-estimators", type=int, default=8, help="TabFM ensemble members (library default is 32)")
    p.add_argument("--max-num-rows", type=int, default=None,
                   help="TabFM option: random rows per ensemble member (default None = all context rows)")
    p.add_argument("--valid-max-rows", type=int, default=4000, help="cap validation rows (by composition)")
    p.add_argument("--predict-batch", type=int, default=4096, help="rows per TabFM predict call")
    p.add_argument("--device", default=None, help="e.g. cuda, cpu (default: let TabFM decide)")
    p.add_argument("--no-tabfm", action="store_true", help="baselines only (checks pipeline and split, no GPU)")
    return p.parse_args()


# ---------------------------------------------------------------- split (copy of inspect script)
def make_split(df):
    groups = df["composition_id"].to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tv_i, te_i = next(first.split(df, groups=groups))
    tv, test = df.iloc[tv_i], df.iloc[te_i]
    second = GroupShuffleSplit(n_splits=1, test_size=0.1764705882353, random_state=SEED)
    tr_i, va_i = next(second.split(tv, groups=tv["composition_id"].to_numpy()))
    return tv.iloc[tr_i], tv.iloc[va_i], test


def composition_split_labels(df):
    """Array: composition_id -> 0 train, 1 valid, 2 test."""
    train, valid, test = make_split(df)
    lab = np.full(df["composition_id"].max() + 1, -1, dtype=np.int8)
    lab[train["composition_id"].unique()] = 0
    lab[valid["composition_id"].unique()] = 1
    lab[test["composition_id"].unique()] = 2
    assert (lab >= 0).all(), "some compositions were not assigned to a split"
    return lab


# ---------------------------------------------------------------- sampling
def cap_by_composition(d, max_rows, rng):
    """Keep whole compositions (all temperatures) until about max_rows rows are reached."""
    if max_rows <= 0 or len(d) <= max_rows:
        return d
    ids = d["composition_id"].unique()
    rng.shuffle(ids)
    sizes = d.groupby("composition_id").size().reindex(ids)
    n_keep = int(np.searchsorted(sizes.cumsum().to_numpy(), max_rows)) + 1
    return d[d["composition_id"].isin(ids[:n_keep])]


def pick_context(train, max_rows, strategy, rng):
    """Classification: choose WHOLE compositions; 'balanced' keeps positives from being drowned out."""
    if max_rows <= 0 or len(train) <= max_rows:
        return train
    if strategy == "random":
        return cap_by_composition(train, max_rows, rng)
    g = train.groupby("composition_id")["y"].agg(["size", "max"])
    rows_per = max(1, int(g["size"].median()))
    n_comp = max(1, min(len(g), max_rows // rows_per))
    pos = g.index[g["max"] == 1].to_numpy()
    neg = g.index[g["max"] == 0].to_numpy()
    n_pos = min(len(pos), n_comp // 2)
    n_neg = min(len(neg), n_comp - n_pos)
    n_pos = min(len(pos), n_comp - n_neg)  # top up if negatives ran out
    ids = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
    return train[train["composition_id"].isin(ids)]


# ---------------------------------------------------------------- metrics
def best_f1_threshold(y, p):
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    return float(th[int(np.nanargmax(f1[:-1]))])


def clf_scores(y, p, thr):
    has_both = 0 < y.sum() < len(y)
    pred = p >= thr
    return {
        "auc": roc_auc_score(y, p) if has_both else np.nan,
        "ap": average_precision_score(y, p) if y.sum() > 0 else np.nan,
        "f1_at_0.5": f1_score(y, p >= 0.5, zero_division=0),
        "thr_from_valid": thr,
        "f1_tuned": f1_score(y, pred, zero_division=0),
        "precision_tuned": precision_score(y, pred, zero_division=0),
        "recall_tuned": recall_score(y, pred, zero_division=0),
    }


def reg_scores(y, p):
    return {
        "mae": mean_absolute_error(y, p),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": r2_score(y, p),
    }


# ---------------------------------------------------------------- models
def fit_gbm_clf(X, y):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=63, early_stopping=False, random_state=SEED
    ).fit(X, y)


def fit_gbm_reg(X, y):
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=63, early_stopping=False, random_state=SEED
    ).fit(X, y)


def load_tabfm_network(task, device):
    import torch
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
    # TabFM's load() stays on the CPU unless a device is passed, so pick the GPU when there is one.
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TabFM device: {device}   torch {torch.__version__}")
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory: {free / 2**30:.1f} GB free of {total / 2**30:.1f} GB (model needs about 3.3 GB in bf16)")
    else:
        print("WARNING: running TabFM on the CPU. Expect it to be very slow.")
    return tabfm_v1_0_0.load(model_type=task, device=device)


def tabfm_clf_predict(network, Xc, yc, eval_frames, a):
    from tabfm import TabFMClassifier
    clf = TabFMClassifier(model=network, n_estimators=a.n_estimators, max_num_rows=a.max_num_rows)
    clf.fit(Xc, yc)
    classes = list(clf.classes_)
    outs = []
    for X in eval_frames:
        ps = []
        for i in range(0, len(X), a.predict_batch):
            pr = np.asarray(clf.predict_proba(X.iloc[i:i + a.predict_batch]))
            if pr.ndim == 2 and pr.shape[1] == len(classes) and 1 in classes:
                ps.append(pr[:, classes.index(1)])
            else:
                ps.append(np.full(len(pr), 1.0 if classes == [1] else 0.0))
        outs.append(np.concatenate(ps))
    return outs


def tabfm_reg_predict(network, Xc, yc, eval_frames, a):
    from tabfm import TabFMRegressor
    reg = TabFMRegressor(model=network, n_estimators=a.n_estimators, max_num_rows=a.max_num_rows)
    reg.fit(Xc, yc)
    outs = []
    for X in eval_frames:
        ps = [np.asarray(reg.predict(X.iloc[i:i + a.predict_batch])) for i in range(0, len(X), a.predict_batch)]
        outs.append(np.concatenate(ps))
    return outs


def is_oom(err):
    return "out of memory" in str(err).lower()


def run_with_oom_retry(build_ctx, work, start_rows, floor=1000):
    """Run work(ctx). If the GPU runs out of memory, halve the context rows and try again."""
    rows = start_rows
    while True:
        ctx = build_ctx(rows)
        try:
            return work(ctx)
        except RuntimeError as e:
            used = rows if rows else len(ctx)
            if not is_oom(e) or used <= floor:
                raise
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            rows = used // 2
            print(f"    out of memory, retrying with about {rows} context rows")


# ---------------------------------------------------------------- main
def main():
    a = parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    is_clf = a.task == "classification"

    header = pd.read_csv(a.data, nrows=0).columns.tolist()
    temp_col = a.temp_col or next((c for c in TEMP_CANDIDATES if c in header), None)
    if temp_col is None:
        sys.exit(f"Could not find a temperature column. Columns are: {header}. Pass --temp-col NAME.")
    feats = COMPOSITION + [temp_col]
    print(f"Task: {a.task}   Features: {feats}")

    df = pd.read_csv(a.data, usecols=COMPOSITION + [temp_col, "phase", "NP"])
    df["composition_id"] = pd.factorize(pd.MultiIndex.from_frame(df[COMPOSITION]))[0]
    lab = composition_split_labels(df)
    print(f"Rows: {len(df):,}   compositions: {len(lab):,}   "
          f"train/valid/test compositions: {(lab == 0).sum()}/{(lab == 1).sum()}/{(lab == 2).sum()}")

    network = None if a.no_tabfm else load_tabfm_network(a.task, a.device)

    phases = [s.strip() for s in a.phases.split(",") if s.strip()]
    results, pred_frames = [], []

    for phase in phases:
        d = df[df["phase"] == phase].copy()
        if d.empty:
            print(f"[{phase}] no rows, skipped")
            continue
        d["y"] = (d["NP"].to_numpy() >= a.np_threshold).astype(int)
        d["split"] = lab[d["composition_id"].to_numpy()]

        n_pos_comp = d[(d["split"] == 0) & (d["y"] == 1)]["composition_id"].nunique()
        exp = EXPECTED_TRAIN_POS_COMPS.get(phase)
        tag = "" if exp is None else ("  [split matches your support table]" if exp == n_pos_comp
                                      else f"  [WARNING: support table says {exp}, rebuilt {n_pos_comp}]")
        print(f"\n[{phase}] train positive compositions: {n_pos_comp}{tag}")

        if not is_clf:
            d = d[d["y"] == 1]  # regression only on rows where the phase is present
            target = "NP"
        else:
            target = "y"

        tr, va, te = (d[d["split"] == k] for k in (0, 1, 2))
        va = cap_by_composition(va, a.valid_max_rows, rng)
        if len(tr) < 50 or len(va) < 10 or len(te) < 10:
            print(f"[{phase}] too few rows (train {len(tr)}, valid {len(va)}, test {len(te)}), skipped")
            continue
        print(f"[{phase}] train {len(tr):,}  valid {len(va):,}  test {len(te):,}"
              + (f"  test positive rate {te['y'].mean():.4f}" if is_clf else ""))

        probs = {}  # model -> (pred_valid, pred_test, context_rows, seconds)

        # --- baseline on all train rows
        t0 = time.time()
        if is_clf:
            g = fit_gbm_clf(tr[feats], tr[target])
            probs["gbm_full"] = (g.predict_proba(va[feats])[:, 1], g.predict_proba(te[feats])[:, 1],
                                 len(tr), time.time() - t0)
        else:
            g = fit_gbm_reg(tr[feats], tr[target])
            probs["gbm_full"] = (g.predict(va[feats]), g.predict(te[feats]), len(tr), time.time() - t0)

        # --- context-limited models
        def build_ctx(rows):
            return (pick_context(tr, rows, a.context_strategy, rng) if is_clf
                    else cap_by_composition(tr, rows, rng))

        def work(ctx):
            out = {}
            if is_clf and ctx["y"].nunique() < 2:
                print(f"[{phase}] context has a single class, skipping context models")
                return out
            t = time.time()
            if is_clf:
                g2 = fit_gbm_clf(ctx[feats], ctx[target])
                out["gbm_ctx"] = (g2.predict_proba(va[feats])[:, 1], g2.predict_proba(te[feats])[:, 1],
                                  len(ctx), time.time() - t)
            else:
                g2 = fit_gbm_reg(ctx[feats], ctx[target])
                out["gbm_ctx"] = (g2.predict(va[feats]), g2.predict(te[feats]), len(ctx), time.time() - t)
            if network is not None:
                t = time.time()
                fn = tabfm_clf_predict if is_clf else tabfm_reg_predict
                pv, pt = fn(network, ctx[feats], ctx[target].to_numpy(), [va[feats], te[feats]], a)
                out["tabfm"] = (pv, pt, len(ctx), time.time() - t)
            return out

        probs.update(run_with_oom_retry(build_ctx, work, a.max_context))

        # --- score (thresholds / nothing tuned on test)
        for name, (pv, pt, n_ctx, secs) in probs.items():
            row = {"task": a.task, "phase": phase, "model": name, "context_rows": n_ctx,
                   "test_rows": len(te), "seconds": round(secs, 1)}
            if is_clf:
                thr = best_f1_threshold(va["y"].to_numpy(), pv)
                row["test_positive_rate"] = te["y"].mean()
                row.update(clf_scores(te["y"].to_numpy(), pt, thr))
                print(f"[{phase}] {name:9s} AUC {row['auc']:.4f}  AP {row['ap']:.4f}  "
                      f"F1(tuned) {row['f1_tuned']:.4f}  ctx {n_ctx:,}  {secs:.0f}s")
            else:
                row["test_target_std"] = float(te["NP"].std())
                row.update(reg_scores(te["NP"].to_numpy(), pt))
                print(f"[{phase}] {name:9s} MAE {row['mae']:.5f}  RMSE {row['rmse']:.5f}  "
                      f"R2 {row['r2']:.4f}  ctx {n_ctx:,}  {secs:.0f}s")
            results.append(row)

        # --- keep test predictions
        pf = te[["composition_id", temp_col, "NP", "y"] + COMPOSITION].copy()
        pf.insert(0, "phase", phase)
        for name, (_, pt, _, _) in probs.items():
            pf[f"pred_{name}"] = pt
        pred_frames.append(pf)

        pd.DataFrame(results).to_csv(out_dir / f"results_{a.task}.csv", index=False)  # save as we go

    if not results:
        sys.exit("No results produced.")
    res = pd.DataFrame(results)
    res.to_csv(out_dir / f"results_{a.task}.csv", index=False)
    pd.concat(pred_frames).to_csv(out_dir / f"test_predictions_{a.task}.csv", index=False)

    order = [p for p in phases if p in set(res["phase"])]
    if is_clf:
        show = res.pivot(index="phase", columns="model", values="ap").reindex(order)
        show.insert(0, "base_rate", res.groupby("phase")["test_positive_rate"].first())
        print("\nAverage precision on test (higher is better; base_rate = what random guessing gets):")
    else:
        show = res.pivot(index="phase", columns="model", values="r2").reindex(order)
        print("\nR2 on test, rows where the phase is present (higher is better; 0 = predicting the mean):")
    print(show.round(4).to_string())
    print(f"\nWrote results_{a.task}.csv and test_predictions_{a.task}.csv in {out_dir}")


if __name__ == "__main__":
    main()