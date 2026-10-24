#!/usr/bin/env python3
"""Publication-focused Fe-Nd-Dy-B phase benchmark.

This is intentionally stricter than the exploratory benchmark.

Prime-phase selection is based only on the training split:
  * at least --min-positive-compositions distinct positive alloy compositions
  * at least --min-positive-test positive rows in validation and test

Classification:
  * one binary presence task per selected phase, using only rows for that phase
  * presence = NP >= threshold; other phase rows are not treated as negatives
  * primary selection metric: validation average precision (PR-AUC)
  * threshold is tuned on validation data to maximize F1; test is untouched
  * DummyMostFrequent is reported as a baseline, never selected as winner

NP regression:
  * conditional regression on positive phase rows
  * selected by validation RMSE
  * combined with the selected classifier to evaluate end-to-end NP prediction

X_* regression:
  * only variable targets are modeled; near-constant stoichiometric targets are
    reported in constant_targets.csv instead of receiving misleading R2 scores
  * predictions are clipped to [0, 1]

The split is by exact composition, so all temperatures/phases of one alloy stay
in one split. Outputs are CSV/JSON files suitable for further paper analysis.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, roc_auc_score, r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
SEED = 42
FEATURES = ["x_B", "x_DY", "x_FE", "x_ND", "temperature_C"]
X_TARGETS = ["X_B", "X_DY", "X_FE", "X_ND"]
REQUIRED = FEATURES + ["phase", "NP"] + X_TARGETS


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--outdir", default="publication_results")
    p.add_argument("--max-rows", type=int, default=0, help="0 uses all rows")
    p.add_argument("--np-threshold", type=float, default=0.005)
    p.add_argument("--min-positive-compositions", type=int, default=500,
                   help="Strict prime-phase support requirement, measured in training compositions")
    p.add_argument("--min-positive-split-rows", type=int, default=50,
                   help="Minimum positive rows required in validation and test")
    p.add_argument("--min-negative-split-rows", type=int, default=50,
                   help="Minimum negative rows required in train, validation, and test")
    p.add_argument("--min-x-std", type=float, default=0.005,
                   help="Minimum X_* training standard deviation for variable-target regression")
    p.add_argument("--n-jobs", type=int, default=4)
    return p.parse_args()


def pipe(model, scale=False):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def classifiers(n_jobs):
    return {
        "LogisticRegression": pipe(LogisticRegression(max_iter=500, class_weight="balanced"), True),
        "LinearSVC": pipe(LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=4000), True),
        "DecisionTree": pipe(DecisionTreeClassifier(min_samples_leaf=3, class_weight="balanced", random_state=SEED)),
        "RandomForest": pipe(RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced", n_jobs=n_jobs, random_state=SEED)),
        "GradientBoosting": pipe(GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED)),
        "AdaBoost": pipe(AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=SEED)),
        "Bagging": pipe(BaggingClassifier(n_estimators=60, n_jobs=n_jobs, random_state=SEED)),
    }


def regressors(n_jobs):
    return {
        "LinearRegression": pipe(LinearRegression(), True),
        "Ridge": pipe(Ridge(alpha=1.0), True),
        "ElasticNet": pipe(ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=4000, random_state=SEED), True),
        "DecisionTree": pipe(DecisionTreeRegressor(min_samples_leaf=3, random_state=SEED)),
        "RandomForest": pipe(RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=n_jobs, random_state=SEED)),
        "GradientBoosting": pipe(GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=SEED)),
        "AdaBoost": pipe(AdaBoostRegressor(n_estimators=100, learning_rate=0.05, loss="square", random_state=SEED)),
        "Bagging": pipe(BaggingRegressor(n_estimators=60, n_jobs=n_jobs, random_state=SEED)),
    }


def score_vector(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def best_threshold(y, score):
    candidates = np.unique(np.r_[np.quantile(score, np.linspace(0.01, 0.99, 99)), score])
    best = (0.5, -1.0)
    for t in candidates:
        f = f1_score(y, score >= t, zero_division=0)
        if f > best[1]:
            best = (float(t), float(f))
    return best[0]


def classification_metrics(y, score, threshold):
    pred = score >= threshold
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return {
        "average_precision": float(average_precision_score(y, score)) if y.sum() else np.nan,
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "brier": float(brier_score_loss(y, score)) if np.all((score >= 0) & (score <= 1)) else np.nan,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def comp_ids(df):
    return pd.factorize(pd.MultiIndex.from_frame(df[["x_B", "x_DY", "x_FE", "x_ND"]]))[0]


def split(df):
    g = df.composition_id.to_numpy()
    a = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tv_i, te_i = next(a.split(df, groups=g))
    tv, te = df.iloc[tv_i].copy(), df.iloc[te_i].copy()
    b = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=SEED)
    tr_i, va_i = next(b.split(tv, groups=tv.composition_id.to_numpy()))
    return tv.iloc[tr_i].copy(), tv.iloc[va_i].copy(), te


def select_prime_phases(train, valid, test, threshold, min_comps, min_pos_rows, min_neg_rows):
    """Select phases with enough positive and negative examples in every split."""
    def support(frame):
        rows = []
        for phase, z in frame.groupby("phase"):
            y = (z.NP >= threshold)
            pos = z.loc[y]
            neg = z.loc[~y]
            rows.append({
                "phase": phase,
                "positive_rows": int(y.sum()),
                "positive_compositions": int(pos.composition_id.nunique()),
                "negative_rows": int((~y).sum()),
            })
        return pd.DataFrame(rows).set_index("phase") if rows else pd.DataFrame()

    a, b, c = support(train), support(valid), support(test)
    all_phases = sorted(set(train.phase) & set(valid.phase) & set(test.phase))
    rows = []
    for phase in all_phases:
        x = {"phase": phase}
        for label, table in [("train", a), ("valid", b), ("test", c)]:
            x[f"{label}_positive_rows"] = int(table.loc[phase, "positive_rows"]) if phase in table.index else 0
            x[f"{label}_positive_compositions"] = int(table.loc[phase, "positive_compositions"]) if phase in table.index else 0
            x[f"{label}_negative_rows"] = int(table.loc[phase, "negative_rows"]) if phase in table.index else 0
        x["selected_prime"] = bool(
            x["train_positive_compositions"] >= min_comps
            and x["valid_positive_rows"] >= min_pos_rows
            and x["test_positive_rows"] >= min_pos_rows
            and x["train_negative_rows"] >= min_neg_rows
            and x["valid_negative_rows"] >= min_neg_rows
            and x["test_negative_rows"] >= min_neg_rows
        )
        rows.append(x)
    return pd.DataFrame(rows).sort_values("phase")


def run_classification(train, valid, test, phases, threshold, outdir, n_jobs):
    """Train one phase-presence classifier on rows for that phase only."""
    rows, winners = [], []
    for phase in tqdm(phases, desc="Prime-phase classification", unit="phase"):
        # Important: the long table has one row per phase/condition. For this
        # binary task, use only this phase's rows; other phases at the same
        # composition/temperature are NOT negative examples for this classifier.
        def phase_frame(frame):
            z = frame.loc[frame.phase == phase].copy()
            z["y"] = (z.NP >= threshold).astype(np.int8)
            return z

        tr, va, te = phase_frame(train), phase_frame(valid), phase_frame(test)
        for split_name, part in [("train", tr), ("validation", va), ("test", te)]:
            if part.empty or part.y.nunique() < 2:
                raise ValueError(
                    f"Phase {phase!r} has fewer than two classes in {split_name}; "
                    "check phase rows and support thresholds."
                )

        baseline = DummyClassifier(strategy="most_frequent").fit(tr[FEATURES], tr.y)
        base_score = baseline.predict_proba(va[FEATURES])[:, 1]
        base_test = baseline.predict_proba(te[FEATURES])[:, 1]
        bm = classification_metrics(te.y.to_numpy(), base_test, 0.5)
        rows.append({"phase": phase, "model": "DummyMostFrequent", "selected": False,
                     "threshold": 0.5, "positive_train": int(tr.y.sum()),
                     "negative_train": int((tr.y == 0).sum()),
                     "validation_average_precision": average_precision_score(va.y, base_score),
                     **{"test_" + k: v for k, v in bm.items()}, "error": ""})
        candidates = classifiers(n_jobs)
        model_rows = []
        for name, model in tqdm(candidates.items(), total=len(candidates), desc=f"Classify {phase}", leave=False):
            try:
                model.fit(tr[FEATURES], tr.y)
                sv, st = score_vector(model, va[FEATURES]), score_vector(model, te[FEATURES])
                threshold_v = best_threshold(va.y.to_numpy(), sv)
                vm = classification_metrics(va.y.to_numpy(), sv, threshold_v)
                tm = classification_metrics(te.y.to_numpy(), st, threshold_v)
                r = {"phase": phase, "model": name, "selected": False, "threshold": threshold_v,
                     "positive_train": int(tr.y.sum()), "negative_train": int((tr.y == 0).sum()),
                     **{"validation_" + k: v for k, v in vm.items()},
                     **{"test_" + k: v for k, v in tm.items()}, "error": ""}
                model_rows.append(r)
                rows.append(r)
            except Exception as e:
                rows.append({"phase": phase, "model": name, "selected": False, "error": repr(e)})
        good = [r for r in model_rows if not r["error"]]
        if good:
            win = max(good, key=lambda r: (r["validation_average_precision"], r["validation_f1"]))
            for r in rows:
                if r.get("phase") == phase and r.get("model") == win["model"]:
                    r["selected"] = True
            winners.append(win)
    scores = pd.DataFrame(rows)
    scores.to_csv(outdir / "classification_scores.csv", index=False)
    pd.DataFrame(winners).to_csv(outdir / "classification_winners.csv", index=False)
    return scores, pd.DataFrame(winners)


def regression_metrics(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(mean_squared_error(y, pred) ** 0.5),
            "R2": float(r2_score(y, pred)) if np.var(y) > 1e-12 else np.nan}


def run_np_regression(train, valid, test, cls_winners, threshold, outdir, n_jobs):
    rows, selected = [], {}
    for phase in tqdm(cls_winners.phase, desc="Prime-phase NP regression", unit="phase"):
        def pos(frame): return frame[(frame.phase == phase) & (frame.NP >= threshold)].copy()
        tr, va, te = pos(train), pos(valid), pos(test)
        if min(len(tr), len(va), len(te)) < 30: continue
        for name, model in regressors(n_jobs).items():
            try:
                model.fit(tr[FEATURES], tr.NP)
                pv, pt = model.predict(va[FEATURES]), model.predict(te[FEATURES])
                vm, tm = regression_metrics(va.NP, pv), regression_metrics(te.NP, pt)
                rows.append({"phase": phase, "model": name, "selected": False,
                             "n_train": len(tr), "n_valid": len(va), "n_test": len(te),
                             **{"validation_" + k: v for k, v in vm.items()},
                             **{"test_" + k: v for k, v in tm.items()}, "error": ""})
            except Exception as e:
                rows.append({"phase": phase, "model": name, "error": repr(e)})
        good = [r for r in rows if r.get("phase") == phase and not r.get("error")]
        if good:
            win = min(good, key=lambda r: r["validation_RMSE"]); win["selected"] = True; selected[phase] = win
    scores = pd.DataFrame(rows); scores.to_csv(outdir / "np_regression_scores.csv", index=False)
    pd.DataFrame(selected.values()).to_csv(outdir / "np_regression_winners.csv", index=False)
    return scores, selected


def run_end_to_end_np(train, test, cls_winners, np_winners, threshold, outdir, n_jobs):
    """Evaluate end-to-end NP on rows belonging to the target phase only."""
    rows = []
    for phase in cls_winners.phase:
        if phase not in np_winners:
            continue
        cr = cls_winners[cls_winners.phase == phase].iloc[0]
        rr = np_winners[phase]
        clf = classifiers(n_jobs)[cr.model]
        reg = regressors(n_jobs)[rr["model"]]

        # Fit the phase classifier only on examples for this phase.
        train_phase = train.loc[train.phase == phase].copy()
        train_phase["y"] = (train_phase.NP >= threshold).astype(np.int8)
        if train_phase.y.nunique() < 2:
            continue
        clf.fit(train_phase[FEATURES], train_phase.y)

        pos = train_phase.loc[train_phase.NP >= threshold]
        if len(pos) < 2:
            continue
        reg.fit(pos[FEATURES], pos.NP)

        # Evaluate on the corresponding phase rows, not on other phases' rows.
        test_phase = test.loc[test.phase == phase].copy()
        if test_phase.empty:
            continue
        score = score_vector(clf, test_phase[FEATURES])
        pred_present = score >= float(cr.threshold)
        pred_positive = np.clip(reg.predict(test_phase[FEATURES]), 0, 1)
        pred_np = np.where(pred_present, pred_positive, 0.0)
        actual_np = test_phase.NP.to_numpy()
        m = regression_metrics(actual_np, pred_np)
        actual_present = actual_np >= threshold
        cond = actual_present
        cm = regression_metrics(actual_np[cond], pred_np[cond]) if cond.sum() else {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
        rows.append({"phase": phase, "classifier": cr.model, "regressor": rr["model"],
                     "threshold": float(cr.threshold), "n_test_all": len(test_phase),
                     "n_test_present": int(cond.sum()),
                     "overall_MAE": m["MAE"], "overall_RMSE": m["RMSE"], "overall_R2": m["R2"],
                     "conditional_MAE": cm["MAE"], "conditional_RMSE": cm["RMSE"],
                     "conditional_R2": cm["R2"]})
    pd.DataFrame(rows).to_csv(outdir / "np_end_to_end_scores.csv", index=False)


def run_x_regression(train, valid, test, phases, min_std, outdir, n_jobs):
    rows, constants = [], []
    for phase in tqdm(phases, desc="Prime-phase X regression", unit="phase"):
        for target in X_TARGETS:
            def sub(frame): return frame[(frame.phase == phase) & frame[target].notna()].copy()
            tr, va, te = sub(train), sub(valid), sub(test)
            if min(len(tr), len(va), len(te)) < 30: continue
            std, unique = float(tr[target].std(ddof=0)), int(tr[target].nunique())
            if unique < 3 or std < min_std:
                constants.append({"phase": phase, "target": target, "training_mean": float(tr[target].mean()),
                                  "training_std": std, "training_unique": unique,
                                  "description": "near-constant/stoichiometric target; no ML regression"})
                continue
            for name, model in regressors(n_jobs).items():
                try:
                    model.fit(tr[FEATURES], tr[target])
                    pv = np.clip(model.predict(va[FEATURES]), 0, 1)
                    pt = np.clip(model.predict(te[FEATURES]), 0, 1)
                    vm, tm = regression_metrics(va[target], pv), regression_metrics(te[target], pt)
                    rows.append({"phase": phase, "target": target, "model": name, "selected": False,
                                 "training_std": std, "training_unique": unique,
                                 **{"validation_" + k: v for k, v in vm.items()},
                                 **{"test_" + k: v for k, v in tm.items()}, "error": ""})
                except Exception as e:
                    rows.append({"phase": phase, "target": target, "model": name, "error": repr(e)})
            good = [r for r in rows if r.get("phase") == phase and r.get("target") == target and not r.get("error")]
            if good: min(good, key=lambda r: r["validation_RMSE"])["selected"] = True
    pd.DataFrame(rows).to_csv(outdir / "x_regression_scores.csv", index=False)
    pd.DataFrame(constants).to_csv(outdir / "constant_targets.csv", index=False)


def main():
    a = parse_args(); outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.data)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    df["composition_id"] = comp_ids(df)
    if a.max_rows and len(df) > a.max_rows:
        # Keep a deterministic complete-composition sample.
        rng = np.random.default_rng(SEED); ids = df.composition_id.drop_duplicates().to_numpy(copy=True); rng.shuffle(ids)
        sizes = df.groupby("composition_id").size(); chosen, n = [], 0
        for gid in ids:
            if n and n + int(sizes.loc[gid]) > a.max_rows: continue
            chosen.append(gid); n += int(sizes.loc[gid])
            if n >= a.max_rows: break
        df = df[df.composition_id.isin(chosen)].copy()
    train, valid, test = split(df)
    support = select_prime_phases(
        train, valid, test, a.np_threshold,
        a.min_positive_compositions, a.min_positive_split_rows, a.min_negative_split_rows
    )
    support.to_csv(outdir / "phase_support.csv", index=False)
    phases = support.loc[support.selected_prime, "phase"].tolist()
    if not phases: raise RuntimeError("No phases satisfy the strict support criteria; lower --min-positive-compositions.")
    manifest = {"data": str(Path(a.data).resolve()), "rows_used": len(df), "phases_selected": phases,
                "np_threshold": a.np_threshold, "min_positive_compositions": a.min_positive_compositions,
                "min_positive_split_rows": a.min_positive_split_rows,
                "min_negative_split_rows": a.min_negative_split_rows, "min_x_std": a.min_x_std,
                "split": "70/15/15 exact-composition grouped", "random_seed": SEED,
                "train_rows": len(train), "validation_rows": len(valid), "test_rows": len(test)}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    _, cw = run_classification(train, valid, test, phases, a.np_threshold, outdir, a.n_jobs)
    _, nw = run_np_regression(train, valid, test, cw, a.np_threshold, outdir, a.n_jobs)
    run_end_to_end_np(train, test, cw, nw, a.np_threshold, outdir, a.n_jobs)
    run_x_regression(train, valid, test, phases, a.min_x_std, outdir, a.n_jobs)
    print(f"Selected prime phases: {', '.join(phases)}", flush=True)
    print(f"Results written to: {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
