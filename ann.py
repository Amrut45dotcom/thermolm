#!/usr/bin/env python3
"""Multi-label phase-presence ANN for the long-format Nd-Dy-Fe-B dataset."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ["x_B", "x_DY", "x_FE", "x_ND", "temperature_C"]
PHASE_COLUMN = "phase"
NP_COLUMN = "NP"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def phase_name(value):
    return str(value).strip().upper()


def load_multilabel_data(csv_path, chunksize, threshold):
    """Return one row per condition and one binary target column per phase.

    The final groupby/max also handles a condition split across CSV chunks.
    """
    columns = FEATURES + [PHASE_COLUMN, NP_COLUMN]
    summaries = []
    reader = pd.read_csv(csv_path, usecols=columns,
                         chunksize=chunksize if chunksize > 0 else None)
    chunks = reader if chunksize > 0 else [reader]
    for chunk in chunks:
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [PHASE_COLUMN, NP_COLUMN])
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk[PHASE_COLUMN] = chunk[PHASE_COLUMN].map(phase_name)
        chunk["present"] = (chunk[NP_COLUMN].astype(float) > threshold).astype(np.int8)
        # If a phase appears more than once for a condition, presence is max.
        grouped = (chunk.groupby(FEATURES + [PHASE_COLUMN], as_index=False)["present"].max())
        wide = grouped.pivot_table(index=FEATURES, columns=PHASE_COLUMN,
                                    values="present", aggfunc="max", fill_value=0).reset_index()
        wide.columns.name = None
        summaries.append(wide)
    if not summaries:
        raise ValueError("No usable rows were found in the CSV.")

    data = pd.concat(summaries, ignore_index=True, sort=False).fillna(0)
    phase_columns = sorted(set(data.columns) - set(FEATURES))
    # Merge condition rows produced in different chunks.
    data = data.groupby(FEATURES, as_index=False)[phase_columns].max()
    phase_columns = sorted(set(data.columns) - set(FEATURES))
    if len(phase_columns) < 2:
        raise ValueError("Fewer than two phases were found; multi-label classification needs multiple phases.")
    return data, phase_columns


def split_data(data, seed, phase_columns):
    groups = data[["x_B", "x_DY", "x_FE", "x_ND"]].round(8).astype(str).agg("_".join, axis=1)
    idx = np.arange(len(data))
    # Retry grouped seeds because a rare phase can otherwise land entirely
    # outside the training split. Composition groups are never broken.
    for attempt in range(100):
        current_seed = seed + attempt
        if groups.nunique() >= 3:
            # First hold out 15% for final testing, then 15/85 of the
            # remainder for validation: approximately 70/15/15 overall.
            first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=current_seed)
            train_valid_idx, test_idx = next(first.split(data, groups=groups))
            second = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=current_seed + 1)
            train_rel, valid_rel = next(second.split(data.iloc[train_valid_idx], groups=groups.iloc[train_valid_idx]))
            train_idx = np.asarray(train_valid_idx)[train_rel]
            valid_idx = np.asarray(train_valid_idx)[valid_rel]
        else:
            train_valid_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=current_seed)
            train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.15 / 0.85, random_state=current_seed + 1)
        if np.all(data.iloc[train_idx][phase_columns].to_numpy().sum(axis=0) > 0):
            return (data.iloc[train_idx].reset_index(drop=True),
                    data.iloc[valid_idx].reset_index(drop=True),
                    data.iloc[test_idx].reset_index(drop=True))
    missing = [p for p in phase_columns if data.iloc[train_idx][p].sum() == 0]
    raise ValueError(f"Could not place every phase in training after 100 grouped splits. Missing: {missing}")


def standardize(train_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-12] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


class MultiLabelANN(nn.Module):
    def __init__(self, n_features, n_phases, hidden1=64, hidden2=32, dropout=0.10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, n_phases),
        )

    def forward(self, x):
        return self.network(x)


def choose_thresholds(y_valid, valid_probabilities):
    """Choose each phase threshold on validation data only by maximizing F1."""
    thresholds = []
    candidates = np.arange(0.10, 0.91, 0.01)
    for i in range(y_valid.shape[1]):
        # If a rare phase is absent from validation, validation cannot select
        # a meaningful F1-optimal threshold. Keep the neutral default.
        if len(np.unique(y_valid[:, i])) < 2:
            thresholds.append(0.50)
            continue
        best_threshold, best_f1 = 0.50, -1.0
        for threshold in candidates:
            predictions = (valid_probabilities[:, i] >= threshold).astype(int)
            score = f1_score(y_valid[:, i], predictions, zero_division=0)
            if score > best_f1:
                best_f1, best_threshold = score, float(threshold)
        thresholds.append(best_threshold)
    return np.asarray(thresholds, dtype=np.float32)


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data, phases = load_multilabel_data(args.csv, args.chunksize, args.presence_threshold)
    # A phase needs enough positive rows and enough distinct composition groups
    # to support a meaningful grouped train/validation/test evaluation.
    composition_groups = data[["x_B", "x_DY", "x_FE", "x_ND"]].round(8).astype(str).agg("_".join, axis=1)
    excluded_phases = {}
    usable_phases = []
    for phase in phases:
        positive_rows = int(data[phase].sum())
        positive_groups = int(composition_groups[data[phase] > 0].nunique())
        if positive_rows < args.min_phase_rows or positive_groups < args.min_phase_groups:
            excluded_phases[phase] = {
                "positive_rows": positive_rows,
                "positive_composition_groups": positive_groups,
                "reason": f"requires at least {args.min_phase_rows} rows and {args.min_phase_groups} composition groups",
            }
        else:
            usable_phases.append(phase)
    phases = usable_phases
    if len(phases) < 2:
        raise ValueError(f"Fewer than two phases remain after rarity filtering. Excluded: {excluded_phases}")
    data = data[FEATURES + phases].copy()
    train_df, valid_df, test_df = split_data(data, args.seed, phases)

    x_train_raw = train_df[FEATURES].to_numpy(dtype=np.float32)
    x_valid_raw = valid_df[FEATURES].to_numpy(dtype=np.float32)
    x_test_raw = test_df[FEATURES].to_numpy(dtype=np.float32)
    x_train, x_valid, x_mean, x_std = standardize(x_train_raw, x_valid_raw)
    x_test = (x_test_raw - x_mean) / x_std
    y_train = train_df[phases].to_numpy(dtype=np.float32)
    y_valid = valid_df[phases].to_numpy(dtype=np.float32)
    y_test = test_df[phases].to_numpy(dtype=np.float32)
    model = MultiLabelANN(len(FEATURES), len(phases), args.hidden1, args.hidden2, args.dropout).to(device)
    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives
    pos_weight = torch.tensor(negatives / np.maximum(positives, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=args.batch_size, shuffle=True)
    xv = torch.from_numpy(x_valid).to(device)
    yv = torch.from_numpy(y_valid).to(device)
    xt = torch.from_numpy(x_test).to(device)

    best_loss, best_state, patience_left = float("inf"), None, args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(xv), yv).item()
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"epoch={epoch:4d} train_loss={train_total / len(train_df):.6f} valid_loss={valid_loss:.6f}")
        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        valid_probabilities = torch.sigmoid(model(xv)).cpu().numpy()
        probabilities = torch.sigmoid(model(xt)).cpu().numpy()
    thresholds = choose_thresholds(y_valid, valid_probabilities)
    predictions = (probabilities >= thresholds[None, :]).astype(int)
    metrics = {
        "micro_f1": float(f1_score(y_test, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_test, predictions, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_test, predictions, average="micro", zero_division=0)),
        "per_phase": {},
    }
    for i, phase in enumerate(phases):
        item = {
            "positive_test_rows": int(y_test[:, i].sum()),
            "f1": float(f1_score(y_test[:, i], predictions[:, i], zero_division=0)),
            "precision": float(precision_score(y_test[:, i], predictions[:, i], zero_division=0)),
            "recall": float(recall_score(y_test[:, i], predictions[:, i], zero_division=0)),
        }
        if len(np.unique(y_test[:, i])) == 2:
            item["roc_auc"] = float(roc_auc_score(y_test[:, i], probabilities[:, i]))
        metrics["per_phase"][phase] = item

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(), "model_class": "MultiLabelANN",
        "features": FEATURES, "phases": phases,
        "feature_mean": x_mean.tolist(), "feature_std": x_std.tolist(),
        "presence_threshold": args.presence_threshold,
        "prediction_thresholds": dict(zip(phases, thresholds.tolist())),
        "excluded_phases": excluded_phases,
        "metrics": metrics, "n_rows": len(data), "n_train": len(train_df),
        "n_valid": len(valid_df), "n_test": len(test_df),
    }, output)
    print(json.dumps({"device": str(device), "phases": phases,
                      "rows": {"all": len(data), "train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
                      "validation_thresholds": dict(zip(phases, thresholds.tolist())),
                      "excluded_phases": excluded_phases,
                      "metrics": metrics, "saved_model": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="multilabel_phase_ann.pt")
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--presence-threshold", type=float, default=1e-8)
    parser.add_argument("--min-phase-rows", type=int, default=20,
                        help="Minimum positive rows required to keep a phase")
    parser.add_argument("--min-phase-groups", type=int, default=3,
                        help="Minimum positive nominal-composition groups required to keep a phase")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
