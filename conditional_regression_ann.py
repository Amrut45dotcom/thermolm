#!/usr/bin/env python3
"""Positive-only conditional phase-fraction regressors.

For every usable phase, trains an ANN only on rows where NP >= --np-threshold.
The saved checkpoint contains one regressor per phase. It answers:
"given that this phase is present, how much of it is present?"
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ["x_B", "x_DY", "x_FE", "x_ND", "temperature_C"]
COLUMNS = FEATURES + ["phase", "NP"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def phase_name(value):
    return str(value).strip().upper()


def load_wide(path, chunksize):
    pieces = []
    reader = pd.read_csv(path, usecols=lambda c: c in COLUMNS,
                         chunksize=chunksize if chunksize > 0 else None)
    chunks = reader if chunksize > 0 else [reader]
    for chunk in chunks:
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["phase", "NP"])
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["phase"] = chunk["phase"].map(phase_name)
        wide = chunk.pivot_table(index=FEATURES, columns="phase", values="NP",
                                 aggfunc="max", fill_value=0.0).reset_index()
        wide.columns.name = None
        pieces.append(wide)
    if not pieces:
        raise ValueError("No usable rows found in the CSV.")
    data = pd.concat(pieces, ignore_index=True, sort=False).fillna(0.0)
    phases = sorted(set(data.columns) - set(FEATURES))
    data = data.groupby(FEATURES, as_index=False)[phases].max()
    return data[FEATURES + phases], phases


def split_data(data, seed):
    groups = data[["x_B", "x_DY", "x_FE", "x_ND"]].round(8).astype(str).agg("_".join, axis=1)
    indices = np.arange(len(data))
    if groups.nunique() >= 3:
        first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_valid, test = next(first.split(data, groups=groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed + 1)
        train_rel, valid_rel = next(second.split(data.iloc[train_valid], groups=groups.iloc[train_valid]))
        train = np.asarray(train_valid)[train_rel]
        valid = np.asarray(train_valid)[valid_rel]
    else:
        train_valid, test = train_test_split(indices, test_size=0.15, random_state=seed)
        train, valid = train_test_split(train_valid, test_size=0.15 / 0.85, random_state=seed + 1)
    return data.iloc[train].reset_index(drop=True), data.iloc[valid].reset_index(drop=True), data.iloc[test].reset_index(drop=True)


def standardize(x, mean, std):
    return ((x - mean) / std).astype(np.float32, copy=True)


class ConditionalANN(nn.Module):
    def __init__(self, hidden1=64, hidden2=32, dropout=0.10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def train_one_phase(phase, train_df, valid_df, test_df, mean, std, args, device):
    train_df = train_df[train_df[phase] >= args.np_threshold]
    valid_df = valid_df[valid_df[phase] >= args.np_threshold]
    test_df = test_df[test_df[phase] >= args.np_threshold]
    if len(train_df) < args.min_positive_rows or len(valid_df) < 10 or len(test_df) < 10:
        return None

    x_train = standardize(train_df[FEATURES].to_numpy(dtype=np.float32), mean, std)
    x_valid = standardize(valid_df[FEATURES].to_numpy(dtype=np.float32), mean, std)
    x_test = standardize(test_df[FEATURES].to_numpy(dtype=np.float32), mean, std)
    y_train = np.array(train_df[phase].to_numpy(dtype=np.float32), copy=True)
    y_valid = np.array(valid_df[phase].to_numpy(dtype=np.float32), copy=True)
    y_test = np.array(test_df[phase].to_numpy(dtype=np.float32), copy=True)

    model = ConditionalANN(args.hidden1, args.hidden2, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=args.batch_size, shuffle=True)
    xv, yv = torch.from_numpy(x_valid).to(device), torch.from_numpy(y_valid).to(device)
    xt = torch.from_numpy(x_test).to(device)
    best_loss, best_state, patience_left = float("inf"), None, args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(xv), yv).item()
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(xt).cpu().numpy()
    metrics = {
        "positive_train_rows": len(train_df), "positive_valid_rows": len(valid_df), "positive_test_rows": len(test_df),
        "mae": float(mean_absolute_error(y_test, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, prediction))),
        "r2": float(r2_score(y_test, prediction)) if np.std(y_test) > 1e-12 else None,
    }
    return {"state_dict": model.state_dict(), "metrics": metrics,
            "hidden1": args.hidden1, "hidden2": args.hidden2, "dropout": args.dropout}


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data, phases = load_wide(args.csv, args.chunksize)
    train_df, valid_df, test_df = split_data(data, args.seed)
    train_x = train_df[FEATURES].to_numpy(dtype=np.float32)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0)
    std[std < 1e-12] = 1.0
    mean, std = mean.astype(np.float32), std.astype(np.float32)

    models, metrics, excluded = {}, {}, {}
    for phase in phases:
        positive_rows = int((data[phase] >= args.np_threshold).sum())
        positive_groups = int((data[phase] >= args.np_threshold).sum())
        result = train_one_phase(phase, train_df, valid_df, test_df, mean, std, args, device)
        if result is None:
            excluded[phase] = {"positive_rows_all": positive_rows,
                               "reason": f"fewer than {args.min_positive_rows} train, 10 validation, or 10 test positive rows"}
        else:
            models[phase] = result["state_dict"]
            metrics[phase] = result["metrics"]
            print(f"{phase}: positive train/valid/test = {result['metrics']['positive_train_rows']}/"
                  f"{result['metrics']['positive_valid_rows']}/{result['metrics']['positive_test_rows']}  "
                  f"MAE={result['metrics']['mae']:.6f} RMSE={result['metrics']['rmse']:.6f} R2={result['metrics']['r2']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_class": "ConditionalANN", "models": models, "features": FEATURES,
                "phases": list(models), "excluded_phases": excluded,
                "feature_mean": mean.tolist(), "feature_std": std.tolist(),
                "np_threshold": args.np_threshold, "metrics": metrics,
                "rows": {"all": len(data), "train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
                "seed": args.seed}, output)
    print(json.dumps({"device": str(device), "trained_phases": list(models),
                      "excluded_phases": excluded, "rows": {"all": len(data), "train": len(train_df),
                      "valid": len(valid_df), "test": len(test_df)}, "metrics": metrics,
                      "saved_model": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="conditional_phase_regressors.pt")
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--np-threshold", type=float, default=0.005)
    parser.add_argument("--min-positive-rows", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
