#!/usr/bin/env python3
"""General multi-output phase-fraction ANN for long-format CALPHAD data.

Inputs:  x_B, x_DY, x_FE, x_ND, temperature_C
Outputs: predicted NP for every phase in the long-format CSV
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


def load_wide_data(path, chunksize, presence_threshold):
    """Pivot long phase rows into one row per composition-temperature condition."""
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
        # max handles duplicate rows and chunk boundaries safely.
        wide = chunk.pivot_table(index=FEATURES, columns="phase", values="NP",
                                 aggfunc="max", fill_value=0.0).reset_index()
        wide.columns.name = None
        pieces.append(wide)
    if not pieces:
        raise ValueError("No usable rows found in the CSV.")

    data = pd.concat(pieces, ignore_index=True, sort=False).fillna(0.0)
    phase_columns = sorted(set(data.columns) - set(FEATURES))
    data = data.groupby(FEATURES, as_index=False)[phase_columns].max()
    phase_columns = sorted(set(data.columns) - set(FEATURES))
    # Remove labels with no positive NP anywhere; they cannot be learned as amounts.
    phase_columns = [p for p in phase_columns if float(data[p].max()) > presence_threshold]
    if not phase_columns:
        raise ValueError("No phase has a positive NP value.")
    return data[FEATURES + phase_columns], phase_columns


def grouped_split(data, seed):
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


def standardize(train_x, valid_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-12] = 1.0
    return ((train_x - mean) / std, (valid_x - mean) / std,
            (test_x - mean) / std, mean, std)


class PhaseFractionANN(nn.Module):
    def __init__(self, n_phases, hidden1=64, hidden2=32, dropout=0.10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            # Phase fractions form a closed composition: all outputs are
            # non-negative and sum to one. Softmax enforces that constraint.
            nn.Linear(hidden2, n_phases), nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.network(x)


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data, phases = load_wide_data(args.csv, args.chunksize, args.presence_threshold)
    train_df, valid_df, test_df = grouped_split(data, args.seed)

    train_raw = train_df[FEATURES].to_numpy(dtype=np.float32)
    valid_raw = valid_df[FEATURES].to_numpy(dtype=np.float32)
    test_raw = test_df[FEATURES].to_numpy(dtype=np.float32)
    x_train, x_valid, x_test, mean, std = standardize(train_raw, valid_raw, test_raw)
    y_train = train_df[phases].to_numpy(dtype=np.float32)
    y_valid = valid_df[phases].to_numpy(dtype=np.float32)
    y_test = test_df[phases].to_numpy(dtype=np.float32)

    model = PhaseFractionANN(len(phases), args.hidden1, args.hidden2, args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=args.batch_size, shuffle=True)
    xv, yv = torch.from_numpy(x_valid).to(device), torch.from_numpy(y_valid).to(device)
    xt = torch.from_numpy(x_test).to(device)

    best_loss, best_state, patience_left = float("inf"), None, args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
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
            print(f"epoch={epoch:4d} train_loss={total / len(train_df):.8f} valid_loss={valid_loss:.8f}")
        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(xt).cpu().numpy()
    metrics = {
        "overall_mae": float(mean_absolute_error(y_test, predictions)),
        "overall_rmse": float(math.sqrt(mean_squared_error(y_test, predictions))),
        "overall_r2": float(r2_score(y_test, predictions, multioutput="variance_weighted")),
        "per_phase": {},
        "mean_actual_sum_np": float(y_test.sum(axis=1).mean()),
        "mean_predicted_sum_np": float(predictions.sum(axis=1).mean()),
        "mean_absolute_sum_error": float(np.abs(y_test.sum(axis=1) - predictions.sum(axis=1)).mean()),
    }
    for i, phase in enumerate(phases):
        metrics["per_phase"][phase] = {
            "mae": float(mean_absolute_error(y_test[:, i], predictions[:, i])),
            "rmse": float(math.sqrt(mean_squared_error(y_test[:, i], predictions[:, i]))),
            "r2": float(r2_score(y_test[:, i], predictions[:, i])) if np.std(y_test[:, i]) > 1e-12 else None,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(), "model_class": "PhaseFractionANN",
        "features": FEATURES, "phases": phases,
        "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        "presence_threshold": args.presence_threshold,
        "metrics": metrics, "n_rows": len(data), "n_train": len(train_df),
        "n_valid": len(valid_df), "n_test": len(test_df),
    }, output)
    print(json.dumps({"device": str(device), "phases": phases,
                      "rows": {"all": len(data), "train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
                      "metrics": metrics, "saved_model": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="phase_fraction_ann.pt")
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--presence-threshold", type=float, default=1e-8)
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
