#!/usr/bin/env python3
"""Predict multi-label phase presence for a new composition-temperature condition."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn


class MultiLabelANN(nn.Module):
    def __init__(self, n_features, n_phases, hidden1, hidden2, dropout=0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, n_phases),
        )

    def forward(self, x):
        return self.network(x)


def main(args):
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    phases = checkpoint["phases"]
    features = checkpoint["features"]
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    thresholds = checkpoint.get("prediction_thresholds", {p: 0.5 for p in phases})

    hidden1 = state["network.0.weight"].shape[0]
    hidden2 = state["network.3.weight"].shape[0]
    model = MultiLabelANN(len(features), len(phases), hidden1, hidden2)
    model.load_state_dict(state)
    model.eval()

    values = np.asarray([args.x_B, args.x_DY, args.x_FE, args.x_ND, args.temperature_C], dtype=np.float32)
    scaled = (values - mean) / std
    with torch.no_grad():
        logits = model(torch.from_numpy(scaled).unsqueeze(0))
        probabilities = torch.sigmoid(logits).numpy()[0]

    result = {}
    for phase, probability in zip(phases, probabilities):
        threshold = float(thresholds.get(phase, 0.5))
        result[phase] = {
            "probability": float(probability),
            "threshold": threshold,
            "classification": "present" if probability >= threshold else "absent",
        }
    print(json.dumps({"input": dict(zip(features, values.tolist())), "phases": result}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Saved multilabel_phase_ann.pt checkpoint")
    parser.add_argument("--x_B", type=float, required=True)
    parser.add_argument("--x_DY", type=float, required=True)
    parser.add_argument("--x_FE", type=float, required=True)
    parser.add_argument("--x_ND", type=float, required=True)
    parser.add_argument("--temperature_C", type=float, required=True)
    main(parser.parse_args())
