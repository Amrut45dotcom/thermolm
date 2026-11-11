#!/usr/bin/env python3
"""Predict all phase fractions from a saved PhaseFractionANN checkpoint."""
import argparse
import json
import numpy as np
import torch
from torch import nn

class PhaseFractionANN(nn.Module):
    def __init__(self, n_phases, hidden1, hidden2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, hidden1), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, n_phases), nn.Softmax(dim=1),
        )
    def forward(self, x):
        return self.network(x)

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--x_B", type=float, required=True)
parser.add_argument("--x_DY", type=float, required=True)
parser.add_argument("--x_FE", type=float, required=True)
parser.add_argument("--x_ND", type=float, required=True)
parser.add_argument("--temperature_C", type=float, required=True)
parser.add_argument("--phase-threshold", type=float, default=0.01,
                    help="Report phases with predicted NP at or above this value")
args = parser.parse_args()

ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
state = ckpt["model_state_dict"]
phases = ckpt["phases"]
mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
std = np.asarray(ckpt["feature_std"], dtype=np.float32)
hidden1 = state["network.0.weight"].shape[0]
hidden2 = state["network.3.weight"].shape[0]
model = PhaseFractionANN(len(phases), hidden1, hidden2)
model.load_state_dict(state)
model.eval()
values = np.asarray([args.x_B, args.x_DY, args.x_FE, args.x_ND, args.temperature_C], dtype=np.float32)
with torch.no_grad():
    amounts = model(torch.from_numpy(((values - mean) / std)).unsqueeze(0)).numpy()[0]
result = {phase: float(amount) for phase, amount in zip(phases, amounts)}
print(json.dumps({"input": dict(zip(ckpt["features"], values.tolist())), "predicted_NP": result,
                  "predicted_sum_NP": float(amounts.sum()),
                  "phase_threshold": args.phase_threshold,
                  "phases_above_threshold": [p for p, a in result.items() if a >= args.phase_threshold]}, indent=2))
