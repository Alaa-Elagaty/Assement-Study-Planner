"""Mark predictor used by the planner.

Loading is LAZY (nothing is loaded at import time) and PyTorch is optional:
if torch is missing, the same weights are read from models/study_planner_nn.npz
(create it with src/nn/export_numpy_weights.py). Both paths give the same
numbers (checked in tests/test_requirements.py).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "study_planner_nn.pth"
NPZ_PATH = PROJECT_ROOT / "models" / "study_planner_nn.npz"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

# Model input features - order must match training
FEATURES = [
    "date", "weight", "previous_tma_count", "previous_tma_mean",
    "previous_tma_last_score", "previous_clicks", "previous_active_days",
    "previous_resources_accessed",
]

_state = {}


def _load():
    """Load scaler + weights once (torch if available, else numpy)."""
    if _state:
        return _state
    _state["scaler"] = joblib.load(SCALER_PATH)
    try:
        import sys
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model import StudyPlannerNN
        net = StudyPlannerNN(input_size=len(FEATURES))
        net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        net.eval()
        _state["torch"], _state["net"] = torch, net
    except ImportError:
        if not NPZ_PATH.exists():
            from export_numpy_weights import main as export_weights  # noqa
            export_weights()
        _state["np"] = np.load(NPZ_PATH)
    return _state


def _forward(x):
    st = _load()
    if "net" in st:
        torch = st["torch"]
        with torch.no_grad():
            return float(st["net"](torch.tensor(x, dtype=torch.float32)).item())
    w = st["np"]
    h = np.maximum(0.0, x @ w["network_0_weight"].T + w["network_0_bias"])
    h = np.maximum(0.0, h @ w["network_2_weight"].T + w["network_2_bias"])
    return float((h @ w["network_4_weight"].T + w["network_4_bias"])[0, 0])


def predict_mark(features):
    """Predict a student's expected TMA mark from the 8 model features.

    Returns {"raw_prediction": ..., "predicted_mark": ...} (mark kept in 0-100).
    """
    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    frame = pd.DataFrame([[features[n] for n in FEATURES]], columns=FEATURES)
    scaled = _load()["scaler"].transform(frame).astype(np.float32)
    raw = _forward(scaled)
    return {"raw_prediction": round(raw, 2),
            "predicted_mark": round(max(0.0, min(100.0, raw)), 2)}


if __name__ == "__main__":
    test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
    sample = test.iloc[0]
    result = predict_mark({f: sample[f] for f in FEATURES})
    print("Actual mark:", sample["score"], "| prediction:", result)
