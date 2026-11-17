"""加载训练好的 MLP，对单个 34 维样本做预测。

运行：
    python src/predict.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data import load_ionosphere
from model import MLP

ROOT = Path(__file__).resolve().parents[1]


def load_model(path: str | Path = ROOT / "results" / "model_best.pt"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = MLP(
        in_dim=ckpt["input_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        activation=cfg["activation"],
        dropout=cfg["dropout"],
        batchnorm=cfg["batchnorm"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mu"], ckpt["sd"], ckpt["class_map"]


def predict_one(x: np.ndarray, model, mu: np.ndarray, sd: np.ndarray) -> tuple[str, float]:
    """x: (34,)；返回 (类别, P(g))。"""
    x = (x.reshape(1, -1) - mu) / sd
    with torch.no_grad():
        logit = model(torch.as_tensor(x, dtype=torch.float32))
        p_g = float(torch.sigmoid(logit).item())
    return ("g" if p_g >= 0.5 else "b"), p_g


if __name__ == "__main__":
    X, y = load_ionosphere(ROOT / "data" / "ionosphere.data")
    model, mu, sd, class_map = load_model()
    label, p_g = predict_one(X[0], model, mu, sd)
    print("真实类别:", "g" if y[0] == 1 else "b")
    print(f"预测类别: {label}   P(g)={p_g:.4f}")
    print("类别映射:", class_map)
