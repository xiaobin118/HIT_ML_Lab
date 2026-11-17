"""训练、评估与分层 K 折交叉验证。"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import IonDataset, standardize, stratified_kfold
from model import MLP, count_parameters

Array = np.ndarray


@dataclass
class Config:
    """一组超参数。"""

    hidden_dims: tuple[int, ...] = (64, 64)
    activation: str = "relu"
    dropout: float = 0.0
    batchnorm: bool = False
    lr: float = 0.01
    optimizer: str = "sgd"
    batch_size: int = 16
    epochs: int = 100
    weight_decay: float = 0.0
    early_stopping: int = 0
    seed: int = 42

    def describe(self) -> str:
        hidden = "x".join(str(h) for h in self.hidden_dims)
        return (
            f"{len(self.hidden_dims)}x{hidden}-{self.activation}"
            f"-do{self.dropout:g}-{self.optimizer}-lr{self.lr:g}"
            f"-bs{self.batch_size}-ep{self.epochs}"
        )


def binary_metrics(y_true: Array, prob: Array, threshold: float = 0.5) -> dict:
    """二分类指标；正类为 g(good)=1。"""
    pred = (prob >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    n = len(y_true)
    acc = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"未知优化器: {cfg.optimizer}")


@torch.no_grad()
def predict_proba(model: nn.Module, X: Array) -> Array:
    """返回正类概率，形状 (N,)。"""
    model.eval()
    loader = DataLoader(IonDataset(X, np.zeros(len(X))), batch_size=256)
    outs = []
    for xb, _ in loader:
        outs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def evaluate(model: nn.Module, X: Array, y: Array, criterion: nn.Module) -> tuple[float, Array]:
    """返回 (平均损失, 正类概率)。"""
    model.eval()
    loader = DataLoader(IonDataset(X, y), batch_size=256)
    total, n = 0.0, 0
    probs = []
    for xb, yb in loader:
        logits = model(xb)
        total += float(criterion(logits, yb)) * len(yb)
        n += len(yb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return total / n, np.concatenate(probs)


def train_fold(
    X_train: Array, y_train: Array, X_val: Array, y_val: Array,
    cfg: Config, seed: int, track_history: bool = True,
) -> tuple[MLP, list[dict], int]:
    """训练一个 fold，返回模型、每轮曲线和最佳 epoch（1 起）。"""
    torch.manual_seed(seed)
    model = MLP(
        in_dim=X_train.shape[1],
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
        dropout=cfg.dropout,
        batchnorm=cfg.batchnorm,
    )
    criterion = nn.BCEWithLogitsLoss()
    opt = _optimizer(model, cfg)
    train_ds = IonDataset(X_train, y_train)
    drop_last = cfg.batchnorm and len(train_ds) % cfg.batch_size == 1
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        generator=generator, drop_last=drop_last,
    )

    history: list[dict] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict | None = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(yb)
            n += len(yb)

        train_loss = total / max(n, 1)

        if track_history or cfg.early_stopping:
            val_loss, val_prob = evaluate(model, X_val, y_val, criterion)
            if track_history:
                _, train_prob = evaluate(model, X_train, y_train, criterion)
                history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": float(((train_prob >= 0.5) == y_train).mean()),
                    "val_acc": float(((val_prob >= 0.5) == y_val).mean()),
                })
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if cfg.early_stopping and epoch - best_epoch >= cfg.early_stopping:
                break
        elif epoch == cfg.epochs:
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def cross_validate(
    X: Array,
    y: Array,
    cfg: Config,
    folds: list[tuple[Array, Array]] | None = None,
    keep_history: bool = False,
    collect_oof: bool = False,
) -> dict:
    """对一组超参数做分层 K 折交叉验证。

    封闭测试：在每折的训练集上评估（模型见过这些样本）。
    开放测试：在每折的验证集上评估（模型没见过这些样本）。
    """
    if folds is None:
        folds = stratified_kfold(y, n_splits=5, seed=cfg.seed)

    closed_accs, closed_f1s, open_accs, open_f1s = [], [], [], []
    fold_rows = []
    histories = []
    oof_prob = np.zeros(len(y), dtype=np.float64) if collect_oof else None
    n_params = 0
    t0 = time.perf_counter()

    for k, (tr_idx, te_idx) in enumerate(folds):
        X_tr, X_te, _, _ = standardize(X[tr_idx], X[te_idx])
        model, history, best_epoch = train_fold(
            X_tr, y[tr_idx], X_te, y[te_idx], cfg, seed=cfg.seed + k,
            track_history=keep_history and k == 0,
        )
        n_params = count_parameters(model)
        criterion = nn.BCEWithLogitsLoss()

        _, p_tr = evaluate(model, X_tr, y[tr_idx], criterion)
        _, p_te = evaluate(model, X_te, y[te_idx], criterion)
        m_tr = binary_metrics(y[tr_idx], p_tr)
        m_te = binary_metrics(y[te_idx], p_te)

        closed_accs.append(m_tr["acc"])
        closed_f1s.append(m_tr["f1"])
        open_accs.append(m_te["acc"])
        open_f1s.append(m_te["f1"])
        fold_rows.append({
            "fold": k + 1,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "closed_acc": m_tr["acc"],
            "closed_f1": m_tr["f1"],
            "open_acc": m_te["acc"],
            "open_f1": m_te["f1"],
            "best_epoch": int(best_epoch),
        })
        if collect_oof:
            oof_prob[te_idx] = p_te
        if keep_history and k == 0:
            histories.append(history)

    elapsed = time.perf_counter() - t0
    result = {
        "config": cfg.describe(),
        "hidden_dims": list(cfg.hidden_dims),
        "activation": cfg.activation,
        "dropout": cfg.dropout,
        "batchnorm": cfg.batchnorm,
        "lr": cfg.lr,
        "optimizer": cfg.optimizer,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "weight_decay": cfg.weight_decay,
        "early_stopping": cfg.early_stopping,
        "n_params": int(n_params),
        "closed_acc_mean": float(np.mean(closed_accs)),
        "closed_acc_std": float(np.std(closed_accs)),
        "closed_f1_mean": float(np.mean(closed_f1s)),
        "open_acc_mean": float(np.mean(open_accs)),
        "open_acc_std": float(np.std(open_accs)),
        "open_f1_mean": float(np.mean(open_f1s)),
        "fold_open_accs": [float(v) for v in open_accs],
        "folds": fold_rows,
        "time_sec": float(elapsed),
    }
    if keep_history:
        result["histories"] = histories
    if collect_oof:
        result["oof_prob"] = oof_prob.tolist()
    return result
