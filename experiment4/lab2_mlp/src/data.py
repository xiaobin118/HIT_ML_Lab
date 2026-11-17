"""Ionosphere 数据加载、分层 K 折划分与标准化。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

Array = np.ndarray


@dataclass
class Split:
    # """一个训练/验证划分。"""
    X_train: Array
    y_train: Array
    X_test: Array
    y_test: Array
    train_idx: Array
    test_idx: Array


def load_ionosphere(path: str | Path) -> tuple[Array, Array]:
    """读取 ionosphere.data，返回 (X, y)。

    X: (351, 34) float32；y: (351,) int64，g=1，b=0。
    """
    raw = np.loadtxt(path, delimiter=",", dtype=str)
    if raw.shape != (351, 35):
        raise ValueError(f"期望 (351, 35)，实际 {raw.shape}")
    X = raw[:, :-1].astype(np.float32)  # X means features, all columns except the last one
    y = (raw[:, -1] == "g").astype(np.int64)  # y means labels, last column, 1 for 'g', 0 for 'b'
    return X, y


def stratified_kfold(
    y: Array, n_splits: int = 5, seed: int = 42
) -> list[tuple[Array, Array]]:
    """手写分层 K 折，返回 [(train_idx, test_idx), ...]。

    每个类别内部先打乱，再轮流分配到各折，保证每折的正负比例接近。
    """
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]  # folds[k] will hold the indices for the k-th fold
    # for cls in (0, 1): this loop iterates over the two classes (0 and 1)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)  # get indices of samples belonging to class cls
        rng.shuffle(idx)  # shuffle the indices
        for j, i in enumerate(idx):
            folds[j % n_splits].append(int(i))  # distribute indices to folds in a round-robin fashion
    all_idx = set(range(len(y)))  # all indices of the dataset
    splits: list[tuple[Array, Array]] = []
    for k in range(n_splits):
        test_idx = np.array(sorted(folds[k]), dtype=np.int64)  # indices for the k-th fold as test set
        train_idx = np.array(sorted(all_idx - set(folds[k])), dtype=np.int64)  # remaining indices as training set
        splits.append((train_idx, test_idx))
    return splits


def standardize(
    X_train: Array, X_test: Array
) -> tuple[Array, Array, Array, Array]:
    """只用训练集的均值和标准差做标准化，避免测试集信息泄漏。"""
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


class IonDataset(Dataset):
    """把 NumPy 数组包装成 PyTorch Dataset。"""

    def __init__(self, X: Array, y: Array) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]
