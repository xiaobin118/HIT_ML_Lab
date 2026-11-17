"""数据生成与加载。

包含三类人工构造的二维四类数据（用于观察决策边界），以及 MNIST 手写数字
数据集的 IDX 文件解析。
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 二维合成数据
# ---------------------------------------------------------------------------


def make_blobs(n_per_class=200, spread=1.0, seed=0):
    """4 个高斯团，分别位于四个象限的对角方向。

    类别之间可以被直线分开，是线性分类器的「友好」场景，用作对照。
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[2.5, 2.5], [2.5, -2.5], [-2.5, 2.5], [-2.5, -2.5]])
    X, y = [], []
    for c, mu in enumerate(centers):
        X.append(rng.normal(mu, spread, size=(n_per_class, 2)))
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def make_rings(n_per_class=250, radii=(1.0, 2.2, 3.4, 4.6), width=0.2, seed=0):
    """4 个同心圆环，一个类别占一个半径区间。

    类与类之间的边界是圆，任何直线都无法把某一环完整地与其他环分开，
    因此线性分类器最多只能做「内 / 外」的粗分。
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c, r in enumerate(radii):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_per_class)
        rad = rng.normal(r, width, size=n_per_class)
        X.append(np.column_stack([rad * np.cos(theta), rad * np.sin(theta)]))
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def make_spiral(n_per_class=250, n_classes=4, turns=2.0, noise=0.03, seed=0):
    """多臂螺旋：每条臂是一个类别，不同臂在空间中交错缠绕。

    类与类之间的界面是弯曲的螺旋面，线性分类器只能切成扇形，MLP 则能
    用分段线性的折线去逼近弯曲线。这是本实验中最能体现非线性能力的场景。
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        t = np.linspace(0.08, 1.0, n_per_class)
        r = t
        theta = 2.0 * np.pi * c / n_classes + turns * np.pi * t
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        pts += rng.normal(0.0, noise, size=pts.shape)
        X.append(pts)
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def train_test_split(X, y, test_frac=0.3, seed=0, shuffle=True):
    """分层留出法：每类按相同比例分到训练集与测试集。"""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        idx = np.flatnonzero(y == c)
        if shuffle:
            idx = rng.permutation(idx)
        n_test = int(round(len(idx) * test_frac))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    if shuffle:
        train_idx = rng.permutation(train_idx)
        test_idx = rng.permutation(test_idx)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


DATASETS_2D = {
    "blobs": make_blobs,
    "rings": make_rings,
    "spiral": make_spiral,
}

DATASET_TITLES = {
    "blobs": "四团高斯（线性可分）",
    "rings": "四同心圆环",
    "spiral": "四臂螺旋",
}


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} 不是 IDX 图像文件 (magic={magic})")
        buf = f.read(n * rows * cols)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)
    return arr.astype(np.float64) / 255.0


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} 不是 IDX 标签文件 (magic={magic})")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def load_mnist(data_dir):
    """返回 (X_train, y_train, X_test, y_test)，像素已缩放至 [0, 1]。"""
    d = Path(data_dir)
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    missing = [f for f in files.values() if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"缺少 MNIST 文件: {missing}，请先运行 data/download_mnist.sh")
    X_train = _read_idx_images(d / files["train_images"])
    y_train = _read_idx_labels(d / files["train_labels"])
    X_test = _read_idx_images(d / files["test_images"])
    y_test = _read_idx_labels(d / files["test_labels"])
    return X_train, y_train, X_test, y_test


def stratified_subsample(X, y, n, seed=0):
    """按类别等比例抽取 n 个样本，保证小样本下类别仍然均衡。"""
    if n >= len(y):
        return X, y
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = n // len(classes)
    idx = []
    for c in classes:
        ci = np.flatnonzero(y == c)
        take = min(per_class, len(ci))
        idx.append(rng.choice(ci, size=take, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]
