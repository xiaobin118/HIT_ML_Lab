"""额外数据集：Fashion-MNIST、Pendigits、Iris、Breast Cancer。

用课程里第 3、4 步的 MLP 和第 1 步的线性分类器，在同一套代码上跑四个不同的数据集：
    - Fashion-MNIST：28x28 灰度服装图，10 类，替代 MNIST 的图像任务；
    - Pendigits：16 维手写数字笔迹特征，10 类，表格数据；
    - Iris：4 维鸢尾花，3 类，150 个样本，小数据；
    - Breast Cancer WDBC：30 维乳腺肿瘤特征，2 类，569 个样本。

运行：python extra_datasets.py
"""

from __future__ import annotations

import gzip
import json
import struct
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import train_test_split
from step01_linear import LinearClassifier
from step04_train import MLP, accuracy, fit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
UCI = ROOT.parent / "experiment3" / "data"
sys.path.insert(0, str(ROOT / "code"))
from datasets import stratified_subsample  # noqa: E402

Array = np.ndarray

FASHION_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
IRIS_NAMES = ["setosa", "versicolor", "virginica"]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def _read_idx_images(path: Path) -> Array:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} 不是 IDX 图像文件")
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols).astype(float) / 255.0


def _read_idx_labels(path: Path) -> Array:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} 不是 IDX 标签文件")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(int)


def load_fashion_mnist(n_train: int = 20000, n_test: int = 5000) -> tuple[Array, Array, Array, Array]:
    """返回分层抽取的 Fashion-MNIST 子集。"""
    d = DATA / "fashion"
    Xtr = _read_idx_images(d / "train-images-idx3-ubyte.gz")
    ytr = _read_idx_labels(d / "train-labels-idx1-ubyte.gz")
    Xte = _read_idx_images(d / "t10k-images-idx3-ubyte.gz")
    yte = _read_idx_labels(d / "t10k-labels-idx1-ubyte.gz")
    Xtr, ytr = stratified_subsample(Xtr, ytr, n_train, seed=0)
    Xte, yte = stratified_subsample(Xte, yte, n_test, seed=1)
    return Xtr, ytr, Xte, yte


def load_iris() -> tuple[Array, Array]:
    """UCI Iris：4 维特征，3 类，150 个样本。"""
    X, labels = [], []
    for line in (UCI / "iris.data").read_text().splitlines():
        if not line.strip():
            continue
        p = line.strip().split(",")
        X.append([float(v) for v in p[:4]])
        labels.append(p[4])
    classes = sorted(set(labels))
    y = np.array([classes.index(v) for v in labels], dtype=int)
    return np.array(X), y


def load_pendigits() -> tuple[Array, Array]:
    """UCI Pendigits：16 维特征，10 类，10992 个样本。"""
    lines = (UCI / "downloads" / "pendigits.arff").read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.strip().lower().startswith("@data"))
    rows = []
    for line in lines[start + 1:]:
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        rows.append([float(v) for v in s.split(",")])
    arr = np.array(rows)
    return arr[:, :-1], arr[:, -1].astype(int)


def load_arff(path: Path) -> tuple[Array, Array]:
    """通用的 ARFF 加载器：最后一列是类别，其余是数值特征。"""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.strip().lower().startswith("@data"))
    rows = []
    for line in lines[start + 1:]:
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        rows.append([p.strip() for p in s.split(",")])
    labels = [r[-1] for r in rows]
    classes = sorted(set(labels))
    X = np.array([[float(v) for v in r[:-1]] for r in rows])
    y = np.array([classes.index(v) for v in labels], dtype=int)
    return X, y


def load_wdbc() -> tuple[Array, Array]:
    """UCI Breast Cancer Wisconsin：30 维特征，2 类（恶性/良性）。"""
    X, y = [], []
    for line in (UCI / "wdbc.data").read_text().splitlines():
        if not line.strip():
            continue
        p = line.strip().split(",")
        X.append([float(v) for v in p[2:]])
        y.append(1 if p[1] == "M" else 0)
    return np.array(X), np.array(y, dtype=int)


# ---------------------------------------------------------------------------
# 训练工具
# ---------------------------------------------------------------------------
class LinearAsList(LinearClassifier):
    """让线性分类器复用 fit：提供 Ws/bs 和梯度列表。"""

    @property
    def Ws(self) -> list[Array]:
        return [self.W]

    @property
    def bs(self) -> list[Array]:
        return [self.b]

    def loss_and_grads(self, X: Array, y: Array, l2: float = 0.0) -> tuple[float, list[Array]]:
        loss, dW, db = super().loss_and_grads(X, y, l2=l2)
        return float(loss), [dW, db]


def standardize(Xtr: Array, Xte: Array) -> tuple[Array, Array]:
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def confusion_matrix(y_true: Array, y_pred: Array, n_classes: int) -> Array:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def run_one(
    name: str,
    Xtr: Array, ytr: Array, Xte: Array, yte: Array,
    hidden: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    standard: bool = True,
) -> dict:
    """训练线性分类器与同结构的 MLP，返回准确率和预测。"""
    if standard:
        Xtr, Xte = standardize(Xtr, Xte)
    D, C = Xtr.shape[1], int(ytr.max()) + 1

    lin = LinearAsList(D, C, seed=0)
    fit(lin, Xtr, ytr, Xte, yte, epochs=epochs, batch_size=batch_size,
        lr=lr, optimizer="adam", seed=0)
    acc_lin = accuracy(lin, Xte, yte)

    mlp = MLP([D] + hidden + [C], activation="relu", init="he", seed=0)
    t0 = time.perf_counter()
    fit(mlp, Xtr, ytr, Xte, yte, epochs=epochs, batch_size=batch_size,
        lr=lr, optimizer="adam", seed=0)
    t_mlp = time.perf_counter() - t0
    pred = mlp.predict(Xte)
    acc_mlp = accuracy(mlp, Xte, yte)

    return {
        "name": name, "n_train": len(Xtr), "n_test": len(Xte),
        "D": D, "C": C, "linear": acc_lin, "mlp": acc_mlp,
        "pred": pred, "time": t_mlp,
    }


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_fashion_samples(X: Array, y: Array, path: str, n: int = 20) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(9.0, 7.8))
    for ax, i in zip(axes.ravel(), range(n)):
        ax.imshow(X[i].reshape(28, 28), cmap="gray_r")
        ax.axis("off")
        ax.set_title(FASHION_NAMES[y[i]], fontsize=8)
    fig.suptitle("Fashion-MNIST 样本", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {path}")


def plot_cm(cm: Array, path: str, title: str, names: list[str] | None = None) -> None:
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    n = len(cm)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = names if names else [str(i) for i in range(n)]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            if norm[i, j] > 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {path}")


def plot_iris_pca(X: Array, y: Array, path: str) -> None:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for c in range(3):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=22, color=colors[c], label=IRIS_NAMES[c],
                   edgecolors="white", linewidths=0.3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Iris 的 PCA 前两维（按类别着色）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {path}")


def plot_compare(results: list[dict], path: str) -> None:
    names = [r["name"] for r in results]
    lin = [r["linear"] for r in results]
    mlp = [r["mlp"] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.2, lin, width=0.4, label="线性分类器", color="#1f77b4")
    ax.bar(x + 0.2, mlp, width=0.4, label="MLP", color="#d62728")
    for i in range(len(names)):
        ax.text(i - 0.2, lin[i] + 0.01, f"{lin[i]:.3f}", ha="center", fontsize=7.5)
        ax.text(i + 0.2, mlp[i] + 0.01, f"{mlp[i]:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("测试准确率")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    ax.set_title(f"{len(names)} 个数据集上的线性分类器 vs MLP")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    results = []

    # Fashion-MNIST（数据没下完时跳过，不影响其他数据集）
    print("\n=== Fashion-MNIST（20000 训练 / 5000 测试 / 12 轮）===")
    fashion_X = fashion_y = None
    try:
        Xtr, ytr, Xte, yte = load_fashion_mnist()
        fashion_X, fashion_y = Xtr, ytr
        r = run_one("Fashion-MNIST", Xtr, ytr, Xte, yte,
                    hidden=[256, 128], epochs=12, batch_size=128, lr=1e-3,
                    standard=False)
        r["cm"] = confusion_matrix(yte, r["pred"], 10)
        results.append(r)
        print(f"  线性 {r['linear']:.4f}  MLP {r['mlp']:.4f}  ({r['time']:.1f}s)")
    except Exception as exc:
        print(f"  数据未就绪，跳过：{exc}")

    # Pendigits
    print("\n=== Pendigits（16 维特征 / 10 类）===")
    X, y = load_pendigits()
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.3, seed=0)
    r = run_one("Pendigits", Xtr, ytr, Xte, yte,
                hidden=[64, 64], epochs=60, batch_size=64, lr=1e-3,
                standard=True)
    results.append(r)
    print(f"  线性 {r['linear']:.4f}  MLP {r['mlp']:.4f}")

    # Optdigits
    print("\n=== Optdigits（64 维 / 10 类 / 5620 样本）===")
    X, y = load_arff(UCI / "downloads" / "optdigits.arff")
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.3, seed=0)
    r = run_one("Optdigits", Xtr, ytr, Xte, yte,
                hidden=[128, 64], epochs=60, batch_size=64, lr=1e-3,
                standard=True)
    results.append(r)
    print(f"  线性 {r['linear']:.4f}  MLP {r['mlp']:.4f}")

    # Letter
    print("\n=== Letter（16 维 / 26 类 / 20000 样本）===")
    X, y = load_arff(UCI / "downloads" / "letter.arff")
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.3, seed=0)
    r = run_one("Letter", Xtr, ytr, Xte, yte,
                hidden=[128, 64], epochs=60, batch_size=128, lr=1e-3,
                standard=True)
    results.append(r)
    print(f"  线性 {r['linear']:.4f}  MLP {r['mlp']:.4f}")

    # Iris
    print("\n=== Iris（4 维 / 3 类 / 150 样本）===")
    X, y = load_iris()
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.3, seed=0)
    r = run_one("Iris", Xtr, ytr, Xte, yte,
                hidden=[32, 32], epochs=600, batch_size=16, lr=1e-3,
                standard=True)
    results.append(r)
    print(f"  线性 {r['linear']:.4f}  MLP {r['mlp']:.4f}")

    # WDBC
    print("\n=== Breast Cancer WDBC（30 维 / 2 类 / 569 样本）===")
    X, y = load_wdbc()
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.3, seed=0)
    r = run_one("WDBC", Xtr, ytr, Xte, yte,
                hidden=[32], epochs=80, batch_size=32, lr=1e-3,
                standard=True)
    results.append(r)
    print(f"  线性 {r['linear']:.4f}  MLP {r['mlp']:.4f}")

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'数据集':<14}{'N_train':>8}{'N_test':>8}{'D':>5}{'C':>4}"
          f"{'线性':>8}{'MLP':>8}{'gap':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<14}{r['n_train']:>8}{r['n_test']:>8}{r['D']:>5}"
              f"{r['C']:>4}{r['linear']:>8.4f}{r['mlp']:>8.4f}"
              f"{r['mlp'] - r['linear']:>+8.4f}")

    if fashion_X is not None:
        plot_fashion_samples(fashion_X, fashion_y, "extra_fashion_samples.png")
        fashion_result = next(r for r in results if r["name"] == "Fashion-MNIST")
        plot_cm(fashion_result["cm"], "extra_fashion_confusion.png",
                f"Fashion-MNIST MLP 混淆矩阵（{fashion_result['mlp']:.3f}）",
                FASHION_NAMES)
    X, y = load_iris()
    plot_iris_pca(X, y, "extra_iris_pca.png")
    plot_compare(results, "extra_compare.png")

    summary = []
    for r in results:
        summary.append({
            "name": r["name"], "n_train": r["n_train"], "n_test": r["n_test"],
            "D": r["D"], "C": r["C"],
            "linear": round(r["linear"], 4), "mlp": round(r["mlp"], 4),
            "gap": round(r["mlp"] - r["linear"], 4),
        })
    out = Path(__file__).resolve().parent / "extra_results.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[结果] {out}")


if __name__ == "__main__":
    main()
