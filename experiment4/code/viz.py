"""可视化工具：决策边界、训练曲线、混淆矩阵、MNIST 样本网格。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "AR PL UMing CN",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 四类 / 十类通用配色
CLASS_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
LIGHT_COLORS = [
    "#aec7e8",
    "#ff9896",
    "#98df8a",
    "#ffbb78",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]


def savefig(fig, path, dpi=150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")
    return path


def plot_decision_boundary(ax, model, X, y, title="", resolution=260, pad=0.4,
                           show_train=True, n_classes=4):
    """在 ax 上画分类器的决策区域与样本点。"""
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = model.predict(grid).reshape(xx.shape)
    cmap = ListedColormap(LIGHT_COLORS[:n_classes])
    ax.contourf(xx, yy, zz, levels=np.arange(n_classes + 1) - 0.5, cmap=cmap, alpha=0.75)
    ax.contour(xx, yy, zz, levels=np.arange(n_classes) - 0.5, colors="k", linewidths=0.6, alpha=0.5)
    for c in range(n_classes):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], s=11, color=CLASS_COLORS[c], edgecolors="white",
                   linewidths=0.3, label=f"class {c}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")


def plot_history(ax, history, key, label, color=None, linestyle="-", marker=None):
    epochs = [h["epoch"] for h in history]
    vals = [h[key] for h in history]
    ax.plot(epochs, vals, label=label, color=color, linestyle=linestyle,
            marker=marker, markersize=3, linewidth=1.6)


def plot_confusion_matrix(ax, cm, title="混淆矩阵", cmap="Blues", normalize=False):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
    im = ax.imshow(cm, cmap=cmap)
    n = cm.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    fmt = (lambda v: f"{v:.2f}") if normalize else (lambda v: f"{int(v)}")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, fmt(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    return im


def plot_mnist_grid(images, labels, preds=None, ncols=8, title="", save_path=None,
                    correct_mask=None):
    """把若干 28x28 图像排成网格；给了 preds 时在标题里显示预测。"""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.15, nrows * 1.35))
    axes = np.atleast_2d(axes)
    for k in range(nrows * ncols):
        ax = axes[k // ncols, k % ncols]
        ax.axis("off")
        if k >= n:
            continue
        ax.imshow(images[k].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
        if preds is None:
            ax.set_title(str(int(labels[k])), fontsize=9)
        else:
            ok = preds[k] == labels[k] if correct_mask is None else correct_mask[k]
            color = "black" if ok else "crimson"
            ax.set_title(f"{int(labels[k])}→{int(preds[k])}", fontsize=9, color=color)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig
