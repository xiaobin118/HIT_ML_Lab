import os

import matplotlib
matplotlib.use("Agg")  # 无显示器环境下也能保存图片；Jupyter 中可注释掉这一行
import matplotlib.pyplot as plt
import numpy as np

from step1_forward import predict_proba
from step4 import (
    accuracy,
    make_gaussian_data,
    split_dataset,
    standardize,
    train_logreg_gd,
)


def make_meshgrid(X: np.ndarray, resolution: int = 200) -> tuple:
    """
    根据数据范围生成二维网格。

    返回:
        xx: shape (R, R)
        yy: shape (R, R)
        grid: shape (R*R, 2)
    """
    x_min = X[:, 0].min() - 1.0
    x_max = X[:, 0].max() + 1.0
    y_min = X[:, 1].min() - 1.0
    y_max = X[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def plot_data(X: np.ndarray, y: np.ndarray, ax=None):
    """画两类样本的散点图。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    X0 = X[y == 0]
    X1 = X[y == 1]

    ax.scatter(X0[:, 0], X0[:, 1],
               s=20, alpha=0.7, color="tab:orange", label="class 0")
    ax.scatter(X1[:, 0], X1[:, 1],
               s=20, alpha=0.7, color="tab:blue", label="class 1")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper left")
    return ax


def plot_decision_boundary(X: np.ndarray, w: np.ndarray, b: float,
                           ax=None, resolution: int = 200):
    """
    用网格法画逻辑回归的决策边界。

    X: 用于确定绘图范围的数据 (N, 2)
    w: shape (2,)
    b: 标量
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    xx, yy, grid = make_meshgrid(X, resolution)
    p = predict_proba(grid, w, b)
    zz = p.reshape(xx.shape)

    ax.contourf(xx, yy, zz,
                levels=[0, 0.5, 1],
                alpha=0.2,
                colors=["tab:orange", "tab:blue"])
    ax.contour(xx, yy, zz,
               levels=[0.5],
               colors="black",
               linewidths=1.5)
    return ax


if __name__ == "__main__":
    # 1. 生成数据、划分、标准化
    X, y = make_gaussian_data(n_per_class=500, rho=0.8, seed=0)
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_ratio=0.3, seed=42
    )
    X_train_std, X_test_std, mu, sigma = standardize(X_train, X_test)

    # 2. 训练
    w, b, history = train_logreg_gd(
        X_train_std, y_train,
        lr=0.5, n_iter=20000, tol=1e-6
    )

    train_acc = accuracy(X_train_std, y_train, w, b)
    test_acc = accuracy(X_test_std, y_test, w, b)

    # 3. 同时画决策边界和 loss 曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_data(X_train_std, y_train, ax=axes[0])
    plot_decision_boundary(X_train_std, w, b, ax=axes[0], resolution=200)
    axes[0].set_title(
        "Decision boundary\n"
        "train acc = %.3f, test acc = %.3f" % (train_acc, test_acc)
    )

    axes[1].plot(history)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("NLL loss")
    axes[1].set_title("Training loss curve")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "step5_plot.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("figure saved to", out_path)
    print("train accuracy =", train_acc)
    print("test accuracy =", test_acc)
    plt.close(fig)

    assert train_acc > 0.9
    assert test_acc > 0.9
    assert os.path.exists(out_path)
    print("step5_plot.py 测试通过")
