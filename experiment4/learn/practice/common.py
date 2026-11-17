"""练习公用工具（不需要你实现）。

提供：二维数据生成、训练/测试划分、准确率、决策边界绘图、数值梯度检查。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 类型别名：Array = 任意形状的 NumPy 数组
Array = np.ndarray

CLASS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
LIGHT_COLORS = ["#aec7e8", "#ff9896", "#98df8a", "#ffbb78"]


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------


def make_blobs(n_per_class: int = 150, spread: float = 1.0, seed: int = 0) -> tuple[Array, Array]:
    """四团高斯，中心在 (±2.5, ±2.5)，可以被直线分开。

    Parameters
    ----------
    n_per_class : int
        每个类别生成多少个样本。
    spread : float
        高斯分布的标准差，越大类别重叠越多。
    seed : int
        随机种子。

    Returns
    -------
    X : Array
        形状 (4*n_per_class, 2) 的坐标。
    y : Array
        形状 (4*n_per_class,) 的类别下标 0..3。
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[2.5, 2.5], [2.5, -2.5], [-2.5, 2.5], [-2.5, -2.5]])
    X, y = [], []
    for c, mu in enumerate(centers):
        X.append(rng.normal(mu, spread, size=(n_per_class, 2)))
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def make_rings(
    n_per_class: int = 250,
    radii: tuple[float, ...] = (1.0, 2.2, 3.4, 4.6),
    width: float = 0.2,
    seed: int = 0,
) -> tuple[Array, Array]:
    """四个同心圆环，一个类别占一个半径区间。

    Parameters
    ----------
    n_per_class : int
        每个环的样本数。
    radii : tuple[float, ...]
        四个环的中心半径。
    width : float
        半径方向上高斯噪声的标准差。
    seed : int
        随机种子。

    Returns
    -------
    X : Array
        形状 (4*n_per_class, 2)。
    y : Array
        形状 (4*n_per_class,)，类别 0..3。
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c, r in enumerate(radii):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_per_class)
        rad = rng.normal(r, width, size=n_per_class)
        X.append(np.column_stack([rad * np.cos(theta), rad * np.sin(theta)]))
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def make_spiral(
    n_per_class: int = 150,
    turns: float = 2.0,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[Array, Array]:
    """四条相位相差 90° 的螺旋臂，线性分类器无法分开。

    Parameters
    ----------
    n_per_class : int
        每条臂（每个类别）的样本数。
    turns : float
        螺旋绕原点转多少圈。2.0 表示两整圈。
    noise : float
        坐标上叠加的高斯噪声标准差。
    seed : int
        随机种子。

    Returns
    -------
    X : Array
        形状 (4*n_per_class, 2)。
    y : Array
        形状 (4*n_per_class,)，类别 0..3。
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(4):
        t = np.linspace(0.08, 1.0, n_per_class)
        theta = 2.0 * np.pi * c / 4 + turns * np.pi * t
        pts = np.column_stack([t * np.cos(theta), t * np.sin(theta)])
        pts += rng.normal(0.0, noise, size=pts.shape)
        X.append(pts)
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def train_test_split(
    X: Array, y: Array, test_frac: float = 0.3, seed: int = 0
) -> tuple[Array, Array, Array, Array]:
    """按类别分层划分，保证训练/测试集类别比例一致。

    Parameters
    ----------
    X : Array
        特征，形状 (N, D)。
    y : Array
        标签，形状 (N,)。
    test_frac : float
        测试集比例，0.3 表示 30%。
    seed : int
        随机种子。

    Returns
    -------
    X_train, y_train, X_test, y_test : tuple[Array, Array, Array, Array]
        四个数组，顺序固定为"训练特征、训练标签、测试特征、测试标签"。
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx = rng.permutation(idx)
        n_test = int(round(len(idx) * test_frac))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train_idx = rng.permutation(np.concatenate(train_idx))
    test_idx = rng.permutation(np.concatenate(test_idx))
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def accuracy(predict: Callable[[Array], Array], X: Array, y: Array) -> float:
    """计算准确率。

    Parameters
    ----------
    predict : Callable[[Array], Array]
        输入 (N, D) 返回 (N,) 类别下标的函数，例如 model.predict。
    X : Array
        特征，形状 (N, D)。
    y : Array
        真实标签，形状 (N,)。

    Returns
    -------
    float
        预测正确的比例，0..1。
    """
    return float(np.mean(predict(X) == y))


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------


def plot_decision_boundary(
    predict: Callable[[Array], Array],
    X: Array,
    y: Array,
    path: str | Path | None = None,
    title: str = "",
    ax: plt.Axes | None = None,
) -> None:
    """把 predict（输入 (N,2)，输出 (N,) 类别）的决策区域画出来。

    Parameters
    ----------
    predict : Callable[[Array], Array]
        预测函数，输入 (N, 2) 的坐标，返回 (N,) 的类别。
    X : Array
        训练点，形状 (N, 2)，只用来画散点。
    y : Array
        训练标签，形状 (N,)。
    path : str | Path | None
        图片保存路径，例如 "step01_blobs.png"；None 表示不保存。
    title : str
        图标题。
    ax : plt.Axes | None
        画在哪个子图上；None 表示新建一张图。
    """
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 260), np.linspace(y_min, y_max, 260))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = np.asarray(predict(grid)).reshape(xx.shape)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
    else:
        fig = ax.figure
    ax.contourf(xx, yy, zz, levels=np.arange(5) - 0.5,
                cmap=ListedColormap(LIGHT_COLORS), alpha=0.75)
    ax.contour(xx, yy, zz, levels=np.arange(4) - 0.5, colors="k", linewidths=0.6, alpha=0.5)
    for c in range(4):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], s=13, color=CLASS_COLORS[c],
                   edgecolors="white", linewidths=0.35, label=f"类别 {c}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7.5, loc="lower right", markerscale=1.3)
    if path is not None:
        fig.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[图] 已保存到 {path}")


# ---------------------------------------------------------------------------
# 数值梯度检查
# ---------------------------------------------------------------------------


def finite_difference_error(
    loss_fn: Callable[[], float],
    params: list[Array],
    grads: list[Array],
    eps: float = 1e-5,
    max_per_param: int = 40,
    seed: int = 0,
) -> float:
    """中心差分检查梯度。

    Parameters
    ----------
    loss_fn : callable
        无参函数，返回当前参数下的标量损失（会读取 params 里数组的内容）。
    params, grads : list[np.ndarray]
        参数数组列表与对应的解析梯度列表。
    eps : float
        扰动步长。
    max_per_param : int
        每个参数数组最多抽查多少个元素（全查也可以，这里为了速度抽样）。

    Returns
    -------
    worst : float
        所有抽查元素中 |数值梯度 - 解析梯度| 的最大值。
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for p, g in zip(params, grads):
        flat_p = p.reshape(-1)
        flat_g = g.reshape(-1)
        size = flat_p.size
        k = min(max_per_param, size)
        idx = rng.choice(size, size=k, replace=False)
        for i in idx:
            old = flat_p[i]
            flat_p[i] = old + eps
            lp = loss_fn()
            flat_p[i] = old - eps
            lm = loss_fn()
            flat_p[i] = old
            numeric = (lp - lm) / (2.0 * eps)
            worst = max(worst, abs(numeric - flat_g[i]))
    return worst
