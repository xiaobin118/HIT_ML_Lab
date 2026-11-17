"""讲义 02 的交互演示：手动指定初始中心，看 k-means 的两步怎么走。

这不是 k-means 的实现（实现留给你在 code/kmeans.py 里写），而是一个
"单步执行器"：它只做一次分配、一次更新，然后把过程画出来，让你看清
每个点是怎么被重新归属的、中心是怎么移动的。

运行方式：

    cd learn
    python demo02_kmeans_steps.py                    # 用默认初始中心
    python demo02_kmeans_steps.py "0,-3" "0,4" "5,0" # 自己指定三个初始中心
    python demo02_kmeans_steps.py "0,-3" "5,0"       # 只给两个 → K=2

试着故意给一组"很差"的初始中心（比如三个都挤在左下角），看看结果会
变成什么样。这是讲义 03 要讨论的"局部最优"问题的直观来源。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import data_gen as dg  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def assign(X, centroids):
    """把每个点分配给最近的中心，返回簇编号数组。

    这里用 (N, K) 的距离矩阵一次算完，避免 Python 双重循环：
    X[:, None, :] 把 X 变成 (N, 1, D)，centroids[None, :, :] 是 (1, K, D)，
    两者相减自动广播成 (N, K, D)，再对最后一维（特征维）求和得到 (N, K)。
    """
    diff = X[:, None, :] - centroids[None, :, :]      # (N, K, D)
    dist2 = (diff ** 2).sum(axis=2)                   # (N, K)
    return dist2.argmin(axis=1)                       # (N,)


def compute_inertia(X, labels, centroids):
    """簇内平方和 J = 每个点到它所属中心的距离平方之和。"""
    diff = X - centroids[labels]
    return float((diff ** 2).sum())


def update_centroids(X, labels, k):
    """把每个中心移到它那一簇所有点的平均位置。"""
    new = np.empty((k, X.shape[1]))
    for j in range(k):
        mask = labels == j
        if mask.sum() == 0:
            # 空簇：没有任何点归它管，均值没有定义，这里保留原位置并报告
            print(f"  [注意] 簇 {j} 没有分到任何点（空簇）")
            continue
        new[j] = X[mask].mean(axis=0)
    return new


def parse_centers(argv, default):
    """把命令行里的 "0,-3" 这样的字符串解析成坐标数组。"""
    if not argv:
        return np.array(default, dtype=float)
    pts = []
    for s in argv:
        pts.append([float(v) for v in s.split(",")])
    return np.array(pts, dtype=float)


def draw(ax, X, labels, centroids, title, prev_centroids=None, new_centroids=None):
    """画一张图：点按簇着色，中心用 X 标出，中心移动用箭头表示。"""
    for j in range(len(centroids)):
        mask = labels == j
        ax.scatter(X[mask, 0], X[mask, 1], s=26, alpha=0.7,
                   color=COLORS[j % len(COLORS)], edgecolor="white", linewidth=0.4)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=170,
               c=[COLORS[j % len(COLORS)] for j in range(len(centroids))],
               edgecolor="black", linewidth=1.3, zorder=6)
    if prev_centroids is not None and new_centroids is not None:
        for a, b in zip(prev_centroids, new_centroids):
            ax.annotate("", xy=b, xytext=a,
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.6, alpha=0.8))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.25)


def main():
    argv = sys.argv[1:]
    # 默认初始中心：故意选得"不难也不易"，方便观察中心怎么移动
    default = [[0.0, -3.0], [0.0, 4.0], [5.0, 0.0]]
    centroids = parse_centers(argv, default)
    k = len(centroids)

    X, y_true = dg.make_synthetic_data(
        dg.TRUE_MEANS, dg.TRUE_COVS, dg.TRUE_WEIGHTS, n_samples=90, seed=7
    )

    print(f"数据: {X.shape[0]} 个点, {X.shape[1]} 维")
    print(f"K = {k}，初始中心:\n{centroids}\n")

    history = []
    for step in range(1, 4):
        labels = assign(X, centroids)
        J = compute_inertia(X, labels, centroids)
        counts = [int((labels == j).sum()) for j in range(k)]
        print(f"第 {step} 轮 分配后: J = {J:8.2f}   各簇点数 {counts}")

        new_centroids = update_centroids(X, labels, k)
        shift = np.linalg.norm(new_centroids - centroids, axis=1)
        print(f"          中心移动距离: {np.round(shift, 3)}   (最大 {shift.max():.3f})")

        history.append((labels, centroids.copy(), new_centroids.copy(), J))
        centroids = new_centroids
        if shift.max() < 1e-6:
            print("          中心几乎不再移动 → 可以停止")
            break

    # 画出来：初始状态 + 每轮分配/更新
    n_panels = len(history) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.4))
    axes = np.atleast_1d(axes)

    init = parse_centers(argv, default)
    draw(axes[0], X, np.zeros(len(X), dtype=int), init, "初始：给定了 K 个中心\n（点还没分配）")
    for i, (labels, old_c, new_c, J) in enumerate(history):
        draw(axes[i + 1], X, labels, new_c,
             f"第 {i + 1} 轮：分配后 J={J:.0f}\n箭头=中心往哪移动",
             prev_centroids=old_c, new_centroids=new_c)
    fig.suptitle("k-means 的两步：先按最近中心分配，再把中心移到簇的均值处", y=1.02)
    fig.savefig(FIG / "fig05_kmeans_steps.png", bbox_inches="tight")
    plt.close(fig)
    print("\n[fig] figures/fig05_kmeans_steps.png")


if __name__ == "__main__":
    main()
