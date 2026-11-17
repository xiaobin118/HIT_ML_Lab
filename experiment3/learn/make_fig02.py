"""绘制讲义 02 的插图：Voronoi 决策边界。

k-means 的"分配给最近的中心"在几何上等价于用一组直线把平面切开，
每条线是相邻两个中心的垂直平分线。这张图把这件事画出来。

运行方式：cd learn && python make_fig02.py
输出到 learn/figures/
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
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def nearest_centroid(X, centroids):
    """每个点归给最近的中心（这就是 k-means 的分配步）。"""
    diff = X[:, None, :] - centroids[None, :, :]
    return (diff ** 2).sum(axis=2).argmin(axis=1)


def fig_voronoi():
    """在平面上密集撒点，按最近中心着色，边界就自然显现出来。"""
    X, y_true = dg.make_synthetic_data(
        dg.TRUE_MEANS, dg.TRUE_COVS, dg.TRUE_WEIGHTS, n_samples=90, seed=7
    )
    centroids = X[[11, 71, 88]].copy()          # 一组能暴露局部最优的初始中心
    for _ in range(100):                         # 走到收敛
        labels = nearest_centroid(X, centroids)
        new = np.array([X[labels == j].mean(axis=0) for j in range(3)])
        if np.linalg.norm(new - centroids, axis=1).max() < 1e-12:
            centroids = new
            break
        centroids = new
    labels = nearest_centroid(X, centroids)

    # 造一张密集网格，按最近中心着色，形成 Voronoi 分区
    pad = 1.5
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 420)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 380)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
    zone = nearest_centroid(grid, centroids).reshape(GX.shape)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.contourf(GX, GY, zone, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=[COLORS[j] for j in range(3)], alpha=0.13)
    ax.contour(GX, GY, zone, levels=[0.5, 1.5], colors="black", linewidths=1.6)

    for j in range(3):
        mask = labels == j
        ax.scatter(X[mask, 0], X[mask, 1], s=30, color=COLORS[j],
                   edgecolor="white", linewidth=0.5, zorder=4)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=210, c="black", zorder=6,
               label="收敛后的中心")

    # 标出真实簇中心，方便看出哪里分错了
    for j in range(3):
        c = X[y_true == j].mean(axis=0)
        ax.scatter(*c, marker="P", s=150, facecolor="none", edgecolor="black",
                   linewidth=1.6, zorder=7,
                   label="真实簇中心" if j == 0 else None)

    # 画一条决策边界并标注它垂直于中心连线
    c0, c1 = centroids[0], centroids[1]
    mid = (c0 + c1) / 2
    d = c1 - c0
    n = np.array([-d[1], d[0]])
    n = n / np.linalg.norm(n) * 1.6
    ax.plot([mid[0] - n[0], mid[0] + n[0]], [mid[1] - n[1], mid[1] + n[1]],
            color="red", lw=2.4, zorder=5)
    ax.annotate("决策边界 = 两个中心的垂直平分线\n（线上的点到两边中心等距）",
                xy=mid, xytext=(0.5, 0.955), textcoords="axes fraction",
                ha="center", va="top", fontsize=9, color="red",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="red", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.4,
                                connectionstyle="arc3,rad=0.25"))

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("“分配给最近的中心” = 用垂直平分线把平面切成 K 块", pad=12)
    ax.legend(fontsize=9, loc="upper left")
    fig.savefig(FIG / "fig06_voronoi.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig06_voronoi.png")
    print(f"  最终中心: {np.round(centroids, 3).tolist()}")


if __name__ == "__main__":
    fig_voronoi()


def fig_convergence_compare():
    """好初始化 vs 坏初始化的 J 下降曲线，说明"都收敛但落点不同"。"""
    from demo02_kmeans_steps import assign, compute_inertia, update_centroids

    X, y_true = dg.make_synthetic_data(
        dg.TRUE_MEANS, dg.TRUE_COVS, dg.TRUE_WEIGHTS, n_samples=90, seed=7
    )

    def run(init, max_iter=50):
        C = np.array(init, dtype=float)
        k = len(C)
        curve = []
        for _ in range(max_iter):
            labels = assign(X, C)
            curve.append(compute_inertia(X, labels, C))
            newC = update_centroids(X, labels, k)
            if np.linalg.norm(newC - C, axis=1).max() < 1e-12:
                break
            C = newC
        return curve

    good = run([[0, -3], [0, 4], [5, 0]])
    bad = run(X[[11, 71, 88]].tolist())          # 两个初始点落在同一真实簇

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(range(1, len(good) + 1), good, "o-", lw=1.8, ms=5,
            label=f"好初始化（三个中心分散）→ J={good[-1]:.1f}")
    ax.plot(range(1, len(bad) + 1), bad, "s-", lw=1.8, ms=5, color="#C44E52",
            label=f"坏初始化（两个中心同簇）→ J={bad[-1]:.1f}")
    ax.axhline(good[-1], color="#4C72B0", ls=":", lw=1.2, alpha=0.7)
    ax.text(2.6, 700, "最终 J 相差 5.81 倍\n两者都满足两步最优，算法无法自行察觉",
            fontsize=10, color="#C44E52", ha="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#C44E52", alpha=0.9))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("簇内平方和 J")
    ax.set_title("两个初始化都正常收敛，但落进的坑深浅差 5.81 倍", pad=12)
    ax.legend(fontsize=9)
    fig.savefig(FIG / "fig07_local_optima.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig07_local_optima.png")
    print(f"  好初始化曲线: {[round(v, 1) for v in good]}")
    print(f"  坏初始化曲线: {[round(v, 1) for v in bad]}")


if __name__ == "__main__":
    fig_voronoi()
    fig_convergence_compare()
    print("\n讲义 02 的插图已生成到 learn/figures/")
