"""绘制讲义 01 的插图。

运行方式：cd learn && python make_fig01.py
输出到 learn/figures/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import data_gen as dg  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

COLORS = ["#4C72B0", "#DD8452", "#55A868"]
CLASSES = ["setosa", "versicolor", "virginica"]


def fig_1d_distribution():
    """一维：花瓣长度的三个品种分布，显示空白与重叠。"""
    X, y, _ = dg.load_dataset("iris_2d", standardize=False)
    v = X[:, 0]

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for j, (c, col) in enumerate(zip(CLASSES, COLORS)):
        vals = v[y == j]
        # 用抖动让点在一维上不重叠
        ax.scatter(vals, np.full(len(vals), j) + np.random.default_rng(0).normal(0, 0.06, len(vals)),
                   s=18, alpha=0.65, color=col, label=f"{c} (n={len(vals)})")
        ax.plot([vals.min(), vals.max()], [j - 0.22, j - 0.22], color=col, lw=3, solid_capstyle="butt")
        ax.text(vals.mean(), j + 0.22, f"均值 {vals.mean():.2f}", ha="center", fontsize=9, color=col)

    ax.axvspan(1.9, 3.0, color="green", alpha=0.10)
    ax.text(2.45, 2.42, "空白区 1.9 – 3.0\n两类之间无样本", ha="center", fontsize=9, color="green")
    ax.axvspan(4.5, 5.1, color="red", alpha=0.10)
    ax.text(4.8, -0.42, "重叠区 4.5 – 5.1\n两类都有样本", ha="center", fontsize=9, color="darkred")

    ax.set_yticks(range(3), CLASSES)
    ax.set_ylim(-0.75, 2.85)
    ax.set_xlabel("花瓣长度 (cm)")
    ax.set_title("一维视角：仅用花瓣长度区分三个品种", pad=14)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(FIG / "fig01_1d_petal_length.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig01_1d_petal_length.png")


def fig_2d_clusters():
    """二维：三个簇的散点图，setosa 孤立，另两类相接。"""
    X, y, _ = dg.load_dataset("iris_2d", standardize=False)

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for j, (c, col) in enumerate(zip(CLASSES, COLORS)):
        ax.scatter(X[y == j, 0], X[y == j, 1], s=32, alpha=0.75, color=col,
                   edgecolor="white", linewidth=0.6, label=c)
        ax.annotate(c, X[y == j].mean(axis=0), fontsize=11, weight="bold", color=col,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.85))
    ax.set_xlabel("花瓣长度 (cm)")
    ax.set_ylabel("花瓣宽度 (cm)")
    ax.set_title("二维视角：三个簇（横轴花瓣长、纵轴花瓣宽）")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(FIG / "fig02_2d_clusters.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig02_2d_clusters.png")


def fig_scale_matters():
    """量纲问题：某一维数值范围远大于其他维时，距离被它垄断。"""
    X, y, _ = dg.load_dataset("iris_4d", standardize=False)
    # 注意：整体乘一个常数不会改变各维占比（约分掉了），必须只放大某一维。
    X_scaled = X.copy()
    X_scaled[:, 0] = X_scaled[:, 0] * 1000.0   # 只把第 0 列换成小单位

    def share_per_dim(data):
        """每一维对全部两两距离平方和的贡献占比。"""
        per_dim = np.array([
            ((data[:, None, j] - data[None, :, j]) ** 2).sum() for j in range(data.shape[1])
        ])
        return per_dim / per_dim.sum()

    print("各维对总距离的贡献占比（4 个特征，看的是谁在主导距离）:")
    for tag, data in (("原始单位", X),
                      ("只把第 0 列 x1000", X_scaled),
                      ("标准化后", dg._standardize(X))):
        shares = share_per_dim(data)
        print(f"  {tag:<20}" + "  ".join(f"{s:.3f}" for s in shares))

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    for ax, (title, data) in zip(axes, [
        ("原始单位（各列数值范围相近）", X),
        ("只把第 0 列乘 1000", X_scaled),
        ("标准化后（各维公平）", dg._standardize(X)),
    ]):
        shares = share_per_dim(data)
        ax.bar([f"特征{j}" for j in range(len(shares))], shares,
               color=[COLORS[i % 3] for i in range(len(shares))])
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("该维对总距离的贡献占比")
        for i, s in enumerate(shares):
            ax.text(i, s + 0.02, f"{s:.2f}", ha="center", fontsize=8)
    fig.suptitle("量纲问题：某一维的数值范围决定它能否垄断距离", y=1.02)
    fig.savefig(FIG / "fig03_scale_matters.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig03_scale_matters.png")


def fig_euclidean():
    """几何示意：二维欧氏距离就是直角三角形的斜边。"""
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    p, q = np.array([1.4, 0.2]), np.array([4.5, 1.3])
    ax.scatter(*p, s=90, color=COLORS[0], zorder=5)
    ax.scatter(*q, s=90, color=COLORS[1], zorder=5)
    ax.plot([p[0], q[0]], [p[1], q[1]], color="black", lw=2, zorder=4)
    ax.plot([p[0], q[0]], [p[1], p[1]], color="gray", ls="--", lw=1.4)
    ax.plot([q[0], q[0]], [p[1], q[1]], color="gray", ls="--", lw=1.4)
    ax.text(p[0] - 0.3, p[1] - 0.28, "x = (1.4, 0.2)", fontsize=10, color=COLORS[0])
    ax.text(q[0] - 1.75, q[1] + 0.30, "z = (4.5, 1.3)", fontsize=10, color=COLORS[1])
    ax.text((p[0] + q[0]) / 2, p[1] - 0.22, "|x1 - z1| = 3.1", ha="center", fontsize=10, color="gray")
    ax.text(q[0] + 0.12, (p[1] + q[1]) / 2, "|x2 - z2|\n= 1.1", fontsize=10, color="gray")
    ax.text(0.75, 2.02, "d = √(3.1² + 1.1²) ≈ 3.29", fontsize=10.5, ha="left", va="top")
    ax.set_xlim(0.5, 6.0)
    ax.set_ylim(-1.0, 2.4)
    ax.set_xlabel("花瓣长度")
    ax.set_ylabel("花瓣宽度")
    ax.set_title("欧氏距离 = 勾股定理往高维推广", pad=10)
    ax.set_aspect("equal")
    fig.savefig(FIG / "fig04_euclidean.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig04_euclidean.png")


if __name__ == "__main__":
    fig_1d_distribution()
    fig_2d_clusters()
    fig_scale_matters()
    fig_euclidean()
    print("\n全部插图已生成到 learn/figures/")
