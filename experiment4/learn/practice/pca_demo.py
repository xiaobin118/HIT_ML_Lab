"""PCA 的最小演示：二维相关数据 → 两个主方向。

运行：python pca_demo.py
输出：pca_demo.png，以及特征值、方差解释比例、主方向。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    rng = np.random.default_rng(0)
    mean = np.array([2.0, 1.0])
    cov = np.array([[3.0, 2.0],
                    [2.0, 2.0]])
    X = rng.multivariate_normal(mean, cov, size=800)   # (N, 2)

    # 1. 中心化
    Xc = X - X.mean(axis=0, keepdims=True)

    # 2. 协方差矩阵（无偏估计）
    C = Xc.T @ Xc / (len(X) - 1)

    # 3. 特征分解，按特征值从大到小排序
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # 特征向量的符号可以整体取反；固定一个方向方便复现
    for i in range(2):
        j = int(np.argmax(np.abs(eigvecs[:, i])))
        if eigvecs[j, i] < 0:
            eigvecs[:, i] *= -1

    ratio = eigvals / eigvals.sum()
    print("样本均值:", X.mean(axis=0))
    print("样本协方差:\n", C)
    print("特征值（各主方向的方差）:", eigvals)
    print("方差解释比例:", ratio)
    print("第一主方向 w1:", eigvecs[:, 0])
    print("第二主方向 w2:", eigvecs[:, 1])

    # 4. 投影：把数据中心化后投到前 k 个方向
    Z = Xc @ eigvecs
    print("投影后坐标 Z 的形状:", Z.shape)
    print("Z 的样本方差:", Z.var(axis=0, ddof=1))

    # 画图
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.35, color="#4c72b0", label="样本")

    center = X.mean(axis=0)
    colors = ["#d62728", "#2ca02c"]
    for i in range(2):
        v = eigvecs[:, i] * 2.2 * np.sqrt(eigvals[i])
        ax.annotate("", xy=center + v, xytext=center,
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=2.2))
        ax.text(*(center + v * 1.08), f"PC{i + 1}\n{ratio[i]:.1%}",
                color=colors[i], fontsize=9, ha="center")

    # 把 5 个点投影到 PC1，画虚线表示“丢掉 PC2 的误差”
    idx = np.linspace(0, len(X) - 1, 5, dtype=int)
    for i in idx:
        proj = center + (Xc[i] @ eigvecs[:, 0]) * eigvecs[:, 0]
        ax.plot([X[i, 0], proj[0]], [X[i, 1], proj[1]],
                color="gray", linestyle="--", linewidth=0.9)
        ax.scatter(*proj, color="gray", s=18, zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"PCA：第一主方向解释 {ratio[0]:.1%} 的方差")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "pca_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("图已保存到", out)


if __name__ == "__main__":
    main()
