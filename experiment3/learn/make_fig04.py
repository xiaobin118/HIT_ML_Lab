"""生成讲义 04 的插图。

三件事：
  1. 一元高斯的形状：均值决定位置、标准差决定胖瘦，以及 68-95 规则
  2. 用真实数据看"一个高斯不够"：Iris 花瓣长度整体 vs 分品种
  3. 二维高斯的等密度线是椭圆：协方差矩阵决定椭圆的胖瘦和朝向

运行方式：cd learn && python make_fig04.py
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
from viz import ellipse_points  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
C1, C2, C3, C4 = "#4C72B0", "#DD8452", "#55A868", "#C44E52"


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def fig_gaussian_shape():
    """一元高斯的形状与 68-95 规则。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    x = np.linspace(-9, 9, 800)

    ax = axes[0]
    for mu, col, lab in [(-3, C1, r"$\mu=-3$"), (0, C2, r"$\mu=0$"), (3.5, C3, r"$\mu=3.5$")]:
        ax.plot(x, normal_pdf(x, mu, 1.0), color=col, lw=2, label=lab)
    ax.set_title(r"标准差固定 $\sigma=1$：均值决定位置", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("密度 p(x)")
    ax.legend(fontsize=9)

    ax = axes[1]
    for sd, col, lab in [(0.5, C1, r"$\sigma=0.5$"), (1.0, C2, r"$\sigma=1$"), (2.0, C3, r"$\sigma=2$")]:
        ax.plot(x, normal_pdf(x, 0, sd), color=col, lw=2, label=lab)
    ax.fill_between(x, 0, normal_pdf(x, 0, 1), where=(np.abs(x) <= 1),
                    color=C2, alpha=0.18)
    ax.annotate("距均值 1 个标准差内\n约占 68%", xy=(0, 0.24), xytext=(1.8, 0.30),
                fontsize=9, color=C2,
                arrowprops=dict(arrowstyle="->", color=C2, lw=1.2))
    ax.annotate("2 个标准差内约占 95%", xy=(-2, 0.06), xytext=(-8.5, 0.20),
                fontsize=9, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.set_title(r"均值固定 $\mu=0$：标准差决定胖瘦", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("密度 p(x)")
    ax.legend(fontsize=9)

    fig.suptitle(r"一元高斯 $\mathcal{N}(\mu,\sigma^2)$ 的两个参数各管一件事", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_gaussian_shape.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig10_gaussian_shape.png")


def fig_one_vs_three():
    """真实数据：整体用一个高斯 vs 分品种各用一个高斯。"""
    X, y, names = dg.load_dataset("iris_4d")
    pl = X[:, 2]                       # 花瓣长度
    groups = ["setosa", "versicolor", "virginica"]
    cols = [C1, C2, C3]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # 左：全部 150 朵，用一个高斯拟合
    ax = axes[0]
    ax.hist(pl, bins=24, density=True, color="#BBBBBB", edgecolor="white", alpha=0.85)
    xs = np.linspace(pl.min() - 0.5, pl.max() + 0.5, 500)
    ax.plot(xs, normal_pdf(xs, pl.mean(), pl.std()), color=C4, lw=2.4)
    ax.set_title(f"全部 150 朵用一个高斯\n"
                 f"$\\mu={pl.mean():.2f},\\ \\sigma={pl.std():.2f}$", fontsize=10)
    ax.set_xlabel("花瓣长度 (cm)")
    ax.set_ylabel("密度")
    ax.annotate("单个高斯预测：\n这里样本最多", xy=(pl.mean(), normal_pdf(pl.mean(), pl.mean(), pl.std())),
                xytext=(5.05, 0.52), fontsize=9, color=C4, ha="center",
                arrowprops=dict(arrowstyle="->", color=C4, lw=1.2))
    ax.annotate("2.0-3.0 之间\n几乎没有样本", xy=(2.5, 0.012), xytext=(5.05, 0.30),
                fontsize=9, color="gray", ha="center",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.set_ylim(0, 0.66)

    # 右：按品种分成三组，各用一个高斯
    ax = axes[1]
    for j, (g, c) in enumerate(zip(groups, cols)):
        v = pl[y == j]
        ax.hist(v, bins=14, density=True, color=c, edgecolor="white", alpha=0.55)
        xs = np.linspace(v.min() - 0.4, v.max() + 0.4, 300)
        ax.plot(xs, normal_pdf(xs, v.mean(), v.std()), color=c, lw=2.2,
                label=f"{g}: $\\mu={v.mean():.2f},\\ \\sigma={v.std():.2f}$")
    ax.set_title("分成三个品种，各用一个高斯", fontsize=10)
    ax.set_xlabel("花瓣长度 (cm)")
    ax.set_ylim(0, 6.2)
    ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("一个高斯不够用：真实数据可能是几个高斯的叠加", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig11_one_vs_three_gaussians.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig11_one_vs_three_gaussians.png")
    print(f"  三个品种的花瓣长度: " + "  ".join(
        f"{g}({pl[y==j].mean():.2f}±{pl[y==j].std():.2f})" for j, g in enumerate(groups)))


def fig_2d_ellipse():
    """二维高斯：协方差矩阵决定等密度椭圆的胖瘦与朝向。"""
    covs = [
        (np.array([[1.0, 0.0], [0.0, 1.0]]), "各向同性\n(圆)"),
        (np.array([[2.5, 0.0], [0.0, 0.3]]), "横躺\n(长轴沿 x)"),
        (np.array([[1.2, 0.9], [0.9, 1.2]]), "倾斜\n(主轴 45°)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    rng = np.random.default_rng(0)
    for ax, (cov, title) in zip(axes, covs):
        pts = rng.multivariate_normal([0, 0], cov, size=400)
        ax.scatter(pts[:, 0], pts[:, 1], s=9, alpha=0.35, color=C1, edgecolor="none")
        for r, ls in [(1.0, "-"), (2.0, "--")]:
            e = ellipse_points([0, 0], cov, n_std=r)
            ax.plot(e[:, 0], e[:, 1], color=C4, lw=1.8, ls=ls,
                    label=f"{r:g}σ 等密度线" if ls == "-" else None)
        vals, vecs = np.linalg.eigh(cov)
        ax.annotate("", xy=vecs[:, -1] * np.sqrt(vals[-1]) * 1.9, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2.0))
        ax.set_title(f"{title}\n" + r"$\Sigma=$" + f"{cov.tolist()}".replace(" ", ""), fontsize=9.5)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(fontsize=8, loc="upper left")
        print(f"  {title.replace(chr(10), ' ')}: 特征值 {np.round(vals, 3)} 长短轴比 {vals[-1]/vals[0]:.2f}")

    fig.suptitle(r"二维高斯：均值决定椭圆中心，协方差矩阵决定椭圆的胖瘦与朝向", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig12_2d_ellipse.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig12_2d_ellipse.png")


if __name__ == "__main__":
    fig_gaussian_shape()
    fig_one_vs_three()
    fig_2d_ellipse()
    print("\n讲义 04 的插图已生成到 learn/figures/")
