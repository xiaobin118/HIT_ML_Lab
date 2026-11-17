"""生成讲义 05 的插图。

核心对照：同一份数据上，k-means 的分界点与 Bayes 最优分界点差很远，
而"责任度"给出了两者之间的连续过渡。

运行方式：cd learn && python make_fig05.py
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
from kmeans import kmeans  # noqa: E402
from metrics import adjusted_rand_index  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
C1, C2, C3 = "#4C72B0", "#DD8452", "#55A868"

# 讲义 05 使用的两簇参数：均值分开、方差相差 8 倍
MU0, VAR0 = 0.0, 0.25
MU1, VAR1 = 3.0, 4.0
W0 = W1 = 0.5


def log_normal(x, mu, var):
    return -0.5 * np.log(2 * np.pi * var) - (x - mu) ** 2 / (2 * var)


def responsibility(x, mu0=MU0, var0=VAR0, mu1=MU1, var1=VAR1, w0=W0, w1=W1):
    """x 属于第 1 个簇的概率（责任度），数值稳定写法。"""
    l0 = np.log(w0) + log_normal(x, mu0, var0)
    l1 = np.log(w1) + log_normal(x, mu1, var1)
    m = np.maximum(l0, l1)
    e0, e1 = np.exp(l0 - m), np.exp(l1 - m)
    return e1 / (e0 + e1)


def bayes_boundaries():
    """返回责任度等于 0.5 的全部 x（即 l0 = l1 的解）。

    注意这里有【两个】解。因为两个分量的方差不同，对数密度之差是一个开口
    向下的抛物线：它在中间为正（簇 0 占优），两侧为负（簇 1 占优）。所以
    Bayes 最优划分是"簇 1 当 x 落在两侧的尾巴里，簇 0 当 x 在中间"，
    而不是一个分界点把数轴切成两半。

    返回 (左分界, 右分界)，升序。
    """
    f = lambda x: (np.log(W0) + log_normal(x, MU0, VAR0)) - (np.log(W1) + log_normal(x, MU1, VAR1))
    # 粗扫找到两个变号区间，再各自二分
    xs = np.linspace(-40.0, 40.0, 4001)
    vals = f(xs)
    roots = []
    for i in range(len(xs) - 1):
        if vals[i] * vals[i + 1] < 0:
            lo, hi = xs[i], xs[i + 1]
            for _ in range(200):
                mid = (lo + hi) / 2
                if f(mid) * f(lo) > 0:
                    lo = mid
                else:
                    hi = mid
            roots.append((lo + hi) / 2)
    return tuple(sorted(roots))


def get_data(n=2000, seed=3):
    means = np.array([[MU0], [MU1]])
    covs = np.array([[[VAR0]], [[VAR1]]])
    X, y = dg.make_synthetic_data(means, covs, np.array([W0, W1]), n_samples=n, seed=seed)
    return X[:, 0], y


def fig_boundary_compare():
    """k-means 的分界 vs Bayes 最优分界。"""
    x, y = get_data()
    C, lab, J, _ = kmeans(x[:, None], 2, seed=0)
    km_bound = (C[0, 0] + C[1, 0]) / 2
    b_left, b_right = bayes_boundaries()

    # Bayes 最优划分：责任度 > 0.5 的地方归簇 1（数轴上会分成两段）
    bayes_lab = (responsibility(x) > 0.5).astype(int)
    ari_km = adjusted_rand_index(y, lab)
    ari_bayes = adjusted_rand_index(y, bayes_lab)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.3]})

    # 上：数据的真实分布（按真实簇着色）
    ax = axes[0]
    bins = np.linspace(-8, 14, 90)
    ax.hist(x[y == 0], bins=bins, color=C1, alpha=0.65, label=f"真实簇 0：$\\mathcal{{N}}({MU0}, {VAR0})$（紧）")
    ax.hist(x[y == 1], bins=bins, color=C2, alpha=0.65, label=f"真实簇 1：$\\mathcal{{N}}({MU1}, {VAR1})$（散）")
    xs = np.linspace(-8, 14, 800)
    ax.plot(xs, np.exp(log_normal(xs, MU0, VAR0)) * len(x) * (bins[1] - bins[0]) * W0,
            color=C1, lw=2)
    ax.plot(xs, np.exp(log_normal(xs, MU1, VAR1)) * len(x) * (bins[1] - bins[0]) * W1,
            color=C2, lw=2)
    ax.axvline(km_bound, color="black", ls="--", lw=2,
               label=f"k-means 分界 {km_bound:.2f}（ARI={ari_km:.3f}）")
    ax.axvline(b_left, color=C3, ls="-", lw=2,
               label=f"Bayes 分界 {b_left:.2f}（两条，ARI={ari_bayes:.3f}）")
    ax.axvline(b_right, color=C3, ls="-", lw=2)
    ax.set_ylabel("样本数")
    ax.set_title("k-means 只能切一刀（2.19）；方差不同时最优解要切两刀（−1.37 与 0.97）", pad=10)
    ax.legend(fontsize=9, loc="upper right")

    # 下：责任度曲线
    ax = axes[1]
    ax.plot(xs, responsibility(xs), color=C3, lw=2.4, label="责任度 $r_1(x)$ = 该点属于簇 1 的概率")
    ax.axhline(0.5, color="gray", ls=":", lw=1.4)
    ax.axvline(km_bound, color="black", ls="--", lw=2, label=f"k-means 分界 {km_bound:.2f}")
    ax.axvline(b_left, color=C3, ls="-", lw=1.6, alpha=0.6)
    ax.axvline(b_right, color=C3, ls="-", lw=1.6, alpha=0.6)
    ax.fill_between(xs, 0, 1, where=(xs > km_bound), color=C2, alpha=0.08)
    ax.text(km_bound + 0.5, 0.28, "k-means：整片算簇 1", fontsize=9, color=C2)
    ax.text(-7.8, 0.62, "k-means：\n整片算簇 0", fontsize=9, color=C1)
    ax.text(-1.2, 0.06, "左边这截尾巴\n最优解其实归簇 1", fontsize=8.5, color=C3, ha="center")
    ax.annotate("责任度在 0.5 附近来回穿越：\n尾部高于 0.5，中间低于 0.5",
                xy=(0.2, 0.30), xytext=(4.6, 0.56), fontsize=9, color=C3,
                arrowprops=dict(arrowstyle="->", color=C3, lw=1.3))
    ax.set_xlabel("x")
    ax.set_ylabel("责任度 $r_1(x)$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG / "fig13_boundary_vs_bayes.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig13_boundary_vs_bayes.png")
    print(f"  k-means 分界 {km_bound:.4f}  ARI {ari_km:.4f}  各簇点数 {np.bincount(lab, minlength=2).tolist()}")
    print(f"  Bayes   分界 {b_left:.4f} 与 {b_right:.4f}  ARI {ari_bayes:.4f}  各簇点数 {np.bincount(bayes_lab, minlength=2).tolist()}")
    print(f"  真实簇点数 {np.bincount(y).tolist()}")


def fig_mixture_weights():
    """混合权重与各分量形状的关系：三个高斯按不同权重混合。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    xs = np.linspace(-6, 10, 800)
    comps = [(0.0, 0.6, C1, "分量0: $\\mu=0,\\sigma^2=0.6$"),
             (4.0, 1.5, C2, "分量1: $\\mu=4,\\sigma^2=1.5$"),
             (7.5, 0.4, C3, "分量2: $\\mu=7.5,\\sigma^2=0.4$")]

    for ax, (ws, title) in zip(axes, [
        ((1 / 3, 1 / 3, 1 / 3), "等权重 $\\pi=(1/3,1/3,1/3)$"),
        ((0.6, 0.25, 0.15), "不等权重 $\\pi=(0.6,0.25,0.15)$"),
    ]):
        total = np.zeros_like(xs)
        for (mu, var, col, lab), w in zip(comps, ws):
            y = w * np.exp(log_normal(xs, mu, var))
            ax.plot(xs, y, color=col, lw=1.6, ls="--", label=f"{lab}, $\\pi={w:.2f}$")
            total += y
        ax.plot(xs, total, color="black", lw=2.4, label="混合密度 $p(x)$")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("密度")
        ax.legend(fontsize=8)
    fig.suptitle("混合高斯：每个分量有自己的 $\\mu_k,\\sigma_k^2$，再按权重 $\\pi_k$ 叠加", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig14_mixture.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig14_mixture.png")


if __name__ == "__main__":
    fig_boundary_compare()
    fig_mixture_weights()
    print("\n讲义 05 的插图已生成到 learn/figures/")
