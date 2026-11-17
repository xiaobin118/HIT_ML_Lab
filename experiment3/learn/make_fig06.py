"""生成讲义 06 的插图。

三件事：
  1. Jensen 不等式的几何含义：ln(加权平均) >= 加权平均的 ln
  2. EM 的两条曲线：对数似然（真实目标）与下界 ELBO
  3. EM 迭代过程中两个高斯分量的移动

运行方式：cd learn && python make_fig06.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_gen as dg  # noqa: E402
from practice05 import log_normal_1d  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
C1, C2, C3, C4 = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

TRUE_MU = np.array([0.0, 3.0])
TRUE_VAR = np.array([0.25, 4.0])
TRUE_W = np.array([0.5, 0.5])


# ---------------------------------------------------------------------------
# 一个最小的一维 EM 实现（讲义 06 用它生成所有数字）
# ---------------------------------------------------------------------------


def log_resp(x, mu, var, w):
    """返回 (对数责任度矩阵, 逐样本对数似然, 总对数似然)。"""
    K = len(mu)
    lp = np.column_stack([np.log(w[k]) + log_normal_1d(x, mu[k], var[k]) for k in range(K)])
    m = lp.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(lp - m).sum(axis=1))
    return lp - lse[:, None], lse


def elbo(x, log_r, mu, var, w):
    """用给定的（任意）责任度 q 计算下界。"""
    K = len(mu)
    lp = np.column_stack([np.log(w[k]) + log_normal_1d(x, mu[k], var[k]) for k in range(K)])
    r = np.exp(log_r)
    ent = np.where(r > 0, r * np.log(np.where(r > 0, r, 1.0)), 0.0)
    return float((r * lp).sum() - ent.sum())


def run_em(x, mu, var, w, n_iter=40):
    """跑 EM，记录每轮的 logL、参数、以及（用最终参数算的）下界。"""
    mu, var, w = np.array(mu, float), np.array(var, float), np.array(w, float)
    hist = {"loglik": [], "mu": [], "var": [], "w": []}
    for _ in range(n_iter):
        log_r, lse = log_resp(x, mu, var, w)
        hist["loglik"].append(float(lse.sum()))
        hist["mu"].append(mu.copy())
        hist["var"].append(var.copy())
        hist["w"].append(w.copy())
        r = np.exp(log_r)
        Nk = r.sum(axis=0)
        w = Nk / len(x)
        mu = (r * x[:, None]).sum(axis=0) / Nk
        var = (r * (x[:, None] - mu) ** 2).sum(axis=0) / Nk
    hist["final"] = (mu, var, w)
    return hist


def get_data(n=2000, seed=3):
    X, y = dg.make_synthetic_data(TRUE_MU[:, None], TRUE_VAR[:, None, None],
                                  TRUE_W, n_samples=n, seed=seed)
    return X[:, 0], y


def fig_jensen():
    """Jensen 不等式：ln 是凹函数，所以 ln(E[·]) >= E[ln(·)]。"""
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    xs = np.linspace(0.25, 3.2, 400)
    ax.plot(xs, np.log(xs), color=C1, lw=2.4, label=r"$y=\ln z$（凹函数）")

    z1, z2, t = 0.5, 2.6, 0.55
    zbar = t * z1 + (1 - t) * z2
    lz1, lz2 = np.log(z1), np.log(z2)
    lzbar = t * lz1 + (1 - t) * lz2

    ax.scatter([z1, z2], [lz1, lz2], s=80, color=C4, zorder=5)
    ax.plot([z1, z2], [lz1, lz2], color=C4, lw=2, ls="--",
            label="两点连线（弦）")
    ax.scatter([zbar], [np.log(zbar)], s=90, color=C2, zorder=6)
    ax.scatter([zbar], [lzbar], s=90, color=C4, marker="s", zorder=6)
    ax.annotate(r"$\ln(\mathbb{E}[z])$", xy=(zbar, np.log(zbar)),
                xytext=(zbar + 0.12, np.log(zbar) + 0.10), fontsize=11, color=C2)
    ax.annotate(r"$\mathbb{E}[\ln z]$", xy=(zbar, lzbar),
                xytext=(zbar + 0.12, lzbar - 0.22), fontsize=11, color=C4)
    ax.plot([zbar, zbar], [lzbar, np.log(zbar)], color="black", lw=1.6)
    ax.text(zbar + 0.04, (lzbar + np.log(zbar)) / 2, "差距 ≥ 0\n（就是 KL 散度）",
            fontsize=9, va="center")
    ax.text(z1 - 0.02, lz1 - 0.22, r"$z_1$", fontsize=11, color=C4, ha="center")
    ax.text(z2 - 0.02, lz2 - 0.22, r"$z_2$", fontsize=11, color=C4, ha="center")
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_title(r"Jensen 不等式：$\ln(\sum_k q_k z_k)\ \geq\ \sum_k q_k\ln z_k$", pad=10)
    ax.legend(fontsize=9, loc="lower right")
    fig.savefig(FIG / "fig15_jensen.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig15_jensen.png")
    print(f"  ln(E[z]) = {np.log(zbar):.6f}   E[ln z] = {lzbar:.6f}   差 = {np.log(zbar)-lzbar:.6f}")


def fig_likelihood_and_bound():
    """对数似然曲线与下界曲线。"""
    x, y = get_data()
    hist = run_em(x, [2.0, 5.0], [1.0, 1.0], [0.5, 0.5], n_iter=40)
    L = np.array(hist["loglik"])
    mu_f, var_f, w_f = hist["final"]

    # 用【最终】参数算责任度作为固定的 q，于是得到一条全程有效的下界
    log_r_final, _ = log_resp(x, mu_f, var_f, w_f)
    bound = np.array([elbo(x, log_r_final, hist["mu"][i], hist["var"][i], hist["w"][i])
                      for i in range(len(L))])

    print(f"  logL 首轮 {L[0]:.3f} -> 末轮 {L[-1]:.3f}")
    print(f"  下界 首轮 {bound[0]:.3f} -> 末轮 {bound[-1]:.3f}")
    print(f"  logL 单调: {bool(np.all(np.diff(L) >= -1e-9))}   下界单调: {bool(np.all(np.diff(bound) >= -1e-9))}")

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    iters = np.arange(1, len(L) + 1)
    ax.plot(iters, L, "o-", color=C1, ms=4, lw=2,
            label="对数似然 $\\ln p(X\\mid\\theta)$（EM 每轮真实达到的值）")
    ax.plot(iters, bound, "s--", color=C4, ms=4, lw=2,
            label="下界 $\\mathcal{L}$（用收敛后的 q 固定算）")
    ax.fill_between(iters, bound, L, color=C3, alpha=0.15, label="差距 = KL 散度")
    ax.annotate("两者在这里重合\n（q 已是后验，KL=0）",
                xy=(len(L), L[-1]), xytext=(len(L) - 13, L[-1] - 2600),
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.set_xlabel("EM 迭代轮数")
    ax.set_ylabel("对数似然")
    ax.set_title("EM 的两条曲线：下界单调上升，最终追上对数似然", pad=10)
    ax.legend(fontsize=9, loc="lower right")
    fig.savefig(FIG / "fig16_likelihood_bound.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig16_likelihood_bound.png")


def fig_component_movement():
    """EM 迭代中两个高斯分量的移动（每 2 轮画一次）。"""
    x, y = get_data(n=800, seed=3)
    hist = run_em(x, [2.0, 5.0], [1.0, 1.0], [0.5, 0.5], n_iter=25)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), sharey=True)
    bins = np.linspace(-8, 14, 70)
    show = [0, 1, -1]
    titles = ["初始猜测", "第 2 轮", f"第 {len(hist['loglik'])} 轮（收敛）"]
    xs = np.linspace(-8, 14, 700)

    for ax, idx, title in zip(axes, show, titles):
        ax.hist(x, bins=bins, color="#CCCCCC", edgecolor="white")
        mu, var, w = hist["mu"][idx], hist["var"][idx], hist["w"][idx]
        scale = len(x) * (bins[1] - bins[0])
        for k, col in zip(range(2), (C1, C2)):
            ax.plot(xs, scale * w[k] * np.exp(log_normal_1d(xs, mu[k], var[k])),
                    color=col, lw=2.2, label=f"分量{k}: w={w[k]:.2f}\n  μ={mu[k]:.2f}, σ²={var[k]:.2f}")
        if idx == 0:
            for k, col in zip(range(2), (C1, C2)):
                ax.axvline(TRUE_MU[k], color=col, ls=":", lw=1.6,
                           label=f"真实 μ{k}={TRUE_MU[k]}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.legend(fontsize=7.5, loc="upper right")
    axes[0].set_ylabel("样本数")
    fig.suptitle("EM 把随机初始化的两个高斯逐步移动到真实位置", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig17_component_movement.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig17_component_movement.png")
    mu, var, w = hist["final"]
    print(f"  最终参数: mu={np.round(mu, 3).tolist()}  var={np.round(var, 3).tolist()}  w={np.round(w, 3).tolist()}")
    print(f"  真实参数: mu={TRUE_MU.tolist()}  var={TRUE_VAR.tolist()}  w={TRUE_W.tolist()}")


if __name__ == "__main__":
    fig_jensen()
    fig_likelihood_and_bound()
    fig_component_movement()
    print("\n讲义 06 的插图已生成到 learn/figures/")
