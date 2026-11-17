"""生成讲义 07 的插图。

  1. 对数似然与 BIC 随 K 的变化：BIC 什么时候选对、什么时候选错
  2. 奇异协方差的后果：一个方差塌到 0 的分量
  3. 高维的诅咒：每分量样本数 / 协方差参数个数 的比值

运行方式：cd learn && python make_fig07.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "code"))
sys.path.insert(0, str(_HERE))
import data_gen as dg  # noqa: E402
from practice05 import log_normal_1d  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

FIG = _HERE / "figures"
FIG.mkdir(exist_ok=True)
C1, C2, C3, C4 = "#4C72B0", "#DD8452", "#55A868", "#C44E52"


# ---------------------------------------------------------------------------
# 一个最小但完整的 GMM-EM（讲义 07 用它跑真实数据，也顺便当作参考实现）
# 真实项目里你自己写的 gmm.py 应当比这个更完善（正则化、空分量处理等）
# ---------------------------------------------------------------------------


def log_gauss_1d_or_nd(X, mu, cov):
    """对数高斯密度，X 形状 (N, D)，cov 形状 (D, D)。用 Cholesky。"""
    X = np.atleast_2d(X)
    D = X.shape[1]
    L = np.linalg.cholesky(cov)
    diff = X - mu
    sol = np.linalg.solve(L, diff.T)
    maha = (sol ** 2).sum(axis=0)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (D * np.log(2 * np.pi) + logdet + maha)


def gmm_em(X, K, n_iter=200, tol=1e-6, reg=1e-6, seed=0):
    """最小可用的 GMM-EM。返回 (参数, 责任度, 对数似然序列)。"""
    N, D = X.shape
    rng = np.random.default_rng(seed)
    # 用随机挑点初始化：简单，真实项目里建议用 k-means ++ 多次重启
    mu = X[rng.choice(N, K, replace=False)].copy()
    cov = np.array([np.cov(X.T, bias=True)] * K)
    w = np.full(K, 1.0 / K)
    loglik_hist = []

    def regularize(c):
        return c + reg * np.trace(c) / D * np.eye(D)

    cov = np.array([regularize(c) for c in cov])
    for _ in range(n_iter):
        lp = np.column_stack([np.log(w[k]) + log_gauss_1d_or_nd(X, mu[k], cov[k])
                              for k in range(K)])
        m = lp.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(lp - m).sum(axis=1))
        loglik_hist.append(float(lse.sum()))
        if len(loglik_hist) > 1:
            if abs(loglik_hist[-1] - loglik_hist[-2]) / abs(loglik_hist[-2]) < tol:
                break
        lr = lp - lse[:, None]
        r = np.exp(lr)
        Nk = r.sum(axis=0)
        Nk = np.maximum(Nk, 1e-10)
        w = Nk / N
        mu = (r.T @ X) / Nk[:, None]
        for k in range(K):
            d = X - mu[k]
            cov[k] = regularize((r[:, k, None] * d).T @ d / Nk[k])
    # 收尾一次 E 步，保证 resp 与参数一致
    lp = np.column_stack([np.log(w[k]) + log_gauss_1d_or_nd(X, mu[k], cov[k])
                          for k in range(K)])
    m = lp.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(lp - m).sum(axis=1))
    lr = lp - lse[:, None]
    return {"w": w, "mu": mu, "cov": cov}, np.exp(lr), loglik_hist


def bic(loglik, N, K, D):
    """BIC = p*ln(N) - 2*ln(L)，p 为满协方差 GMM 的自由参数个数。"""
    p = K * (D + D * (D + 1) // 2 + 1) - 1
    return p * np.log(N) - 2.0 * loglik


# ---------------------------------------------------------------------------


def fig_bic_selection():
    """四个真实数据集上的对数似然与 BIC 曲线。"""
    cases = [
        ("iris_4d", 3), ("seeds_7d", 3), ("wdbc", 2), ("wine", 3),
    ]
    ks = [1, 2, 3, 4, 5]

    fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.6))
    for col, (name, k_true) in enumerate(cases):
        X, y, _ = dg.load_dataset(name)
        N, D = X.shape
        lls, bics, aris = [], [], []
        for K in ks:
            if K > min(N, 6):
                lls.append(np.nan); bics.append(np.nan); aris.append(np.nan); continue
            params, resp, hist = gmm_em(X, K, seed=0)
            ll = hist[-1]
            lls.append(ll)
            bics.append(bic(ll, N, K, D))
            from metrics import adjusted_rand_index
            aris.append(adjusted_rand_index(y, resp.argmax(axis=1)))
        lls, bics, aris = np.array(lls), np.array(bics), np.array(aris)
        k_bic = ks[int(np.nanargmin(bics))]
        k_ari = ks[int(np.nanargmax(aris))]

        ax = axes[0, col]
        ax.plot(ks, lls, "o-", color=C1, lw=1.8, ms=5)
        ax.axvline(k_true, color=C3, ls="--", lw=1.5)
        ax.set_title(f"{name}  (N={N}, D={D}, 真实 K={k_true})", fontsize=10)
        ax.set_xlabel("K")
        if col == 0:
            ax.set_ylabel("对数似然（越高越好）")
        ax.text(0.5, 0.06, "对数似然永远随 K 上升", transform=ax.transAxes,
                ha="center", fontsize=8.5, color=C1)

        ax = axes[1, col]
        col_min = C3 if k_bic == k_true else C4
        ax.plot(ks, bics, "s-", color=col_min, lw=1.8, ms=5)
        ax.axvline(k_true, color=C3, ls="--", lw=1.5, label=f"真实 K={k_true}")
        ax.scatter([k_bic], [bics[ks.index(k_bic)]], s=170, facecolor="none",
                   edgecolor="black", linewidth=2, zorder=6)
        verdict = "选对" if k_bic == k_true else f"选错（选了 {k_bic}）"
        ax.set_title(f"BIC 最小处 K={k_bic} → {verdict}", fontsize=10,
                     color="black" if k_bic == k_true else C4)
        ax.set_xlabel("K")
        if col == 0:
            ax.set_ylabel("BIC（越低越好）")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("对数似然随 K 单调上升，BIC 加了罚项后才有极小值——但它不一定选对", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig18_bic_selection.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig18_bic_selection.png")


def fig_singularity():
    """奇异性问题：一个分量塌到单个点上，密度发散。"""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.5, 300), rng.normal(5, 0.8, 300)])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    bins = np.linspace(-3, 9, 70)

    # 左：正常的两分量解
    ax = axes[0]
    ax.hist(x, bins=bins, color="#CCCCCC", edgecolor="white")
    xs = np.linspace(-3, 9, 700)
    scale = len(x) * (bins[1] - bins[0])
    for mu, var, w, col in [(0.0, 0.25, 0.5, C1), (5.0, 0.64, 0.5, C2)]:
        ax.plot(xs, scale * w * np.exp(log_normal_1d(xs, mu, var)), color=col, lw=2.2,
                label=f"$\\mu={mu}, \\sigma^2={var}, \\pi={w}$")
    ax.set_title("正常解：两个分量各自覆盖一团数据", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("样本数")
    ax.legend(fontsize=9)

    # 右：塌陷的分量（均值落在某个样本上，方差极小）
    ax = axes[1]
    ax.hist(x, bins=bins, color="#CCCCCC", edgecolor="white")
    ax.plot(xs, scale * 0.5 * np.exp(log_normal_1d(xs, 0.0, 0.25)), color=C1, lw=2.2,
            label="分量0：正常")
    x0 = x[0]
    tiny_var = 1e-6
    spike = scale * 0.5 * np.exp(log_normal_1d(xs, x0, tiny_var))
    ax.plot(xs, np.minimum(spike, 400), color=C4, lw=2.2,
            label=f"分量1：塌到单点\n$\\mu={x0:.2f}, \\sigma^2=10^{{-6}}$")
    ax.axvline(x0, color=C4, ls=":", lw=1.6)
    ax.set_ylim(0, 380)
    dens_at_x0 = np.exp(log_normal_1d(np.array([x0]), x0, tiny_var))[0]
    ax.set_title(f"奇异解：密度在 $x={x0:.2f}$ 处达到 {dens_at_x0:.2e}", fontsize=10)
    ax.set_xlabel("x")
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("奇异性问题：某个分量把方差压到 0 就能让似然发散，这是退化而不是更好的解", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig19_singularity.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig19_singularity.png")
    print(f"  单个样本在 σ²=1e-6 处的密度: {dens_at_x0:.3e}")
    print(f"  它的对数密度: {log_normal_1d(np.array([x0]), x0, tiny_var)[0]:.3f}")


def fig_dimension_curse():
    """维度上升时，每个分量的样本数相对协方差参数个数急剧变少。"""
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    Ds = np.arange(2, 66, 2)
    for N, K, col, lab in [(150, 3, C1, "Iris：N=150, K=3"),
                           (210, 3, C2, "Seeds：N=210, K=3"),
                           (569, 2, C3, "WDBC：N=569, K=2"),
                           (5620, 10, C4, "Optdigits：N=5620, K=10")]:
        per_comp = N / K                       # 每个分量的样本数
        n_params = Ds * (Ds + 1) / 2 + Ds + 1  # 该分量要估的参数个数
        ax.plot(Ds, per_comp / n_params, "-", color=col, lw=2.2, label=lab)
    ax.axhline(1.0, color="black", ls="--", lw=1.5)
    ax.text(40, 1.15, "样本数 = 参数个数（低于这条线就欠定）", fontsize=9, ha="center")
    ax.axvline(64, color="gray", ls=":", lw=1.4)
    # 用坐标轴比例定位，避免在对数轴上取到极小的数据坐标导致字形渲染溢出
    ax.text(0.955, 0.06, "optdigits\n维数 64", transform=ax.transAxes,
            fontsize=8.5, va="bottom", ha="right", color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("特征维数 D")
    ax.set_ylabel("每分量样本数 / 每分量参数个数（对数轴）")
    ax.set_title("维度的诅咒：D 增大时协方差参数按 D² 增长，样本数跟不上", pad=10)
    ax.legend(fontsize=9)
    fig.savefig(FIG / "fig20_dimension_curse.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig20_dimension_curse.png")
    for D in (4, 13, 30, 64):
        print(f"  D={D:<3} 每分量参数个数（满协方差）= {int(D*(D+1)/2 + D + 1)}")


if __name__ == "__main__":
    fig_singularity()
    fig_dimension_curse()
    fig_bic_selection()
    print("\n讲义 07 的插图已生成到 learn/figures/")
