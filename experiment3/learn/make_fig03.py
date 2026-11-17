"""生成讲义 03 的插图。

三个实验：
  1. 肘部法：J 随 K 的变化，以及每个 K 相对上一个 K 的降幅
  2. 内部指标 vs 外部指标：轮廓系数与 ARI 在 Iris 上给出不同的 K
  3. 三种初始化策略的成功率对比

运行方式：cd learn && python make_fig03.py
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
from kmeans import assign_clusters, compute_inertia  # noqa: E402
from kmeans import kmeans_restarts  # noqa: E402
from metrics import adjusted_rand_index, silhouette_score  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
C1, C2, C3 = "#4C72B0", "#DD8452", "#55A868"


def kmeans_with_init(X, k, init, max_iter=100, tol=1e-6):
    """指定初始中心的 k-means（讲义 03 需要精确控制初始化以做对照）。"""
    C = np.array(init, dtype=float).copy()
    for t in range(1, max_iter + 1):
        lab = assign_clusters(X, C)
        cnt = np.bincount(lab, minlength=k)
        newC = np.zeros_like(C)
        for j in range(k):
            newC[j] = X[lab == j].mean(0) if cnt[j] > 0 else C[j]
        shift = np.linalg.norm(newC - C, axis=1).max()
        C = newC
        if shift < tol:
            break
    lab = assign_clusters(X, C)
    return C, lab, compute_inertia(X, lab, C), t


def sweep_k(X, y, kmax=9, n_init=10):
    """对 K=1..kmax 跑 k-means，返回 inertia / 轮廓系数 / ARI 三条曲线。

    同时返回每个 K 下所有重启的 inertia 范围（最好与最差）。这个范围很
    重要：K 越大局部最优越多，同样的 n_init 越难找到真正的最优解，
    只看最好值会高估聚类效果。
    """
    Js, sils, aris, lo, hi = [], [], [], [], []
    for K in range(1, kmax + 1):
        best, allJ = kmeans_restarts(X, K, n_init=n_init)
        Js.append(best["inertia"])
        lo.append(min(allJ))
        hi.append(max(allJ))
        if K == 1:
            sils.append(0.0)
            aris.append(0.0)
        else:
            sils.append(silhouette_score(X, best["labels"], sample_size=1500, seed=0))
            aris.append(adjusted_rand_index(y, best["labels"]))
    return np.array(Js), np.array(sils), np.array(aris), np.array(lo), np.array(hi)


def fig_elbow():
    """肘部法：J 的绝对曲线 + 每步降幅。"""
    Xs, ys = dg.make_synthetic_data(dg.TRUE_MEANS, dg.TRUE_COVS, dg.TRUE_WEIGHTS, 1500, seed=0)
    Xi, yi, _ = dg.load_dataset("iris_2d")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for col, (tag, X, y, ktrue) in enumerate([
        ("合成数据（真实 K=3）", Xs, ys, 3),
        ("Iris 花瓣两维（真实 K=3）", Xi, yi, 3),
    ]):
        Js, _, _, lo, hi = sweep_k(X, y)
        ks = np.arange(1, len(Js) + 1)
        ax = axes[0, col]
        # 阴影 = 该 K 下 10 次重启的 J 范围，说明解的稳定性
        ax.fill_between(ks, lo, hi, color=C1, alpha=0.18,
                        label="10 次重启的 J 范围")
        ax.plot(ks, Js, "o-", color=C1, lw=1.8, ms=5, label="最好的一次")
        ax.axvline(ktrue, color=C3, ls="--", lw=1.5, label=f"真实 K={ktrue}")
        ax.set_yscale("log")
        ax.set_xlabel("K")
        ax.set_ylabel("簇内平方和 J（对数轴）")
        ax.set_title(f"{tag}：J 随 K 单调下降", fontsize=10)
        ax.legend(fontsize=8)

        ax = axes[1, col]
        drop = (Js[:-1] - Js[1:]) / Js[:-1] * 100
        bars = ax.bar(ks[1:], drop, color=C2, width=0.62)
        ax.axvline(ktrue, color=C3, ls="--", lw=1.5)
        for b, v in zip(bars, drop):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
        ax.set_xlabel("K")
        ax.set_ylabel("相对上一个 K 的降幅 (%)")
        ax.set_title("看降幅拐点：哪一步之后收益骤减", fontsize=10)

    fig.suptitle("肘部法：J 永远在降，要看的不是绝对值而是降幅的拐点", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig08_elbow.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig08_elbow.png")


def fig_internal_vs_external():
    """轮廓系数（内部指标）与 ARI（外部指标）在 Iris 上给出不同的 K。"""
    Xi, yi, _ = dg.load_dataset("iris_2d")
    Js, sils, aris, _, _ = sweep_k(Xi, yi)
    ks = np.arange(1, len(Js) + 1)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    k_axis, sil_axis, ari_axis = ks[1:], sils[1:], aris[1:]
    ax.plot(k_axis, sil_axis, "o-", color=C1, lw=1.8, ms=6, label="轮廓系数（内部指标，无标签可用）")
    ax.plot(k_axis, ari_axis, "s-", color=C2, lw=1.8, ms=6, label="ARI（外部指标，需真实标签）")

    # 注意 argmax 必须作用在"去掉 K=1"之后的数组上，否则索引会偏移一位
    i_sil = int(np.argmax(sil_axis))
    i_ari = int(np.argmax(ari_axis))
    k_sil, k_ari = int(k_axis[i_sil]), int(k_axis[i_ari])
    ax.axvline(k_sil, color=C1, ls=":", lw=1.6)
    ax.axvline(k_ari, color=C2, ls=":", lw=1.6)
    ax.scatter([k_sil], [sil_axis[i_sil]], s=180, facecolor="none", edgecolor=C1, linewidth=2.2, zorder=5)
    ax.scatter([k_ari], [ari_axis[i_ari]], s=180, facecolor="none", edgecolor=C2, linewidth=2.2, zorder=5)
    ax.annotate(f"轮廓系数在此最大\nK={k_sil}（与真实 K=3 不符）", xy=(k_sil, sil_axis[i_sil]),
                xytext=(k_sil + 1.0, sil_axis[i_sil] - 0.03), fontsize=9, color=C1,
                arrowprops=dict(arrowstyle="->", color=C1, lw=1.3))
    ax.annotate(f"ARI 在此最大\nK={k_ari}（真实类别数）", xy=(k_ari, ari_axis[i_ari]),
                xytext=(k_ari + 1.0, ari_axis[i_ari] - 0.10), fontsize=9, color=C2,
                arrowprops=dict(arrowstyle="->", color=C2, lw=1.3))
    ax.axvline(3, color="#55A868", ls="--", lw=1.5, label="真实 K=3")
    ax.set_xlabel("K")
    ax.set_ylabel("指标值")
    ax.set_title("Iris：轮廓系数（内部指标）推荐 K=2，ARI 推荐 K=3", pad=12)
    ax.set_ylim(0.40, 0.95)
    ax.legend(fontsize=9, loc="upper right")
    fig.savefig(FIG / "fig09_internal_vs_external.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig09_internal_vs_external.png")
    print(f"  轮廓系数最大处 K={k_sil} (值 {sil_axis[i_sil]:.4f})，ARI 最大处 K={k_ari} (值 {ari_axis[i_ari]:.4f})")
    print("  真实类别数是 3：K=2 的 ARI=%.4f（把两个品种合并），K=3 的 ARI=%.4f" % (ari_axis[0], ari_axis[1]))


if __name__ == "__main__":
    fig_elbow()
    fig_internal_vs_external()
    print("\n讲义 03 的插图已生成到 learn/figures/")
