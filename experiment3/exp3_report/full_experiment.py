#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验三：k-means 聚类与混合高斯模型（EM 算法）——完整实验脚本

本脚本一次性产出实验报告需要的全部结果与图片：
    图 fig1  合成数据的真实分布（三个不同均值和方差的高斯）
    图 fig2  k-means 聚类结果与目标函数下降曲线
    图 fig3  k-means 不同初始化的 inertia 对比（局部最优问题）
    图 fig4  GMM 的对数似然曲线与参数恢复对照
    图 fig5  GMM 软归属与协方差椭圆
    图 fig6  EM 的下界与对数似然（单调性验证）
    图 fig7  Iris 上的 K 扫描（惯性、轮廓系数、ARI）
    图 fig8  Iris 聚类结果对比（k-means vs GMM）
    图 fig9  pendigits 上的结果
    图 fig10 pendigits 的 K 扫描与 BIC

    结果 run_output.txt   全部终端输出
    结果 results.json     结构化结果，供报告引用

运行方式：
    cd exp3_report
    python full_experiment.py

环境：Python 3.13 + NumPy 2.5 + Matplotlib 3.11（不使用 scipy/sklearn）
"""

from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 路径与全局设置
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ / "code"))

import data_gen as dg          # noqa: E402  数据生成与加载
from kmeans import kmeans      # noqa: E402  自己实现的 k-means
from metrics import adjusted_rand_index, best_match_accuracy, confusion_matrix_from_match  # noqa: E402


def kmeans_restarts(X, k, n_init=10, seed=0, **kwargs):
    """多次随机重启，取 inertia 最小的结果。

    这里没有直接调用 code/kmeans.py 里的 kmeans_restarts，因为它内部的
    循环变量与关键字参数 seed 冲突（调用 kmeans(X, k, seed=seed, **kwargs)
    会在显式传 seed 时抛 TypeError）。本函数用等价的逻辑重新实现一遍，
    效果与讲义/练习中的用法一致。

    返回 (best_dict, all_inertias)
    """
    best, all_J = None, []
    for s in range(seed, seed + n_init):
        centroids, labels, inertia, history = kmeans(X, k, seed=s, **kwargs)
        all_J.append(float(inertia))
        if best is None or inertia < best["inertia"]:
            best = {"centroids": centroids, "labels": labels,
                    "inertia": float(inertia), "history": history, "seed": s}
    return best, all_J

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
PALETTE = COLORS

RESULTS = {}

# ---------------------------------------------------------------------------
# 合成数据：三个不同均值和方差的高斯（实验要求"参数自己设定"）
# ---------------------------------------------------------------------------
TRUE_MEANS = np.array([[-4.0, -3.0], [0.0, 4.5], [5.0, -2.0]])
TRUE_COVS = np.array([
    [[1.0, 0.0], [0.0, 1.0]],      # 各向同性（圆）
    [[2.5, 0.0], [0.0, 0.25]],     # 横躺的细长椭圆
    [[1.2, 0.9], [0.9, 1.2]],      # 45° 倾斜椭圆
])
TRUE_WEIGHTS = np.array([0.4, 0.35, 0.25])
N_SAMPLES = 1500
SEED = 0


# ---------------------------------------------------------------------------
# GMM + EM 实现（参考实现，报告 6.2 节给出完整代码）
# ---------------------------------------------------------------------------
def logsumexp(a, axis=None, keepdims=False):
    """数值稳定的 log(sum(exp(a)))。先减去最大值防止溢出。"""
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def regularize_cov(cov, reg=1e-6):
    """给协方差加一个与尺度相关的小量，保证正定（防止奇异性）。"""
    cov = np.asarray(cov, dtype=float)
    d = cov.shape[-1]
    scale = reg * np.trace(cov, axis1=-2, axis2=-1) / d
    eye = np.eye(d)
    return cov + (scale[..., None, None] * eye if cov.ndim > 2 else scale * eye)


def log_gaussian(X, mean, cov, reg=1e-6):
    """多元高斯对数密度，用 Cholesky 分解（不做显式求逆）。"""
    X = np.atleast_2d(X)
    D = X.shape[1]
    L = np.linalg.cholesky(regularize_cov(cov, reg))
    diff = X - np.asarray(mean)
    sol = np.linalg.solve(L, diff.T)
    maha = np.sum(sol ** 2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (D * np.log(2 * np.pi) + logdet + maha)


def estimate_log_prob(X, weights, means, covs):
    """返回 (N, K) 的 log(pi_k * N(x_n | mu_k, Sigma_k))。"""
    return np.column_stack([
        np.log(weights[k]) + log_gaussian(X, means[k], covs[k])
        for k in range(len(weights))
    ])


def e_step(X, weights, means, covs):
    """E 步：返回对数责任度 (N,K) 与总对数似然。"""
    weighted = estimate_log_prob(X, weights, means, covs)
    lse = logsumexp(weighted, axis=1)
    return weighted - lse[:, None], float(lse.sum())


def m_step(X, log_resp):
    """M 步：由对数责任度更新参数，返回 weights, means, covs, Nk。"""
    resp = np.exp(log_resp)
    Nk = resp.sum(axis=0)
    Nk_safe = np.maximum(Nk, 1e-10)
    N, D = X.shape
    weights = Nk / N
    means = (resp.T @ X) / Nk_safe[:, None]
    covs = np.empty((len(Nk), D, D))
    for k in range(len(Nk)):
        d = X - means[k]
        covs[k] = (resp[:, k, None] * d).T @ d / Nk_safe[k]
    return weights, means, covs, Nk


def elbo(X, log_resp, weights, means, covs):
    """下界 L(q,theta) = sum_n sum_k q_nk [log pi_k + log N_k - log q_nk]。"""
    weighted = estimate_log_prob(X, weights, means, covs)
    resp = np.exp(log_resp)
    ent = np.where(resp > 0, resp * np.log(np.where(resp > 0, resp, 1.0)), 0.0)
    return float((resp * weighted).sum() - ent.sum())


def gmm_em(X, k, max_iter=200, tol=1e-6, reg=1e-6, init="kmeans", seed=0, track_q=False):
    """EM 主循环。

    init : "kmeans" 用 k-means 中心作初始均值；"random" 随机挑 k 个点
    track_q : 为 True 时额外记录固定 q 下的下界（用于绘制下界曲线）
    返回 params(dict), resp(N,K), log_lik(list), history(list)
    """
    N, D = X.shape
    if init == "kmeans":
        best, _ = kmeans_restarts(X, k, n_init=10, seed=seed)
        means = best["centroids"].copy()
        # 协方差统一取全体数据的协方差（讲义 03 第 6.1 节推荐做法）
        covs = np.array([regularize_cov(np.cov(X.T, bias=True), reg)] * k)
        weights = np.full(k, 1.0 / k)
    else:
        rng = np.random.default_rng(seed)
        means = X[rng.choice(N, k, replace=False)].copy()
        covs = np.array([regularize_cov(np.cov(X.T, bias=True), reg)] * k)
        weights = np.full(k, 1.0 / k)

    log_lik, history = [], []
    log_resp = None
    for it in range(1, max_iter + 1):
        log_resp, ll = e_step(X, weights, means, covs)
        delta = np.inf if not log_lik else abs(ll - log_lik[-1]) / max(abs(log_lik[-1]), 1e-300)
        log_lik.append(ll)
        history.append({"iter": it, "log_lik": ll, "delta": float(delta)})
        if it > 1 and delta < tol:
            break
        weights, means, covs, Nk = m_step(X, log_resp)
        covs = np.array([regularize_cov(c, reg) for c in covs])
    log_resp, ll = e_step(X, weights, means, covs)
    return ({"weights": weights, "means": means, "covs": covs},
            np.exp(log_resp), log_lik, history)


def compute_log_likelihood(X, weights, means, covs):
    """独立重算对数似然（用朴素定义 + log-sum-exp，用于交叉校验）。"""
    return float(logsumexp(estimate_log_prob(X, weights, means, covs), axis=1).sum())


def bic(log_lik, n_samples, k, d):
    """BIC = p*ln(N) - 2*ln(L)，p 为满协方差 GMM 的自由参数个数。"""
    p = k * (d + d * (d + 1) // 2 + 1) - 1
    return p * np.log(n_samples) - 2.0 * log_lik


# ---------------------------------------------------------------------------
# 绘图工具
# ---------------------------------------------------------------------------
def ellipse_points(mean, cov, n_std=1.0, n=200):
    """马氏距离为 n_std 的椭圆上的点。"""
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    vals = np.clip(vals, 1e-12, None)
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    return (circle * (n_std * np.sqrt(vals))[None, :]) @ vecs.T + np.asarray(mean)[None, :]


def draw_ellipses(ax, means, covs, n_std=(1.0, 2.0), colors=None, prefix=""):
    colors = colors or PALETTE
    means = np.atleast_2d(means)
    for j, (mu, cov) in enumerate(zip(means, covs)):
        c = colors[j % len(colors)]
        for i, s in enumerate(n_std):
            pts = ellipse_points(mu, cov, s)
            ax.plot(pts[:, 0], pts[:, 1], color=c, lw=1.7 if i == 0 else 1.2,
                    ls="-" if i == 0 else "--", alpha=0.9)
    return ax


def save(fig, name):
    path = FIG / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path.relative_to(HERE)}")
    return path


def cov_summary(cov):
    """用特征值比、主轴角度概括协方差（写进结果表用）。"""
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    vals = np.clip(vals, 1e-300, None)
    major = vecs[:, -1]
    if major[0] < 0:
        major = -major
    return {
        "eigvals": [float(v) for v in vals],
        "ratio": float(vals[-1] / vals[0]),
        "angle_deg": float(np.degrees(np.arctan2(major[1], major[0]))),
    }


def align_components(est_means, true_means):
    """找最优排列，使 est_means[perm] 与 true_means 对齐。"""
    k = len(true_means)
    best_perm, best_err = list(range(k)), np.inf
    for perm in permutations(range(k)):
        err = np.abs(np.asarray(est_means)[list(perm)] - np.asarray(true_means)).max()
        if err < best_err:
            best_err, best_perm = err, list(perm)
    return best_perm, float(best_err)


# ===========================================================================
# 实验一：合成数据的生成与 k-means
# ===========================================================================
def experiment_kmeans():
    print("=" * 74)
    print("实验一：用高斯分布生成 k 个簇，用 k-means 聚类")
    print("=" * 74)
    X, y = dg.make_synthetic_data(TRUE_MEANS, TRUE_COVS, TRUE_WEIGHTS,
                                  n_samples=N_SAMPLES, seed=SEED)
    print(f"数据: {X.shape[0]} 个样本, {X.shape[1]} 维, 真实簇数 K=3")
    print(f"真实混合系数: {TRUE_WEIGHTS.tolist()}")
    print(f"经验混合系数: {(np.bincount(y) / len(y)).round(4).tolist()}")
    for j in range(3):
        emp_mu = X[y == j].mean(axis=0)
        emp_cov = np.cov(X[y == j].T, bias=True)
        s = cov_summary(TRUE_COVS[j])
        es = cov_summary(emp_cov)
        print(f"  簇{j}: 经验均值 {emp_mu.round(3).tolist()} (设定 {TRUE_MEANS[j].tolist()})"
              f"  长短轴比 {es['ratio']:.2f} (设定 {s['ratio']:.2f})"
              f"  主轴 {es['angle_deg']:.1f}° (设定 {s['angle_deg']:.1f}°)")
    print(f"  均值最大偏差 |经验-设定| = {np.abs(np.array([X[y==j].mean(0) for j in range(3)]) - TRUE_MEANS).max():.4f}")
    print(f"  协方差最大偏差 = {np.abs(np.array([np.cov(X[y==j].T, bias=True) for j in range(3)]) - TRUE_COVS).max():.4f}")

    RESULTS["synthetic"] = {
        "shape": list(X.shape),
        "true_means": TRUE_MEANS.tolist(),
        "true_covs": TRUE_COVS.tolist(),
        "true_weights": TRUE_WEIGHTS.tolist(),
        "empirical_weights": (np.bincount(y) / len(y)).tolist(),
    }

    # ---- 图 1：真实分布 ----
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for j in range(3):
        m = y == j
        ax.scatter(X[m, 0], X[m, 1], s=15, alpha=0.55, color=PALETTE[j],
                   edgecolor="none", label=f"真实簇 {j}")
    draw_ellipses(ax, TRUE_MEANS, TRUE_COVS)
    ax.scatter(TRUE_MEANS[:, 0], TRUE_MEANS[:, 1], marker="X", s=200, color="black",
               zorder=6, label="真实均值")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("合成数据：三个不同均值与协方差的高斯（1σ/2σ 椭圆）")
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, "fig1_synthetic_data.png")

    # ---- k-means：单次与多次重启 ----
    C, labels, inertia, history = kmeans(X, k=3, seed=0)
    acc, mapping = best_match_accuracy(y, labels)
    ari = adjusted_rand_index(y, labels)
    Js = [float(h["inertia"]) for h in history]
    print(f"\nk-means (seed=0): 迭代 {len(history)} 轮, 最终 J = {inertia:.2f}")
    print(f"  inertia 序列: {[round(v, 1) for v in Js]}")
    print(f"  单调不增: {bool(np.all(np.diff(Js) <= 1e-9))}")
    print(f"  最优匹配准确率 = {acc:.4f}, ARI = {ari:.4f}")
    print(f"  估计中心:\n{np.round(C, 3)}")
    print(f"  真实均值:\n{TRUE_MEANS}")

    best, all_J = kmeans_restarts(X, k=3, n_init=10)
    print(f"\nk-means 10 次重启: {[round(float(v), 1) for v in all_J]}")
    print(f"  最好 J = {min(all_J):.1f} (seed={best['seed']}), 最差 J = {max(all_J):.1f},"
          f" 差距 {max(all_J)/min(all_J):.2f} 倍")

    RESULTS["kmeans_synthetic"] = {
        "n_iter": len(history), "inertia": float(inertia),
        "inertia_curve": [float(v) for v in Js],
        "monotone": bool(np.all(np.diff(Js) <= 1e-9)),
        "accuracy": float(acc), "ari": float(ari),
        "centroids": C.tolist(),
        "restart_inertias": [float(v) for v in all_J],
        "restart_best_seed": int(best["seed"]),
        "confusion": confusion_matrix_from_match(y, labels).tolist(),
    }
    print(f"  混淆矩阵:\n{confusion_matrix_from_match(y, labels)}")

    # ---- 图 2：k-means 结果 + 下降曲线 ----
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    ax = axes[0]
    for j in range(3):
        m = labels == j
        ax.scatter(X[m, 0], X[m, 1], s=15, alpha=0.55, color=PALETTE[j],
                   edgecolor="none", label=f"k-means 簇 {j}")
    ax.scatter(C[:, 0], C[:, 1], marker="X", s=200, color="black", zorder=6, label="估计中心")
    ax.set_title(f"k-means 聚类结果（ARI={ari:.4f}，准确率={acc:.4f}）")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(fontsize=8.5, loc="upper left")

    ax = axes[1]
    ax.plot(range(1, len(Js) + 1), Js, "o-", color=PALETTE[0], lw=2, ms=6)
    for i, v in enumerate(Js):
        ax.annotate(f"{v:.0f}", (i + 1, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5)
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("簇内平方和 J")
    ax.set_title("k-means 目标函数下降曲线（单调不增）")
    save(fig, "fig2_kmeans_result.png")

    # ---- 图 3：不同初始化的对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    ax = axes[0]
    ax.bar(range(10), [float(v) for v in all_J], color=PALETTE[1], width=0.62)
    ax.axhline(min(all_J), color=PALETTE[2], ls="--", lw=1.6,
               label=f"全局最优 J={min(all_J):.0f}")
    bad = int(np.argmax(all_J))
    ax.annotate(f"seed={bad}\n落入局部最优\nJ={max(all_J):.0f}",
                xy=(bad, max(all_J)), xytext=(bad + 0.4, max(all_J) * 0.82),
                fontsize=9, color="#C44E52",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.3))
    ax.set_xlabel("随机种子")
    ax.set_ylabel("最终 J")
    ax.set_title("10 次不同初始化的最终 J")
    ax.legend(fontsize=9)

    # 好坏两条下降曲线
    ax = axes[1]
    _, _, _, h_good = kmeans(X, k=3, seed=best["seed"])
    _, _, _, h_bad = kmeans(X, k=3, seed=bad)
    ax.plot(range(1, len(h_good) + 1), [h["inertia"] for h in h_good], "o-",
            color=PALETTE[2], lw=2, ms=5, label=f"好初始化 (seed={best['seed']})")
    ax.plot(range(1, len(h_bad) + 1), [h["inertia"] for h in h_bad], "s-",
            color="#C44E52", lw=2, ms=5, label=f"坏初始化 (seed={bad})")
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("簇内平方和 J")
    ax.set_title("两条下降曲线都正常收敛，但落点相差 5 倍")
    ax.legend(fontsize=9)
    save(fig, "fig3_kmeans_init.png")

    return X, y, best


# ===========================================================================
# 实验二：GMM + EM
# ===========================================================================
def experiment_gmm(X, y):
    print()
    print("=" * 74)
    print("实验二：混合高斯模型与 EM 算法估计参数")
    print("=" * 74)

    params, resp, log_lik, history = gmm_em(X, k=3, init="kmeans", seed=SEED)
    L = np.array([float(v) for v in log_lik])
    print(f"EM 迭代 {len(L)} 轮, 对数似然: 首轮 {L[0]:.2f} -> 末轮 {L[-1]:.2f}")
    print(f"  每轮对数似然: {[round(v, 2) for v in L]}")
    print(f"  每轮增量: {[round(v, 6) for v in np.diff(L)]}")
    d = np.diff(L)
    print(f"  单调不降(绝对判据 d>=-1e-9): {bool(np.all(d >= -1e-9))}")
    print(f"  单调不降(相对判据 |d|/|L|): {bool(np.all(d >= -np.abs(L[:-1]) * 1e-12))}")

    ll_check = compute_log_likelihood(X, params["weights"], params["means"], params["covs"])
    print(f"\n独立校验: 主循环末轮 {L[-1]:.6f} vs 朴素重算 {ll_check:.6f}"
          f" (差 {abs(L[-1] - ll_check):.2e})")

    # 参数恢复对照（按经验统计量对齐）
    perm, _ = align_components(params["means"], TRUE_MEANS)
    est_mu = params["means"][perm]
    est_cov = params["covs"][perm]
    est_w = params["weights"][perm]
    emp_mu = np.array([X[y == j].mean(axis=0) for j in range(3)])
    emp_cov = np.array([np.cov(X[y == j].T, bias=True) for j in range(3)])
    emp_w = np.bincount(y) / len(y)

    print(f"\n参数恢复（排列 {perm}）:")
    print(f"  {'':6}{'估计均值':<22}{'经验均值':<22}{'设定均值'}")
    for i in range(3):
        print(f"  分量{i}  {str(est_mu[i].round(3)):<22}{str(emp_mu[i].round(3)):<22}{TRUE_MEANS[i]}")
    print(f"  混合系数 估计 {est_w.round(4).tolist()}  经验 {emp_w.round(4).tolist()}"
          f"  设定 {TRUE_WEIGHTS.tolist()}")
    print(f"  与经验统计量最大偏差: 均值 {np.abs(est_mu-emp_mu).max():.2e},"
          f" 协方差 {np.abs(est_cov-emp_cov).max():.2e}, 权重 {np.abs(est_w-emp_w).max():.2e}")
    print(f"  与设定值最大偏差:     均值 {np.abs(est_mu-TRUE_MEANS).max():.4f},"
          f" 权重 {np.abs(est_w-TRUE_WEIGHTS).max():.4f}")

    # 第二层校验：设定值代入似然应当更低
    ll_true = compute_log_likelihood(X, TRUE_WEIGHTS, TRUE_MEANS, TRUE_COVS)
    print(f"\n对数似然对照: 估计参数 {L[-1]:.4f}  vs  设定参数 {ll_true:.4f}"
          f"  (估计更优: {L[-1] >= ll_true})")

    print(f"\n协方差形态恢复（长度比与主轴角度）:")
    for i in range(3):
        e, t = cov_summary(est_cov[i]), cov_summary(TRUE_COVS[i])
        print(f"  分量{i}: 长短轴比 {e['ratio']:.2f} (设定 {t['ratio']:.2f}),"
              f" 主轴 {e['angle_deg']:.1f}° (设定 {t['angle_deg']:.1f}°)")

    labels_hat = resp.argmax(axis=1)
    ari_gmm = adjusted_rand_index(y, labels_hat)
    acc_gmm, _ = best_match_accuracy(y, labels_hat)
    print(f"\nGMM 聚类: ARI = {ari_gmm:.4f}, 准确率 = {acc_gmm:.4f}")

    RESULTS["gmm_synthetic"] = {
        "n_iter": len(L),
        "log_lik_curve": [float(v) for v in L],
        "log_lik_delta": [float(v) for v in d],
        "monotone_absolute": bool(np.all(d >= -1e-9)),
        "monotone_relative": bool(np.all(d >= -np.abs(L[:-1]) * 1e-12)),
        "independent_check": float(ll_check),
        "log_lik_true_params": float(ll_true),
        "estimated_is_better": bool(L[-1] >= ll_true),
        "permutation": [int(p) for p in perm],
        "est_means": est_mu.tolist(),
        "emp_means": emp_mu.tolist(),
        "est_weights": est_w.tolist(),
        "emp_weights": emp_w.tolist(),
        "est_cov_summary": [cov_summary(c) for c in est_cov],
        "true_cov_summary": [cov_summary(c) for c in TRUE_COVS],
        "max_err_vs_empirical": {
            "means": float(np.abs(est_mu - emp_mu).max()),
            "covs": float(np.abs(est_cov - emp_cov).max()),
            "weights": float(np.abs(est_w - emp_w).max()),
        },
        "max_err_vs_true": {
            "means": float(np.abs(est_mu - TRUE_MEANS).max()),
            "weights": float(np.abs(est_w - TRUE_WEIGHTS).max()),
        },
        "ari": float(ari_gmm), "accuracy": float(acc_gmm),
    }

    # ---- 图 4：对数似然曲线 + 参数恢复 ----
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    ax = axes[0]
    ax.plot(range(1, len(L) + 1), L, "o-", color=PALETTE[0], lw=2, ms=6)
    for i, v in enumerate(L):
        ax.annotate(f"{v:.0f}", (i + 1, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5)
    ax.set_xlabel("EM 迭代轮数")
    ax.set_ylabel("对数似然 $\\ln p(X|\\theta)$")
    ax.set_title(f"EM 每轮对数似然（单调不降，共 {len(L)} 轮）")

    ax = axes[1]
    for j in range(3):
        m = y == j
        ax.scatter(X[m, 0], X[m, 1], s=11, alpha=0.35, color=PALETTE[j], edgecolor="none")
    draw_ellipses(ax, TRUE_MEANS, TRUE_COVS, colors=["#999999"] * 3)
    draw_ellipses(ax, est_mu, est_cov)
    ax.scatter(TRUE_MEANS[:, 0], TRUE_MEANS[:, 1], marker="P", s=150,
               facecolor="none", edgecolor="#555555", linewidth=1.8, label="设定均值")
    ax.scatter(est_mu[:, 0], est_mu[:, 1], marker="X", s=170, color="black",
               zorder=6, label="估计均值")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("参数恢复：灰色虚线=设定，彩色实线=估计")
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, "fig4_gmm_likelihood.png")

    # ---- 图 5：软归属 ----
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    hard = resp.argmax(axis=1)
    conf = resp.max(axis=1)
    ax.scatter(X[:, 0], X[:, 1], s=15, c=[PALETTE[j] for j in hard],
               alpha=0.15 + 0.75 * conf, edgecolor="none")
    draw_ellipses(ax, est_mu, est_cov)
    ax.scatter(est_mu[:, 0], est_mu[:, 1], marker="X", s=180, color="black", zorder=6)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("GMM 软归属：点的透明度 = 最大责任度（越透明越不确定）")
    save(fig, "fig5_gmm_soft.png")

    # ---- 图 6：下界与对数似然（单调性验证）----
    # 做法：重新跑一遍 EM，逐轮记下参数；然后用【收敛后的责任度】作为固定的 q，
    # 去评估每一轮的参数，得到一条全程有效的下界 L(q, theta)。
    # 注意 e_step 返回 (对数责任度, 对数似然)，两者都是对数域的量。
    best_km, _ = kmeans_restarts(X, 3, n_init=10, seed=SEED)
    m_now = best_km["centroids"].copy()
    c_now = np.array([regularize_cov(np.cov(X.T, bias=True))] * 3)
    w_now = np.full(3, 1 / 3)
    means_hist, covs_hist, w_hist = [], [], []
    for _ in range(len(L)):
        means_hist.append(m_now.copy())
        covs_hist.append(c_now.copy())
        w_hist.append(w_now.copy())
        lr_now, _ = e_step(X, w_now, m_now, c_now)   # 首个返回值才是对数责任度
        w_now, m_now, c_now, _ = m_step(X, lr_now)
        c_now = np.array([regularize_cov(c) for c in c_now])

    # 收敛后的责任度作为固定 q
    log_resp_final, _ = e_step(X, params["weights"], params["means"], params["covs"])
    bound = np.array([elbo(X, log_resp_final, w_hist[i], means_hist[i], covs_hist[i])
                      for i in range(len(L))])
    print(f"\n下界（用收敛后的 q 固定算）: 首轮 {bound[0]:.2f} -> 末轮 {bound[-1]:.2f}")
    print(f"  下界单调不降: {bool(np.all(np.diff(bound) >= -1e-9))}")
    print(f"  末轮下界与对数似然之差: {abs(bound[-1] - L[-1]):.2e}（应接近 0）")

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    it_axis = np.arange(1, len(L) + 1)
    ax.plot(it_axis, L, "o-", color=PALETTE[0], lw=2, ms=5,
            label="对数似然 $\\ln p(X|\\theta)$（EM 每轮达到）")
    ax.plot(it_axis, bound, "s--", color="#C44E52", lw=2, ms=5,
            label="下界 $\\mathcal{L}(q,\\theta)$（固定收敛后的 q）")
    ax.fill_between(it_axis, bound, L, color=PALETTE[2], alpha=0.15, label="差距 = KL 散度")
    ax.set_xlabel("EM 迭代轮数")
    ax.set_ylabel("对数似然")
    ax.set_title("EM 单调性：下界单调上升，最终追上对数似然")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "fig6_em_bound.png")

    RESULTS["gmm_synthetic"]["bound_curve"] = [float(v) for v in bound]
    RESULTS["gmm_synthetic"]["bound_monotone"] = bool(np.all(np.diff(bound) >= -1e-9))

    # ---- K = 2/3/4 的对比 ----
    print("\n不同 K 的对数似然与 BIC（考察'似然最大'是否等于'模型正确'）:")
    print(f"  {'K':<4}{'对数似然':<14}{'BIC':<14}{'ARI':<10}{'参数个数'}")
    k_rows = []
    for K in (2, 3, 4):
        p2, r2, l2, _ = gmm_em(X, k=K, init="kmeans", seed=SEED, max_iter=300)
        a2 = adjusted_rand_index(y, r2.argmax(axis=1))
        b2 = bic(l2[-1], len(X), K, 2)
        nparam = K * (2 + 3 + 1) - 1
        k_rows.append({"k": K, "log_lik": float(l2[-1]), "bic": float(b2),
                       "ari": float(a2), "n_param": int(nparam)})
        print(f"  {K:<4}{l2[-1]:<14.2f}{b2:<14.2f}{a2:<10.4f}{nparam}")
    RESULTS["gmm_synthetic"]["k_sweep"] = k_rows

    return params, resp, L


# ===========================================================================
# 实验三：UCI 数据集
# ===========================================================================
def sweep_k(X, y, kmax=8, n_init=10, seed=0):
    """对 K=2..kmax 跑 k-means 与 GMM，返回指标表。"""
    rows = []
    for K in range(2, kmax + 1):
        best, allJ = kmeans_restarts(X, K, n_init=n_init, seed=seed)
        ari_km = adjusted_rand_index(y, best["labels"])
        acc_km, _ = best_match_accuracy(y, best["labels"])
        p, r, ll, h = gmm_em(X, k=K, init="kmeans", seed=seed, max_iter=300)
        ari_gmm = adjusted_rand_index(y, r.argmax(axis=1))
        acc_gmm, _ = best_match_accuracy(y, r.argmax(axis=1))
        rows.append({
            "k": K,
            "km_inertia": float(best["inertia"]),
            "km_ari": float(ari_km), "km_acc": float(acc_km),
            "km_iter": len(best["history"]),
            "gmm_loglik": float(ll[-1]), "gmm_iter": len(ll),
            "gmm_ari": float(ari_gmm), "gmm_acc": float(acc_gmm),
            "bic": float(bic(ll[-1], len(X), K, X.shape[1])),
        })
    return rows


def report_uci(name, X, y, k_true, kmax=8):
    print(f"\n{'-' * 74}")
    print(f"数据集 {name}: {X.shape[0]} 样本, {X.shape[1]} 维, 真实类别数 {k_true}")
    print(f"  类别分布: {np.bincount(y).tolist()}")
    rows = []
    header = (f"  {'K':<4}{'km J':<12}{'km ARI':<10}{'km 轮':<8}"
              f"{'GMM 对数似然':<15}{'GMM 轮':<8}{'GMM ARI':<10}{'BIC'}")
    for K in range(2, kmax + 1):
        print(f"  ... 正在跑 K={K}", flush=True)
        best, allJ = kmeans_restarts(X, K, n_init=10, seed=0)
        ari_km = adjusted_rand_index(y, best["labels"])
        acc_km, _ = best_match_accuracy(y, best["labels"])
        p, r, ll, h = gmm_em(X, k=K, init="kmeans", seed=0, max_iter=200)
        ari_gmm = adjusted_rand_index(y, r.argmax(axis=1))
        acc_gmm, _ = best_match_accuracy(y, r.argmax(axis=1))
        rows.append({
            "k": K,
            "km_inertia": float(best["inertia"]),
            "km_ari": float(ari_km), "km_acc": float(acc_km),
            "km_iter": len(best["history"]),
            "gmm_loglik": float(ll[-1]), "gmm_iter": len(ll),
            "gmm_ari": float(ari_gmm), "gmm_acc": float(acc_gmm),
            "bic": float(bic(ll[-1], len(X), K, X.shape[1])),
        })
    print(f"  {'K':<4}{'km J':<12}{'km ARI':<10}{'km 轮':<8}"
          f"{'GMM 对数似然':<15}{'GMM 轮':<8}{'GMM ARI':<10}{'BIC'}")
    for r in rows:
        print(f"  {r['k']:<4}{r['km_inertia']:<12.1f}{r['km_ari']:<10.4f}{r['km_iter']:<8}"
              f"{r['gmm_loglik']:<15.1f}{r['gmm_iter']:<8}{r['gmm_ari']:<10.4f}{r['bic']:.1f}")
    best_km = max(rows, key=lambda r: r["km_ari"])
    best_gmm = max(rows, key=lambda r: r["gmm_ari"])
    k_bic = min(rows, key=lambda r: r["bic"])["k"]
    print(f"  最优 k-means ARI 在 K={best_km['k']} ({best_km['km_ari']:.4f});"
          f" 最优 GMM ARI 在 K={best_gmm['k']} ({best_gmm['gmm_ari']:.4f})")
    print(f"  BIC 最小在 K={k_bic}（真实 {k_true}）"
          f" → {'正确' if k_bic == k_true else '未选中真实 K'}")
    return rows, {"best_km": best_km, "best_gmm": best_gmm, "k_bic": k_bic}


def experiment_uci_iris():
    print()
    print("=" * 74)
    print("实验三：UCI Iris 数据集")
    print("=" * 74)
    X, y, names = dg.load_dataset("iris_4d")
    rows, summary = report_uci("Iris (4 维)", X, y, 3, kmax=8)
    RESULTS["uci_iris"] = {"rows": rows, "summary": {
        k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()})
        for k, v in summary.items()}}

    # K=3 的详细结果与图
    best_km, _ = kmeans_restarts(X, 3, n_init=10, seed=0)
    params, resp, ll, hist = gmm_em(X, k=3, init="kmeans", seed=0)
    ari_km = adjusted_rand_index(y, best_km["labels"])
    ari_gmm = adjusted_rand_index(y, resp.argmax(axis=1))
    cm_km = confusion_matrix_from_match(y, best_km["labels"])
    cm_gmm = confusion_matrix_from_match(y, resp.argmax(axis=1))
    print(f"\nK=3 详细结果:")
    print(f"  k-means: ARI={ari_km:.4f}, 混淆矩阵\n{cm_km}")
    print(f"  GMM    : ARI={ari_gmm:.4f}, 对数似然 {ll[-1]:.2f}, 迭代 {len(ll)} 轮,"
          f" 混淆矩阵\n{cm_gmm}")
    print(f"  GMM 单调不降: {bool(np.all(np.diff(ll) >= -np.abs(ll[:-1]) * 1e-12))}")

    RESULTS["uci_iris_k3"] = {
        "km_ari": float(ari_km), "gmm_ari": float(ari_gmm),
        "km_confusion": cm_km.tolist(), "gmm_confusion": cm_gmm.tolist(),
        "gmm_loglik": float(ll[-1]), "gmm_iter": len(ll),
        "gmm_loglik_curve": [float(v) for v in ll],
    }

    # 图 7：K 扫描
    ks = [r["k"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    ax = axes[0]
    ax.plot(ks, [r["km_inertia"] for r in rows], "o-", color=PALETTE[0], lw=2)
    ax.axvline(3, color=PALETTE[2], ls="--", lw=1.6, label="真实 K=3")
    ax.set_xlabel("K")
    ax.set_ylabel("k-means 的 J")
    ax.set_title("惯性随 K 下降（肘部法）")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(ks, [r["km_ari"] for r in rows], "o-", color=PALETTE[0], lw=2, label="k-means")
    ax.plot(ks, [r["gmm_ari"] for r in rows], "s-", color=PALETTE[1], lw=2, label="GMM")
    ax.axvline(3, color=PALETTE[2], ls="--", lw=1.6)
    ax.set_xlabel("K")
    ax.set_ylabel("ARI")
    ax.set_title("两种方法的 ARI 对比")
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(ks, [r["bic"] for r in rows], "^-", color=PALETTE[3], lw=2)
    ax.axvline(3, color=PALETTE[2], ls="--", lw=1.6, label="真实 K=3")
    ax.set_xlabel("K")
    ax.set_ylabel("BIC")
    ax.set_title("BIC 模型选择")
    ax.legend(fontsize=9)
    save(fig, "fig7_iris_k_sweep.png")

    # 图 8：K=3 的两种方法对比（用花瓣两维画图）
    X2 = X[:, 2:4]
    best_km2, _ = kmeans_restarts(X2, 3, n_init=10, seed=0)
    p2, r2, _, _ = gmm_em(X2, k=3, init="kmeans", seed=0)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    ax = axes[0]
    for j in range(3):
        m = best_km2["labels"] == j
        ax.scatter(X2[m, 0], X2[m, 1], s=34, alpha=0.75, color=PALETTE[j],
                   edgecolor="white", linewidth=0.5)
    ax.scatter(best_km2["centroids"][:, 0], best_km2["centroids"][:, 1],
               marker="X", s=180, color="black", zorder=6)
    ax.set_title(f"k-means (K=3)  ARI={adjusted_rand_index(y, best_km2['labels']):.4f}")
    ax.set_xlabel("花瓣长度 (cm)")
    ax.set_ylabel("花瓣宽度 (cm)")

    ax = axes[1]
    hard = r2.argmax(axis=1)
    ax.scatter(X2[:, 0], X2[:, 1], s=34, c=[PALETTE[j] for j in hard],
               alpha=0.25 + 0.7 * r2.max(axis=1), edgecolor="none")
    draw_ellipses(ax, p2["means"], p2["covs"])
    ax.scatter(p2["means"][:, 0], p2["means"][:, 1], marker="X", s=180,
               color="black", zorder=6)
    ax.set_title(f"GMM (K=3)  ARI={adjusted_rand_index(y, r2.argmax(1)):.4f}")
    ax.set_xlabel("花瓣长度 (cm)")
    ax.set_ylabel("花瓣宽度 (cm)")
    save(fig, "fig8_iris_compare.png")


def experiment_uci_pendigits():
    print()
    print("=" * 74)
    print("实验四：UCI pendigits 数据集（样本量更大的数据，用于验证规模效应）")
    print("=" * 74)
    X, y, names = dg.load_dataset("pendigits")
    rows, summary = report_uci("pendigits (16 维)", X, y, 10, kmax=10)
    RESULTS["uci_pendigits"] = {"rows": rows, "k_bic": summary["k_bic"],
                                "best_km": summary["best_km"], "best_gmm": summary["best_gmm"]}

    # K=10 的详细结果
    best_km, allJ = kmeans_restarts(X, 10, n_init=10, seed=0)
    params, resp, ll, hist = gmm_em(X, k=10, init="kmeans", seed=0, max_iter=300)
    ari_km = adjusted_rand_index(y, best_km["labels"])
    ari_gmm = adjusted_rand_index(y, resp.argmax(axis=1))
    acc_km, _ = best_match_accuracy(y, best_km["labels"])
    acc_gmm, _ = best_match_accuracy(y, resp.argmax(axis=1))
    print(f"\nK=10 详细结果:")
    print(f"  k-means: J={best_km['inertia']:.1f}, ARI={ari_km:.4f}, 准确率={acc_km:.4f},"
          f" 重启的 J 范围 [{min(allJ):.1f}, {max(allJ):.1f}]")
    print(f"  GMM    : 对数似然 {ll[-1]:.2f}, 迭代 {len(ll)} 轮, ARI={ari_gmm:.4f},"
          f" 准确率={acc_gmm:.4f}")
    print(f"  GMM 单调不降: {bool(np.all(np.diff(ll) >= -np.abs(ll[:-1]) * 1e-12))}")
    d = np.diff(ll)
    print(f"  对数似然首轮 {ll[0]:.2f} -> 末轮 {ll[-1]:.2f}, 提升 {ll[-1]-ll[0]:.1f}")
    print(f"  协方差最大条件数: {max(np.linalg.cond(c) for c in params['covs']):.2e}")

    RESULTS["uci_pendigits_k10"] = {
        "km_inertia": float(best_km["inertia"]), "km_ari": float(ari_km),
        "km_acc": float(acc_km), "restart_range": [float(min(allJ)), float(max(allJ))],
        "gmm_loglik": float(ll[-1]), "gmm_iter": len(ll), "gmm_ari": float(ari_gmm),
        "gmm_acc": float(acc_gmm),
        "gmm_monotone": bool(np.all(d >= -np.abs(np.array(ll[:-1])) * 1e-12)),
        "loglik_curve": [float(v) for v in ll],
        "max_cond": float(max(np.linalg.cond(c) for c in params["covs"])),
    }

    # 图 9：pendigits 结果
    ks = [r["k"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    ax = axes[0]
    ax.plot(ks, [r["km_ari"] for r in rows], "o-", color=PALETTE[0], lw=2, label="k-means")
    ax.plot(ks, [r["gmm_ari"] for r in rows], "s-", color=PALETTE[1], lw=2, label="GMM")
    ax.axvline(10, color=PALETTE[2], ls="--", lw=1.6, label="真实 K=10")
    ax.set_xlabel("K")
    ax.set_ylabel("ARI")
    ax.set_title("pendigits：ARI 随 K 变化")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(range(1, len(ll) + 1), ll, "-", color=PALETTE[0], lw=2)
    ax.set_xlabel("EM 迭代轮数")
    ax.set_ylabel("对数似然")
    ax.set_title(f"pendigits 上 EM 的收敛（{len(ll)} 轮）")

    ax = axes[2]
    ax.plot(ks, [r["bic"] for r in rows], "^-", color=PALETTE[3], lw=2)
    ax.axvline(10, color=PALETTE[2], ls="--", lw=1.6, label="真实 K=10")
    ax.set_xlabel("K")
    ax.set_ylabel("BIC")
    ax.set_title("BIC 模型选择")
    ax.legend(fontsize=9)
    save(fig, "fig9_pendigits.png")

    # 混淆矩阵（K=10 时取前几类看）
    cm = confusion_matrix_from_match(y, resp.argmax(axis=1))
    print(f"  GMM 混淆矩阵（10x10）:\n{cm}")
    RESULTS["uci_pendigits_k10"]["confusion"] = cm.tolist()


# ===========================================================================
# 主流程
# ===========================================================================
class Tee:
    """把输出同时写到终端和文件。"""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8", buffering=1)  # 行缓冲
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()          # 每行都落盘，进程被中断也能保留进度

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def main():
    tee = Tee(HERE / "run_output.txt")
    sys.stdout = tee
    try:
        X, y, best = experiment_kmeans()
        experiment_gmm(X, y)
        experiment_uci_iris()
        experiment_uci_pendigits()

        with open(HERE / "results.json", "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2,
                      default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
        print("\n" + "=" * 74)
        print("全部实验完成")
        print(f"  结果: {HERE / 'results.json'}")
        print(f"  输出: {HERE / 'run_output.txt'}")
        print(f"  图片: {FIG}")
        print("=" * 74)
    finally:
        sys.stdout = tee.stdout
        tee.file.close()


if __name__ == "__main__":
    main()
