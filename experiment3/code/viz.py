"""绘图工具。

约定：
- 所有函数接收 `ax`（Axes），把图元画在给定的坐标轴上并返回 ax，方便
  组合子图；传 None 时自动新建。
- `save_fig` 统一负责落盘到 `figures/`，dpi=150，`bbox_inches="tight"`。
- 中文字体在导入时统一设置，避免图上出现方框。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境下的后端，必须在 pyplot 之前设置

import matplotlib.pyplot as plt
import numpy as np

# 中文字体：本机有 Noto Sans CJK SC 与文泉驿/AR PL 系列，按可用性依次回退。
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "AR PL UMing CN",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False  # 负号显示为正常减号
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

#: 类别配色，最多支持 8 个簇
PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
]


def save_fig(fig, name, subdir=None):
    """保存图片到 figures/ 并返回路径。name 不带扩展名。"""
    out_dir = FIG_DIR if subdir is None else FIG_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")
    return path


def _get_ax(ax, figsize=(6.0, 5.0)):
    if ax is not None:
        return ax, ax.figure
    fig, ax = plt.subplots(figsize=figsize)
    return ax, fig


def ellipse_points(mean, cov, n_std=1.0, n_points=200):
    """返回马氏距离为 n_std 的椭圆上的点，形状 (n_points, 2)。

    对协方差做特征分解 Sigma = U diag(lambda) U^T。特征值开方给出半轴
    长度，特征向量给出轴向；单位圆参数化后再旋转平移即得椭圆。
    注意 np.linalg.eigh 的特征值按升序返回，画图前若需要区分长短轴要
    自行排序（这里不需要，因为半轴长度与特征向量是一一对应的）。
    """
    cov = np.asarray(cov, dtype=np.float64)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)  # 防止数值上出现负特征值
    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)  # (n_points, 2)
    scaled = circle * (n_std * np.sqrt(vals))[None, :]
    return scaled @ vecs.T + np.asarray(mean)[None, :]


def plot_gmm_ellipses(ax, means, covs, n_std=(1.0, 2.0), colors=None, label_prefix="分量"):
    """在 ax 上画出各分量的协方差椭圆（默认 1σ 与 2σ 两圈）。

    n_std 为标量时只画一圈，为序列时逐圈画，外圈用虚线以便区分。
    """
    means = np.atleast_2d(means)
    covs = np.asarray(covs)
    if np.isscalar(n_std):
        n_std = [n_std]
    colors = colors or PALETTE

    for j, (mean, cov) in enumerate(zip(means, covs)):
        color = colors[j % len(colors)]
        for i, s in enumerate(n_std):
            pts = ellipse_points(mean, cov, n_std=s)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=color,
                lw=1.6 if i == 0 else 1.2,
                ls="-" if i == 0 else "--",
                alpha=0.9,
                label=f"{label_prefix}{j} ({s:g}σ)" if i == 0 else None,
            )
    return ax


def plot_clusters(ax=None, X=None, labels=None, centroids=None, title=None, alpha=0.5, s=14):
    """散点图 + 中心十字，用于 k-means 结果或真实标签。"""
    ax, fig = _get_ax(ax)
    X = np.asarray(X)
    labels = np.asarray(labels)
    for j, c in enumerate(np.unique(labels)):
        mask = labels == c
        ax.scatter(X[mask, 0], X[mask, 1], s=s, alpha=alpha, color=PALETTE[j % len(PALETTE)], label=f"簇 {c}")
    if centroids is not None:
        centroids = np.atleast_2d(centroids)
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="X",
            s=180,
            edgecolor="black",
            linewidth=1.2,
            color=[PALETTE[j % len(PALETTE)] for j in range(len(centroids))],
            zorder=5,
            label="中心",
        )
    if title:
        ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(fontsize=8, loc="best")
    return ax


def plot_soft_assignment(ax=None, X=None, resp=None, means=None, covs=None, title=None, s=14):
    """软归属散点图：按最大责任度着色，点的透明度正比于该责任度。

    责任度接近 1 的点几乎不透明，位于两个分量边界上的点（责任度接近
    0.5）呈半透明，这正好把 GMM 与 k-means 的差别画出来。
    """
    ax, fig = _get_ax(ax)
    X = np.asarray(X)
    resp = np.asarray(resp)
    hard = resp.argmax(axis=1)
    conf = resp.max(axis=1)
    alpha = 0.15 + 0.75 * conf
    colors = [PALETTE[j % len(PALETTE)] for j in hard]
    ax.scatter(X[:, 0], X[:, 1], s=s, c=colors, alpha=alpha, edgecolor="none")
    if means is not None and covs is not None:
        plot_gmm_ellipses(ax, means, covs)
        means = np.atleast_2d(means)
        ax.scatter(means[:, 0], means[:, 1], marker="X", s=160, color="black", zorder=5, label="估计均值")
        ax.legend(fontsize=8, loc="best")
    if title:
        ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    return ax


def plot_curve(ax=None, y=None, x=None, label=None, title=None, xlabel="迭代次数", ylabel="", logy=False, marker="o-"):
    """通用曲线图，用于对数似然、inertia、中心位移等序列。"""
    ax, fig = _get_ax(ax, figsize=(6.5, 4.2))
    y = np.asarray(y, dtype=np.float64)
    if x is None:
        x = np.arange(1, len(y) + 1)
    ax.plot(x, y, marker, ms=3.5, lw=1.5, label=label)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label:
        ax.legend(fontsize=9)
    return ax


def plot_convergence_compare(curves, title="不同初始化下的对数似然曲线", xlabel="迭代次数", ylabel="对数似然"):
    """把多条收敛曲线画在同一张图上。

    Parameters
    ----------
    curves : dict[str, sequence]
        键为曲线名称（进入图例），值为序列。
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for i, (name, values) in enumerate(curves.items()):
        values = np.asarray(values, dtype=np.float64)
        ax.plot(
            np.arange(1, len(values) + 1),
            values,
            marker="o",
            ms=3,
            lw=1.4,
            color=PALETTE[i % len(PALETTE)],
            label=name,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    return fig


def plot_parameter_recovery(true_means, est_means, true_covs, est_covs, title="参数恢复：真实（灰）与估计（彩）"):
    """把真实与估计的均值、协方差椭圆画在一起做对照。"""
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    plot_gmm_ellipses(ax, true_means, true_covs, colors=["#BBBBBB"] * len(np.atleast_2d(true_means)), label_prefix="真实")
    plot_gmm_ellipses(ax, est_means, est_covs, label_prefix="估计")
    true_means = np.atleast_2d(true_means)
    est_means = np.atleast_2d(est_means)
    ax.scatter(true_means[:, 0], true_means[:, 1], marker="x", s=120, color="#666666", label="真实均值", zorder=6)
    ax.scatter(est_means[:, 0], est_means[:, 1], marker="X", s=150, color="black", label="估计均值", zorder=6)
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(fontsize=8)
    return fig


def plot_ari_bar(names, ari_values, title="ARI 对比", ylabel="调整兰德指数 (ARI)"):
    """数据集/方法对比的柱状图。"""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(names))
    bars = ax.bar(x, ari_values, color=[PALETTE[i % len(PALETTE)] for i in range(len(names))], width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, color="black", lw=0.8)
    for bar, v in zip(bars, ari_values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    return fig
