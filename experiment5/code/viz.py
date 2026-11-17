"""绘图模块：训练曲线、样本网格、混淆矩阵、错分样本、卷积核/特征图、结构示意图等。

统一字体、配色与保存方式，报告里的插图全部由 exp5_report/full_experiment.py
调用这里的函数生成，保证图文一一对应。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 中文字体：Linux 上一般都有 Noto Sans CJK SC；没有时退回默认字体（中文会显示成方框）
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.size"] = 10

# 统一的颜色（按顺序分配给各方法，报告里同一方法颜色一致）
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

FIG_DIR: Path | None = None


def set_fig_dir(path: Path) -> None:
    global FIG_DIR
    FIG_DIR = Path(path)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig, name: str, tight: bool = True) -> Path:
    """保存图片到 FIG_DIR，并关闭 figure。"""
    if FIG_DIR is None:
        raise RuntimeError("请先调用 set_fig_dir()")
    if tight:
        fig.tight_layout()
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out.name}")
    return out


# ---------------------------------------------------------------------------
# 训练曲线
# ---------------------------------------------------------------------------


def plot_curves(histories: dict[str, dict], paths: tuple[str, ...], out_name: str,
                title: str = "", ylim: tuple[float, float] | None = None,
                ylabel: str = "准确率", logy: bool = False) -> None:
    """把多组实验的训练/验证曲线画在一张图上。

    histories: {曲线名: fit() 返回的 history 字典}
    paths: 画哪些子图，如 ("val_acc", "train_loss")
    """
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.6), squeeze=False)
    metric_titles = {
        "train_acc": "训练准确率", "val_acc": "验证准确率",
        "train_loss": "训练损失", "val_loss": "验证损失",
    }
    for ax, path in zip(axes[0], paths):
        for i, (label, hist) in enumerate(histories.items()):
            ys = hist[path]
            xs = hist["epoch"] if "epoch" in hist else np.arange(1, len(ys) + 1)
            ax.plot(xs, ys, label=label, color=PALETTE[i % len(PALETTE)],
                    linewidth=1.8, marker="o", markersize=3)
        ax.set_xlabel("训练轮次 (epoch)")
        ax.set_ylabel(("对数尺度 " if logy else "") + metric_titles.get(path, path))
        ax.set_title(metric_titles.get(path, path))
        if logy:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        if path.endswith("acc"):
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title, y=1.02)
    savefig(fig, out_name)


def plot_grouped_final(rows: list[dict], out_name: str, title: str = "",
                       ylim: tuple[float, float] | None = None,
                       xlabel: str = "", rotate: int = 0) -> None:
    """柱状图：每组给一个均值（可带误差棒）和数值标注。"""
    labels = [r["label"] for r in rows]
    means = np.array([r["value"] for r in rows])
    errs = np.array([r.get("std", 0.0) for r in rows])
    fig, ax = plt.subplots(figsize=(max(5.0, 1.05 * len(rows) + 2.0), 3.8))
    bars = ax.bar(range(len(rows)), means, yerr=errs if errs.any() else None,
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(rows))],
                  capsize=4, edgecolor="white")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.004 if ylim and ylim[1] <= 1 else 0.3),
                f"{v * 100:.2f}%" if ylim and ylim[1] <= 1 else f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=9)
    ax.set_ylabel("测试准确率" if ylim and ylim[1] <= 1 else "")
    ax.set_xlabel(xlabel)
    if ylim:
        ax.set_ylim(*ylim)
        if ylim[1] <= 1.0:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    if title:
        ax.set_title(title)
    savefig(fig, out_name)


# ---------------------------------------------------------------------------
# 图像数据
# ---------------------------------------------------------------------------


def plot_sample_grid(X: torch.Tensor, y: torch.Tensor, class_names: list[str],
                     out_name: str, n_per_class: int = 5, title: str = "") -> None:
    """每类抽 n_per_class 张图，画成 (类别数 × n) 的网格。"""
    X = X.cpu()
    y = y.cpu()
    classes = sorted(torch.unique(y).tolist())
    fig, axes = plt.subplots(len(classes), n_per_class,
                             figsize=(1.25 * n_per_class + 0.5, 1.3 * len(classes) + 0.3))
    g = torch.Generator().manual_seed(0)
    for r, c in enumerate(classes):
        idx = torch.nonzero(y == c, as_tuple=False).flatten()
        idx = idx[torch.randperm(len(idx), generator=g)[:n_per_class]]
        for k, i in enumerate(idx):
            ax = axes[r, k]
            ax.imshow(X[i, 0], cmap="gray")
            ax.axis("off")
            if k == 0:
                ax.set_ylabel(class_names[c] if c < len(class_names) else str(c),
                              rotation=0, labelpad=18, fontsize=9, va="center")
    if title:
        fig.suptitle(title, y=1.01)
    savefig(fig, out_name)


def plot_class_distribution(y: torch.Tensor, class_names: list[str], out_name: str) -> None:
    counts = torch.bincount(y.cpu(), minlength=len(class_names)).tolist()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(class_names, counts, color=PALETTE[0])
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.01, str(v), ha="center", fontsize=8)
    ax.set_ylabel("样本数")
    ax.set_title("训练集类别分布")
    savefig(fig, out_name)


# ---------------------------------------------------------------------------
# 混淆矩阵与错分样本
# ---------------------------------------------------------------------------


def plot_confusion(cm: np.ndarray, class_names: list[str], out_name: str) -> None:
    """画行归一化的混淆矩阵（百分比），并标注百分比数值。"""
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, f"{cm_norm[i, j] * 100:.1f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if cm_norm[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="行归一化比例")
    ax.set_title("测试集混淆矩阵（数字为行百分比）")
    savefig(fig, out_name)


def plot_errors(X: torch.Tensor, y_true: np.ndarray, y_pred: np.ndarray,
                conf: np.ndarray, class_names: list[str], out_name: str,
                n: int = 16, seed: int = 0) -> None:
    """随机抽 n 个错分样本，标题标出 真实→预测 与置信度。"""
    wrong = np.flatnonzero(y_true != y_pred)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(wrong, size=min(n, len(wrong)), replace=False)
    ncols = 8
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.35 * ncols + 0.4, 1.55 * nrows + 0.3))
    axes = np.atleast_2d(axes)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k >= len(chosen):
            continue
        i = chosen[k]
        ax.imshow(X[i, 0].cpu(), cmap="gray")
        ax.set_title(f"{class_names[y_true[i]]}→{class_names[y_pred[i]]}\n{conf[i] * 100:.0f}%",
                     fontsize=8, color="#C44E52")
    fig.suptitle(f"测试集中的错分样本（共 {len(wrong)} 张，随机展示 {len(chosen)} 张）", y=1.01)
    savefig(fig, out_name)


def plot_confidence(stats: dict, out_name: str) -> None:
    """两个子图：正确/错误预测的置信度分布 + 置信度阈值下的覆盖率-精度曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    centers = np.arange(0.05, 1.0, 0.1)
    ax = axes[0]
    ax.bar(centers, np.array(stats["conf_hist_correct"]) / max(sum(stats["conf_hist_correct"]), 1),
           width=0.08, label="预测正确", color=PALETTE[2], alpha=0.85)
    ax.bar(centers + 0.08, np.array(stats["conf_hist_wrong"]) / max(sum(stats["conf_hist_wrong"]), 1),
           width=0.08, label="预测错误", color=PALETTE[3], alpha=0.85)
    ax.set_xlabel("预测置信度（softmax 最大概率）")
    ax.set_ylabel("组内占比")
    ax.set_yscale("log")
    ax.set_title("正确与错误预测的置信度分布")
    ax.legend()
    ax = axes[1]
    ax.plot(stats["coverage"], stats["precision_at_threshold"], color=PALETTE[0], linewidth=2)
    ax.set_xlabel("覆盖率（置信度 ≥ 阈值的样本占比）")
    ax.set_ylabel("覆盖样本中的准确率")
    ax.set_title("按置信度拒绝预测：覆盖率 vs 精度")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    savefig(fig, out_name)


# ---------------------------------------------------------------------------
# 卷积核与特征图
# ---------------------------------------------------------------------------


def plot_filters(conv1_weight: torch.Tensor, out_name: str, max_show: int = 16) -> None:
    """画第一层卷积核（权重）。自动适配 6 个 5×5（LeNet-5）或 32 个 3×3（SmallVGG）。"""
    w = conv1_weight.detach().cpu().float()
    n = w.shape[0]
    n_show = min(n, max_show)
    ncols = min(8, n_show)
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.35 * ncols, 1.65 * nrows), squeeze=False)
    vmax = float(w.abs().max())
    for k in range(n_show):
        ax = axes[k // ncols, k % ncols]
        ax.imshow(w[k, 0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"核 {k}", fontsize=8)
        ax.axis("off")
    for k in range(n_show, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.suptitle(f"第一层卷积核（共 {n} 个，展示前 {n_show} 个；红=正权重，蓝=负权重）", y=1.02)
    savefig(fig, out_name)


def plot_feature_maps(model, x: torch.Tensor, out_name: str, max_per_layer: int = 8) -> None:
    """输入一张数字图，用 forward hook 抓取第一个和最后一个卷积层的输出并画出部分通道。"""
    device = next(model.parameters()).device
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("模型里没有卷积层")
    acts: dict[int, torch.Tensor] = {}
    hooks = [
        c.register_forward_hook(lambda mod, inp, out, i=i: acts.__setitem__(i, out.detach()))
        for i, c in enumerate(convs)
    ]
    model.eval()
    with torch.no_grad():
        model(x[:1].to(device))
    for h in hooks:
        h.remove()
    picks = [0] if len(convs) == 1 else [0, len(convs) - 1]
    fig, axes = plt.subplots(len(picks), max_per_layer + 1,
                             figsize=(1.15 * (max_per_layer + 1), 1.5 * len(picks) + 1.6))
    axes = np.atleast_2d(axes)
    for r, idx in enumerate(picks):
        fm = acts[idx][0]
        if fm.dim() == 3:  # (C,H,W)，去掉可能存在的 batch 维
            fm = fm
        axes[r, 0].imshow(x[0, 0].cpu(), cmap="gray")
        axes[r, 0].set_ylabel(f"卷积层 {idx + 1}\n输出 {fm.shape[0]} 通道", fontsize=8)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        axes[r, 0].set_title("输入", fontsize=9)
        for k in range(max_per_layer):
            ax = axes[r, k + 1]
            if k < fm.shape[0]:
                ax.imshow(fm[k].cpu(), cmap="viridis")
            ax.axis("off")
            ax.set_title(f"ch{k}", fontsize=8)
    fig.suptitle("卷积层的特征图：浅层保留笔画细节，深层变成局部抽象特征", y=1.02)
    savefig(fig, out_name)


# ---------------------------------------------------------------------------
# LeNet-5 结构示意图
# ---------------------------------------------------------------------------


def plot_architecture(out_name: str) -> None:
    """手绘 LeNet-5 的结构示意图：每个层的尺寸与输出张量形状。"""
    fig, ax = plt.subplots(figsize=(14, 4.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    boxes = [
        ("输入\n1×28×28", 0.35, 1.15, 1.35, 2.1, "#E8EDF5"),
        ("conv1\n6@5×5\n→6×24×24", 2.05, 1.35, 1.55, 1.7, "#D6E4F0"),
        ("maxpool\n2×2\n→6×12×12", 3.95, 1.35, 1.55, 1.7, "#D6E4F0"),
        ("conv2\n16@5×5\n→16×8×8", 5.85, 1.35, 1.55, 1.7, "#D6E4F0"),
        ("maxpool\n2×2\n→16×4×4", 7.75, 1.35, 1.55, 1.7, "#D6E4F0"),
        ("flatten\n256", 9.55, 1.6, 1.15, 1.2, "#F3E8D6"),
        ("fc\n120", 11.0, 1.6, 1.0, 1.2, "#F3E8D6"),
        ("fc\n84", 12.25, 1.6, 0.85, 1.2, "#F3E8D6"),
        ("fc\n10", 13.35, 1.6, 0.55, 1.2, "#F3E8D6"),
    ]
    for text, x, y, w, h, color in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor=color, edgecolor="#4A4A4A", linewidth=1.0)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)
    for i in range(len(boxes) - 1):
        _, x1, y1, w1, h1, _ = boxes[i]
        _, x2, y2, w2, h2, _ = boxes[i + 1]
        arr = FancyArrowPatch((x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2),
                              arrowstyle="-|>", mutation_scale=12, color="#555555",
                              linewidth=1.1)
        ax.add_patch(arr)
    ax.text(0.35, 0.55, "参数量：conv1 156 + conv2 2,416 + fc 30,840 + 10,164 + 850 = 44,426",
            fontsize=9, color="#333333")
    ax.text(0.35, 0.05, "池化后特征图尺寸按 28→24→12→8→4 逐层缩小；全连接层输出 logits，"
                        "训练用交叉熵损失", fontsize=9, color="#333333")
    ax.set_title("LeNet-5（28×28 适配版）结构示意图", fontsize=12)
    savefig(fig, out_name)
