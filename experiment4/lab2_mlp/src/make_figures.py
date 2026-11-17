"""根据 results.json 生成报告用图。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "results.json"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

C_BLUE = "#06638C"
C_GREEN = "#8BC145"
C_CYAN = "#36AFCE"
C_ORANGE = "#ED7D31"
C_GRAY = "#6B7280"


def load() -> dict:
    return json.loads(RES.read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    path = FIG / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {path}")


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.4, "输入层\n34 维特征", "标准化后输入", C_GRAY),
        (3.0, "隐藏层 1\n64 个神经元", "Linear + ReLU\n(+ Dropout)", C_BLUE),
        (5.8, "隐藏层 2\n64 个神经元", "Linear + ReLU\n(+ Dropout)", C_BLUE),
        (8.6, "输出层\n1 个 logit", "Sigmoid → P(g)", C_GREEN),
    ]
    for x, title, sub, color in boxes:
        ax.add_patch(plt.Rectangle((x, 1.25), 2.2, 1.5, facecolor=color,
                                   edgecolor="none", alpha=0.95, zorder=2))
        ax.text(x + 1.1, 2.25, title, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold", zorder=3)
        ax.text(x + 1.1, 1.65, sub, ha="center", va="center",
                color="white", fontsize=8.5, zorder=3)

    for x in (2.6, 5.2, 8.0):
        ax.annotate("", xy=(x + 0.38, 2.0), xytext=(x - 0.38, 2.0),
                    arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.8))

    ax.text(6.0, 0.55, "34 → 64 → 64 → 1（6,465 个参数）；最优配置 2×32 为 2,209 个参数\n"
            "损失函数 BCEWithLogits，优化器 SGD / Adam", ha="center",
            fontsize=10, color="#374151")
    save(fig, "fig_architecture.png")


def plot_curves(R: dict) -> None:
    hist = R["base"]["histories"][0]
    ep = [h["epoch"] for h in hist]
    tr_loss = [h["train_loss"] for h in hist]
    va_loss = [h["val_loss"] for h in hist]
    tr_acc = [h["train_acc"] for h in hist]
    va_acc = [h["val_acc"] for h in hist]
    best_ep = int(np.argmin(va_loss)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    axes[0].plot(ep, tr_loss, color=C_BLUE, label="训练损失")
    axes[0].plot(ep, va_loss, color=C_ORANGE, label="验证损失")
    axes[0].axvline(best_ep, color=C_GRAY, linestyle="--", linewidth=1,
                    label=f"验证损失最低 (epoch {best_ep})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE 损失")
    axes[0].set_title("损失曲线")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(ep, tr_acc, color=C_BLUE, label="训练准确率")
    axes[1].plot(ep, va_acc, color=C_ORANGE, label="验证准确率")
    axes[1].axvline(best_ep, color=C_GRAY, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("准确率曲线")
    axes[1].set_ylim(0.55, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].annotate(
        f"最终差距 {100 * (tr_acc[-1] - va_acc[-1]):.1f} 个百分点",
        xy=(ep[-1], (tr_acc[-1] + va_acc[-1]) / 2),
        xytext=(ep[-1] - 35, 0.63), fontsize=8.5, color=C_GRAY,
        arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=0.8),
    )
    fig.tight_layout()
    save(fig, "fig_curves.png")


def plot_width_depth(R: dict) -> None:
    width = R["sweeps"]["width"]
    depth = R["sweeps"]["depth"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    x = [r["hidden_dims"][0] for r in width]
    y = [r["open_acc_mean"] * 100 for r in width]
    e = [r["open_acc_std"] * 100 for r in width]
    axes[0].errorbar(x, y, yerr=e, marker="o", color=C_BLUE, capsize=4)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(v) for v in x])
    axes[0].set_xlabel("隐藏层宽度")
    axes[0].set_ylabel("五折开放准确率 (%)")
    axes[0].set_title("宽度的影响（1 个隐藏层）")
    axes[0].grid(alpha=0.3)

    d = [len(r["hidden_dims"]) for r in depth]
    y2 = [r["open_acc_mean"] * 100 for r in depth]
    e2 = [r["open_acc_std"] * 100 for r in depth]
    axes[1].errorbar(d, y2, yerr=e2, marker="s", color=C_GREEN, capsize=4)
    axes[1].set_xticks(d)
    axes[1].set_xlabel("隐藏层层数（宽度 32）")
    axes[1].set_ylabel("五折开放准确率 (%)")
    axes[1].set_title("深度的影响")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_width_depth.png")


def plot_lr_batch(R: dict) -> None:
    lr = R["sweeps"]["lr"]
    batch = R["sweeps"]["batch"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    x = [r["lr"] for r in lr]
    y = [r["open_acc_mean"] * 100 for r in lr]
    e = [r["open_acc_std"] * 100 for r in lr]
    axes[0].errorbar(range(len(x)), y, yerr=e, marker="o", color=C_BLUE, capsize=4)
    axes[0].set_xticks(range(len(x)))
    axes[0].set_xticklabels([f"{v:g}" for v in x])
    axes[0].set_xlabel("学习率（SGD）")
    axes[0].set_ylabel("五折开放准确率 (%)")
    axes[0].set_title("学习率的影响")
    axes[0].grid(alpha=0.3)
    for i, v in enumerate(y):
        axes[0].annotate(f"{v:.1f}", (i, v), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=8)

    bs = [r["batch_size"] for r in batch]
    y2 = [r["open_acc_mean"] * 100 for r in batch]
    e2 = [r["open_acc_std"] * 100 for r in batch]
    axes[1].errorbar(range(len(bs)), y2, yerr=e2, marker="s", color=C_GREEN, capsize=4)
    axes[1].set_xticks(range(len(bs)))
    axes[1].set_xticklabels([str(v) for v in bs])
    axes[1].set_xlabel("Batch 大小（Adam）")
    axes[1].set_ylabel("五折开放准确率 (%)")
    axes[1].set_title("Batch 大小的影响")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_lr_batch.png")


def plot_cv(R: dict) -> None:
    table2 = R["table2"]
    names = [r["name"] for r in table2]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    width = 0.24
    x = np.arange(5)
    colors = [C_CYAN, C_BLUE, C_GREEN]
    for i, row in enumerate(table2):
        vals = [v * 100 for v in row["fold_open_accs"]]
        label = f"{names[i]}（均值 {row['open_acc_mean'] * 100:.2f}% ± {row['open_acc_std'] * 100:.2f}）"
        ax.bar(x + (i - 1) * width, vals, width=width, label=label, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {k + 1}" for k in range(5)])
    ax.set_ylabel("开放准确率 (%)")
    ax.set_ylim(75, 100)
    ax.set_title("三组配置的五折开放准确率")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_cv.png")


def plot_regularization(R: dict) -> None:
    rows = R["regularization"]
    names = [r["name"] for r in rows][::-1]
    vals = [r["open_acc_mean"] * 100 for r in rows][::-1]
    errs = [r["open_acc_std"] * 100 for r in rows][::-1]
    colors = [C_GREEN if n != "Baseline" else C_GRAY for n in names]
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    bars = ax.barh(names, vals, xerr=errs, color=colors, capsize=4)
    ax.set_xlim(85, 98)
    ax.set_xlabel("五折开放准确率 (%)")
    ax.set_title("正则化方法对比")
    ax.grid(alpha=0.3, axis="x")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_regularization.png")


def plot_confusion(R: dict) -> None:
    m = R["best"]["oof_metrics"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["预测 b", "预测 g"])
    ax.set_yticklabels(["真实 b", "真实 g"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "#1F2937")
    ax.set_title("最优配置的 OOF 混淆矩阵")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, "fig_confusion.png")


def plot_error_confidence(R: dict) -> None:
    prob = np.array(R["best"]["oof_prob"])
    # 标签顺序与 results 一致：前 n_neg 个是 b? 不能假设顺序，用 error 统计即可。
    # 这里只画“预测概率距离 0.5 的分布”，按是否分对分两组。
    # 需要 y；从 error_analysis 的 n_pos/n_neg 无法恢复顺序，改用 OOF 误差样本索引。
    # results 中保存了 error_examples，但为了通用，读原始数据重新算。
    from data import load_ionosphere
    X, y = load_ionosphere(ROOT / "data" / "ionosphere.data")
    pred = (prob >= 0.5).astype(int)
    correct = pred == y
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    ax.hist(prob[correct], bins=bins, alpha=0.65, color=C_BLUE, label=f"分对 ({correct.sum()})")
    ax.hist(prob[~correct], bins=bins, alpha=0.75, color=C_ORANGE, label=f"分错 ({(~correct).sum()})")
    ax.axvline(0.5, color=C_GRAY, linestyle="--", linewidth=1)
    ax.set_xlabel("OOF 预测概率 P(g)")
    ax.set_ylabel("样本数")
    n_conf = int(((~correct) & (prob >= 0.9)).sum())
    ax.set_title(f"OOF 预测概率分布（错分 {(~correct).sum()} 例，其中 {n_conf} 例 P(g)≥0.90）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save(fig, "fig_error_confidence.png")


def main() -> None:
    R = load()
    plot_architecture()
    plot_curves(R)
    plot_width_depth(R)
    plot_lr_batch(R)
    plot_cv(R)
    plot_regularization(R)
    plot_confusion(R)
    plot_error_confidence(R)


if __name__ == "__main__":
    main()
