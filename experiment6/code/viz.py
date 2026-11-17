"""绘图模块：报告里的全部插图都在这里生成（统一字体、配色、尺寸）。

图片统一存到 figures/ 目录，文件名与报告里的 "figXX_" 引用一一对应。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.size"] = 10

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

FIG_DIR: Path | None = None


def set_fig_dir(path: Path) -> None:
    global FIG_DIR
    FIG_DIR = Path(path)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig, name: str, tight: bool = True) -> str:
    if FIG_DIR is None:
        raise RuntimeError("请先调用 set_fig_dir()")
    if tight:
        fig.tight_layout()
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out.name}")
    return name


# ---------------------------------------------------------------------------
# 图片网格工具
# ---------------------------------------------------------------------------


def images_to_array(images: torch.Tensor, nrow: int, padding: int = 2,
                    scale: int = 1, resize: str = "nearest") -> np.ndarray:
    """把 [0,1] 的 (N,3,64,64) 张量拼成网格 uint8 数组 (H,W,3)。"""
    from torchvision.utils import make_grid

    grid = make_grid(images.clamp(0, 1), nrow=nrow, padding=padding)
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if scale != 1:
        im = Image.fromarray(arr)
        method = Image.NEAREST if resize == "nearest" else Image.LANCZOS
        arr = np.asarray(im.resize((arr.shape[1] * scale, arr.shape[0] * scale), method))
    return arr


def _show(ax, images: torch.Tensor, nrow: int, scale: int = 1, resize: str = "nearest"):
    ax.imshow(images_to_array(images, nrow=nrow, scale=scale, resize=resize))
    ax.axis("off")


def grid_figure(images: torch.Tensor, path: str, nrow: int = 8, scale: int = 2,
                title: str | None = None, figsize: tuple[float, float] | None = None,
                resize: str = "nearest") -> str:
    """单张简单网格图（数据集样本、最终生成画廊等）。"""
    arr = images_to_array(images, nrow=nrow, scale=scale, resize=resize)
    h, w = arr.shape[:2]
    if figsize is None:
        figsize = (w / 100, h / 100 + (0.3 if title else 0))
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    return savefig(fig, path)


def row_grid_figure(rows: dict[str, torch.Tensor], path: str, n_cols: int = 8,
                    scale: int = 2, title: str | None = None,
                    resize: str = "nearest") -> str:
    """多行对比图：每一行是一个配置/时刻，左边标注名称。

    注意：子图会调用 axis("off")，此时 set_ylabel 不会显示，必须用 text +
    transAxes + clip_on=False 才能把行标签画在坐标轴外。
    """
    n_rows = len(rows)
    # 每个格子的显示尺寸约 0.78×0.86 英寸：太小会糊，太大浪费版面
    fig_w = n_cols * 0.78 + 1.3
    fig_h = n_rows * 0.86 + (0.45 if title else 0.1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    for r, (label, imgs) in enumerate(rows.items()):
        imgs = imgs[:n_cols]
        arrs = [np.asarray(Image.fromarray(images_to_array(imgs[i:i + 1], nrow=1, padding=0,
                                                           scale=scale, resize=resize)))
                for i in range(len(imgs))]
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(arrs):
                ax.imshow(arrs[c])
        axes[r][0].text(-0.06, 0.5, label, transform=axes[r][0].transAxes,
                        ha="right", va="center", fontsize=9, clip_on=False)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.12, right=0.995, wspace=0.03, hspace=0.05)
    return savefig(fig, path, tight=False)


def evolution_figure(snapshots: dict[int, torch.Tensor], path: str, n_cols: int = 8,
                     title: str | None = None) -> str:
    """训练过程演化图：每行是某个优化步的固定噪声样本，观察生成质量的变化。"""
    steps = sorted(snapshots)
    fig, axes = plt.subplots(len(steps), n_cols,
                             figsize=(n_cols * 1.05, len(steps) * 1.2), squeeze=False)
    for r, step in enumerate(steps):
        imgs = snapshots[step][:n_cols]
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(imgs):
                ax.imshow(images_to_array(imgs[c:c + 1], nrow=1, padding=0, scale=2))
        axes[r][0].text(-0.06, 0.5, f"step {step}", transform=axes[r][0].transAxes,
                        ha="right", va="center", fontsize=8, clip_on=False)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.10, right=0.995, wspace=0.03, hspace=0.04)
    return savefig(fig, path, tight=False)


# ---------------------------------------------------------------------------
# 数据概览
# ---------------------------------------------------------------------------


def plot_data_stats(images_uint8: torch.Tensor, indices: np.ndarray, stats: dict,
                    path: str, n: int = 2000, seed: int = 0) -> str:
    """数据分布：RGB 通道均值/标准差、亮度直方图、饱和度和色相分布。"""
    rng = np.random.RandomState(seed)
    idx = rng.choice(indices, size=min(n, len(indices)), replace=False)
    x = images_uint8[torch.as_tensor(idx)].float().div_(255.0)
    brightness = x.mean(dim=1).flatten().numpy()
    mx = x.max(dim=1).values
    mn = x.min(dim=1).values
    sat = ((mx - mn) / mx.clamp_min(1e-6)).flatten().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    names = ["R", "G", "B"]
    axes[0].bar(names, stats["mean_rgb"], color=PALETTE[:3], alpha=0.85,
                yerr=stats["std_rgb"], capsize=4)
    axes[0].set_title("RGB 通道均值（误差棒为标准差）")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("像素值（0~1）")
    for i, v in enumerate(stats["mean_rgb"]):
        axes[0].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].hist(brightness, bins=40, color=PALETTE[0], alpha=0.85)
    axes[1].set_title("图像平均亮度分布")
    axes[1].set_xlabel("亮度")
    axes[1].set_ylabel("图像数")

    axes[2].hist(sat, bins=40, color=PALETTE[2], alpha=0.85)
    axes[2].set_title("像素饱和度分布")
    axes[2].set_xlabel("饱和度 (max−min)/max")
    axes[2].set_ylabel("像素数")
    return savefig(fig, path)


# ---------------------------------------------------------------------------
# GAN 训练曲线与诊断
# ---------------------------------------------------------------------------


def plot_gan_training(history: dict, path: str, title: str = "DCGAN 训练曲线",
                      fid_points: list[tuple[int, float]] | None = None) -> str:
    """四联图：损失、D 输出概率、D 判别准确率、梯度范数（必要时附加 FID 曲线）。"""
    n_cols = 4 if not fid_points else 5
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 3.4))
    steps = history["step"]
    ax = axes[0]
    ax.plot(steps, history["loss_d"], color=PALETTE[0], label="判别器损失")
    ax.plot(steps, history["loss_g"], color=PALETTE[1], label="生成器损失")
    ax.set_title("训练损失")
    ax.set_xlabel("优化步")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(steps, history["d_real"], color=PALETTE[2], label="D(真实图)")
    ax.plot(steps, history["d_fake"], color=PALETTE[3], label="D(生成图)")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_title("判别器输出概率")
    ax.set_xlabel("优化步")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(steps, history["d_acc_real"], color=PALETTE[2], label="真实图判对率")
    ax.plot(steps, history["d_acc_fake"], color=PALETTE[3], label="生成图判对率")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_title("判别器准确率（训练批次上）")
    ax.set_xlabel("优化步")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)

    ax = axes[3]
    ax.semilogy(steps, history["grad_norm_g"], color=PALETTE[1], label="G 梯度范数")
    ax.semilogy(steps, history["grad_norm_d"], color=PALETTE[0], label="D 梯度范数")
    ax.set_title("梯度范数（对数轴）")
    ax.set_xlabel("优化步")
    ax.legend(fontsize=8)

    if fid_points:
        ax = axes[4]
        xs = [p[0] for p in fid_points]
        ys = [p[1] for p in fid_points]
        ax.plot(xs, ys, "o-", color=PALETTE[4])
        for x, y in fid_points:
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6),
                        fontsize=8, ha="center")
        ax.set_title("FID 随训练的变化（越低越好）")
        ax.set_xlabel("优化步")
        ax.set_ylabel("FID")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return savefig(fig, path, tight=False)


def plot_config_curves(histories: dict[str, dict], path: str, title: str,
                       metric: str = "loss_g", metric_label: str = "生成器损失") -> str:
    """多组实验的同一条曲线对比（激活函数、Dropout 等）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, (label, hist) in enumerate(histories.items()):
        color = PALETTE[i % len(PALETTE)]
        axes[0].plot(hist["step"], hist[metric], color=color, label=label)
        axes[1].plot(hist["step"], hist["d_real"], color=color, label=f"{label}·真实")
        axes[1].plot(hist["step"], hist["d_fake"], color=color, ls="--", label=f"{label}·生成")
    axes[0].set_title(metric_label)
    axes[0].set_xlabel("优化步")
    axes[0].legend(fontsize=7)
    axes[1].set_title("判别器输出概率（实线=真实，虚线=生成）")
    axes[1].set_xlabel("优化步")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return savefig(fig, path, tight=False)


def plot_fid_bars(labels: list[str], values: list[float], path: str, title: str,
                  ylabel: str = "FID（越低越好）", colors: list[str] | None = None,
                  log: bool = False, figsize: tuple[float, float] | None = None) -> str:
    fig, ax = plt.subplots(figsize=figsize or (max(6, 1.25 * len(labels)), 3.8))
    bars = ax.bar(labels, values, color=colors or PALETTE[:len(labels)], alpha=0.88)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return savefig(fig, path)


# ---------------------------------------------------------------------------
# 数据量 / 记忆化分析
# ---------------------------------------------------------------------------


def plot_data_size_results(sizes: list[int], fids: list[float],
                           nn_real: list[dict], nn_gen: dict, path: str) -> str:
    """数据量实验：FID 随数据量变化 + 生成图/真实图到各自训练集的最近邻距离。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax = axes[0]
    ax.plot(sizes, fids, "o-", color=PALETTE[0], lw=2)
    for x, y in zip(sizes, fids):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    fontsize=9, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("训练图片数（对数轴）")
    ax.set_ylabel("FID（越低越好）")
    ax.set_title("FID 随训练数据量的变化")

    ax = axes[1]
    xs = np.arange(len(sizes))
    gen_mean = [nn_gen[s]["mean"] for s in sizes]
    gen_std = [nn_gen[s]["std"] for s in sizes]
    real_mean = [nn_real[i]["mean"] for i in range(len(sizes))]
    real_std = [nn_real[i]["std"] for i in range(len(sizes))]
    ax.bar(xs - 0.18, gen_mean, width=0.36, yerr=gen_std, capsize=3,
           color=PALETTE[0], label="生成图 → 训练集")
    ax.bar(xs + 0.18, real_mean, width=0.36, yerr=real_std, capsize=3,
           color=PALETTE[7], label="真实留出图 → 训练集（基线）")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s} 张" for s in sizes])
    ax.set_ylabel("最近邻 L2 距离（均值）")
    ax.set_title("生成图到训练集的最近邻距离（越低越像背图）")
    ax.legend(fontsize=8)
    for i in range(len(sizes)):
        ratio = gen_mean[i] / max(real_mean[i], 1e-9)
        ax.text(xs[i], max(gen_mean[i], real_mean[i]) + 1.0, f"比值 {ratio:.2f}",
                ha="center", fontsize=8)
    return savefig(fig, path)


# ---------------------------------------------------------------------------
# 潜空间插值 / VAE
# ---------------------------------------------------------------------------


def plot_interpolation(images: torch.Tensor, path: str, title: str = "潜空间线性插值",
                       scale: int = 2) -> str:
    arr = images_to_array(images, nrow=len(images), padding=2, scale=scale)
    fig, ax = plt.subplots(figsize=(len(images) * 1.25, 1.6))
    ax.imshow(arr)
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    return savefig(fig, path)


def plot_reconstruction(real: torch.Tensor, recon: torch.Tensor, path: str,
                        title: str = "VAE 重建：上排原图，下排重建") -> str:
    n = len(real)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.15 + 0.6, 2.9))
    for i in range(n):
        for r, imgs in ((0, real), (1, recon)):
            ax = axes[r][i]
            ax.imshow(images_to_array(imgs[i:i + 1], nrow=1, padding=0, scale=2,
                                      resize="lanczos"))
            ax.axis("off")
    for r, label in ((0, "原图"), (1, "重建")):
        axes[r][0].text(-0.06, 0.5, label, transform=axes[r][0].transAxes,
                        ha="right", va="center", fontsize=10, clip_on=False)
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.10, wspace=0.04, hspace=0.02)
    return savefig(fig, path, tight=False)


def plot_vae_training(history: dict, path: str, title: str = "VAE 训练曲线") -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    steps = history["step"]
    ax = axes[0]
    ax.plot(steps, history["loss"], color=PALETTE[0], label="总损失 ELBO")
    ax.plot(steps, history["recon"], color=PALETTE[1], label="重建损失")
    ax.plot(steps, history["kl"], color=PALETTE[2], label="KL 项")
    ax.set_title("损失（按 batch 平均，像素和）")
    ax.set_xlabel("优化步")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(steps, history["recon_per_pixel"], color=PALETTE[1])
    ax.set_title("重建损失 / 像素（MSE）")
    ax.set_xlabel("优化步")

    ax = axes[2]
    ax.plot(steps, history["kl_per_dim"], color=PALETTE[2])
    ax.set_title("KL / 潜变量维度")
    ax.set_xlabel("优化步")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return savefig(fig, path, tight=False)


def plot_gan_vae_compare(fid_dcgan: float, fid_vae: float,
                         c2st_dcgan: float, c2st_vae: float, path: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(["DCGAN", "VAE"], [fid_dcgan, fid_vae],
                color=[PALETTE[0], PALETTE[3]], alpha=0.88)
    for i, v in enumerate([fid_dcgan, fid_vae]):
        axes[0].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("FID（越低越好）")
    axes[0].set_title("生成质量：DCGAN vs VAE")

    axes[1].bar(["DCGAN", "VAE"], [c2st_dcgan, c2st_vae],
                color=[PALETTE[0], PALETTE[3]], alpha=0.88)
    axes[1].axhline(0.5, color="gray", ls="--", lw=0.9, label="随机猜测（0.5）")
    for i, v in enumerate([c2st_dcgan, c2st_vae]):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    axes[1].set_ylim(0.4, 1.0)
    axes[1].set_ylabel("C2ST 测试准确率（1.0 = 完全可分）")
    axes[1].set_title("C2ST：小 CNN 区分真实/生成样本的准确率")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return savefig(fig, path, tight=False)


# ---------------------------------------------------------------------------
# 网络结构示意图
# ---------------------------------------------------------------------------


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
                                linewidth=1.0, edgecolor="#444444", facecolor=color, alpha=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 linewidth=0.9, color="#555555"))


TYPE_CN = {
    "ConvTranspose2d": "转置卷积", "Conv2d": "卷积", "BatchNorm2d": "批归一化",
    "Linear": "全连接", "ReLU": "ReLU", "LeakyReLU": "LeakyReLU", "Tanh": "Tanh",
    "Dropout2d": "Dropout", "Sigmoid": "Sigmoid", "MaxPool2d": "最大池化",
}


def _layer_text(layer: dict) -> str:
    tp = TYPE_CN.get(layer["type"], layer["type"])
    shape = layer.get("out_shape")
    if not shape:
        return tp
    if len(shape) == 3:
        return f"{tp}  {shape[0]}×{shape[1]}×{shape[2]}"
    return f"{tp}  {shape[0]}"


def plot_arch_diagram(g_layers: list[dict], d_layers: list[dict], path: str) -> str:
    """把生成器（左）和判别器（右）的逐层尺寸画成示意框图。"""
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1)

    def draw_column(layers, x_center, title, color_a, color_b):
        n = len(layers)
        y0, y1 = 0.10, 0.86
        step = (y1 - y0) / max(n, 1)
        h = step * 0.68
        ax.text(x_center, 0.95, title, ha="center", va="center", fontsize=12, weight="bold")
        for i, layer in enumerate(layers):
            y = y0 + (n - 1 - i) * step
            color = color_a if i % 2 == 0 else color_b
            _box(ax, x_center - 0.13, y, 0.26, h, _layer_text(layer), color)
            if i < n - 1:
                _arrow(ax, x_center, y + h, x_center, y + step + 0.003)

    draw_column(g_layers, 0.27, "Generator（G）", "#DCE6F2", "#B8CCE4")
    draw_column(d_layers, 0.73, "Discriminator（D）", "#FDE2CC", "#F8C9A4")

    ax.text(0.5, -0.03, "G：噪声 z(100,1,1) → 4×4 → 8×8 → 16×16 → 32×32 → 64×64×3（tanh）\n"
                        "D：64×64×3 → 32×32 → 16×16 → 8×8 → 4×4 → 1 个 logit",
            ha="center", va="top", fontsize=9, color="#333333")
    return savefig(fig, path)


def plot_vae_arch(enc_layers: list[dict], dec_layers: list[dict], path: str) -> str:
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1)

    def draw_column(layers, x_center, title, color_a, color_b):
        n = len(layers)
        y0, y1 = 0.10, 0.86
        step = (y1 - y0) / max(n, 1)
        h = step * 0.68
        ax.text(x_center, 0.95, title, ha="center", va="center", fontsize=12, weight="bold")
        for i, layer in enumerate(layers):
            y = y0 + (n - 1 - i) * step
            _box(ax, x_center - 0.14, y, 0.28, h, _layer_text(layer),
                 color_a if i % 2 == 0 else color_b)
            if i < n - 1:
                _arrow(ax, x_center, y + h, x_center, y + step + 0.003)

    draw_column(enc_layers, 0.27, "Encoder（q(z|x)）", "#DCE6F2", "#B8CCE4")
    draw_column(dec_layers, 0.73, "Decoder（p(x|z)）", "#E2F0D9", "#C6E0B4")
    ax.text(0.5, -0.03, "编码器 64→32→16→8→4，输出 μ 与 logσ²；重参数化采样 z；解码器 4→8→16→32→64，输出 logits",
            ha="center", va="top", fontsize=9, color="#333333")
    return savefig(fig, path)


# ---------------------------------------------------------------------------
# 运行日志"截图"
# ---------------------------------------------------------------------------


def render_terminal_log(lines: list[str], path: str, title: str = "训练日志",
                        max_lines: int = 32) -> str:
    """把训练日志渲染成终端风格的图片（报告里的"运行截图"）。"""
    shown = lines[-max_lines:]
    fig, ax = plt.subplots(figsize=(10, 0.22 * len(shown) + 0.5))
    ax.set_facecolor("#1E1E1E")
    fig.patch.set_facecolor("#1E1E1E")
    ax.axis("off")
    # 中文字符需要 CJK 等宽字体，DejaVu Sans Mono 没有中文会导致方框
    mono_family = ["Noto Sans Mono CJK SC", "Noto Sans CJK SC", "DejaVu Sans Mono"]
    for i, line in enumerate(shown):
        ax.text(0.01, 1 - (i + 0.7) / (len(shown) + 0.6), line,
                family=mono_family, fontsize=7.2, color="#D4D4D4",
                transform=ax.transAxes, va="center")
    ax.set_title(title, color="#CCCCCC", fontsize=9, loc="left")
    return savefig(fig, path)
