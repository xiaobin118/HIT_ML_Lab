#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩散模型报告素材脚本：训练结束后运行，生成新增插图并更新 results.json。

输入：diffusion_run/（训练日志、EMA checkpoint、eval_results.json）
输出：
    figures/fig29 ~ fig36（扩散模型结构、训练曲线、样本演化、DDIM 步数对比、
    最终画廊、去噪轨迹、三模型样本对比、三模型指标对比）
    results.json 中的 diffusion 字段
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics  # noqa: E402
import viz  # noqa: E402
from data import load_clean_images, split_indices  # noqa: E402
from diffusion import (GaussianDiffusion, UNet, sample_ddim,  # noqa: E402
                       sample_ddim_trajectory)
from engine import sample_gan  # noqa: E402
from models import Generator, VAE  # noqa: E402

RUN = HERE / "diffusion_run"
FIG_DIR = HERE / "figures"
RESULTS = HERE / "results.json"
MEM_FRACTION = 0.72


def load_best_model(device):
    ev = json.loads((RUN / "eval_results.json").read_text(encoding="utf-8"))
    ckpt = RUN / "ema" / ev["best"]["ckpt"]
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = blob["cfg"]
    model = UNet(base=cfg["base"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(blob["ema"])
    model.eval()
    return model, cfg, ev


# ---------------------------------------------------------------------------
# 图 29：U-Net 结构示意图
# ---------------------------------------------------------------------------


def fig_arch(cfg):
    base = cfg["base"]
    c = [base, base * 2, base * 2, base * 4]
    levels = [
        ("输入", "64×64×3", "#EEEEEE"),
        ("Conv 3×3", f"{c[0]}×64×64", "#DCE6F2"),
        ("Down 0：2×ResBlock", f"{c[0]}×64×64", "#B8CCE4"),
        ("Down 1：2×ResBlock", f"{c[1]}×32×32", "#B8CCE4"),
        ("Down 2：2×ResBlock + 自注意力", f"{c[2]}×16×16", "#B8CCE4"),
        ("Down 3：2×ResBlock + 自注意力", f"{c[3]}×8×8", "#9DB8D8"),
        ("中间层：ResBlock + 自注意力 + ResBlock", f"{c[3]}×8×8", "#9DB8D8"),
        ("Up 3：3×ResBlock + 自注意力", f"{c[3]}×8×8", "#FDE2CC"),
        ("Up 2：3×ResBlock + 自注意力", f"{c[2]}×16×16", "#F8C9A4"),
        ("Up 1：3×ResBlock", f"{c[1]}×32×32", "#F8C9A4"),
        ("Up 0：3×ResBlock", f"{c[0]}×64×64", "#F8C9A4"),
        ("输出 Conv 3×3（零初始化）", "ε 预测 3×64×64", "#EEEEEE"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n = len(levels)
    y0, y1 = 0.06, 0.92
    step = (y1 - y0) / n
    h = step * 0.72
    for i, (name, shape, color) in enumerate(levels):
        y = y0 + (n - 1 - i) * step
        viz._box(ax, 0.08, y, 0.84, h, f"{name}    {shape}", color)
        if i < n - 1:
            viz._arrow(ax, 0.5, y, 0.5, y - step * 0.24)
    ax.text(0.5, 0.985, "DDPM U-Net（64×64，18.55M 参数；跳跃连接略去未画）",
            ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.015, "每个 ResBlock：GroupNorm → SiLU → Conv3×3 → 注入时间步嵌入 → GN → SiLU → Dropout → Conv3×3；"
                        "采样时用 EMA 权重", ha="center", va="center", fontsize=8.5, color="#444444")
    viz.savefig(fig, "fig29_diffusion_arch.png")


# ---------------------------------------------------------------------------
# 图 30：训练曲线
# ---------------------------------------------------------------------------


def fig_training(ev):
    recs = [json.loads(line) for line in (RUN / "train_log.jsonl").read_text().splitlines() if line.strip()]
    steps = np.array([r["step"] for r in recs])
    loss = np.array([r["loss"] for r in recs])
    lr = np.array([r["lr"] for r in recs])
    sps = np.array([r["steps_per_s"] for r in recs])
    # 损失做移动平均，避免 50 步窗口的抖动
    k = max(1, len(loss) // 80)
    smooth = np.convolve(loss, np.ones(k) / k, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    ax = axes[0]
    ax.plot(steps, loss, color="#CCCCCC", lw=0.6, label="每 50 步平均")
    ax.plot(steps[k - 1:], smooth, color=viz.PALETTE[0], lw=1.8, label="移动平均")
    ax.set_xlabel("优化步")
    ax.set_ylabel("ε-预测 MSE 损失")
    ax.set_title("训练损失")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(steps, lr, color=viz.PALETTE[2])
    ax.set_xlabel("优化步")
    ax.set_ylabel("学习率")
    ax.set_title("学习率（线性 warmup + 余弦退火）")

    ax = axes[2]
    screen = ev["screening"]
    xs = [s["step"] for s in screen]
    ys = [s["fid"] for s in screen]
    ax.plot(xs, ys, "o-", color=viz.PALETTE[3])
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    fontsize=8, ha="center")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("FID（DDIM-50，2,048 张）")
    ax.set_title("FID 随训练步数的变化")
    fig.suptitle(f"DDPM 训练过程（{cfg_steps_text()}，batch=32）", fontsize=12)
    fig.tight_layout()
    viz.savefig(fig, "fig30_diffusion_training.png", tight=False)


def cfg_steps_text():
    summ = json.loads((RUN / "train_summary.json").read_text(encoding="utf-8"))
    return f"训练 {summ['final_step']:,} 步，{summ['train_seconds'] / 3600:.1f} 小时"


# ---------------------------------------------------------------------------
# 图 31：样本随训练步数演化（同一固定噪声）
# ---------------------------------------------------------------------------


def fig_evolution(ev, device):
    screen = ev["screening"]
    rows = {}
    for rec in screen:
        path = RUN / "ema" / rec["ckpt"]
        blob = torch.load(path, map_location=device, weights_only=False)
        cfg = blob["cfg"]
        model = UNet(base=cfg["base"], dropout=cfg["dropout"]).to(device)
        model.load_state_dict(blob["ema"])
        model.eval()
        imgs = sample_ddim(model, GaussianDiffusion(cfg["T"], "cosine", device), 8,
                           device, steps=50, seed=9527, batch=8)
        rows[f"训练 {rec['step']:,} 步（FID {rec['fid']:.1f}）"] = imgs
        del model
        torch.cuda.empty_cache()
    viz.row_grid_figure(rows, "fig31_diffusion_evolution.png", n_cols=8, scale=2,
                        title="扩散模型：同一组固定噪声在不同训练阶段的生成结果")


# ---------------------------------------------------------------------------
# 图 32：DDIM 步数对比；图 33：最终画廊；图 34：去噪轨迹
# ---------------------------------------------------------------------------


def fig_ddim_and_gallery(ev, model, cfg, device):
    diffusion = GaussianDiffusion(cfg["T"], "cosine", device)
    sweep = {s["steps"]: s for s in ev["ddim_sweep"]}
    rows = {}
    for steps in (20, 50, 100):
        imgs = sample_ddim(model, diffusion, 8, device, steps=steps, seed=4242, batch=8)
        t64 = sweep[steps]["seconds"]
        rows[f"DDIM {steps} 步（FID {sweep[steps]['fid']:.1f}，64 张耗时 {t64:.1f}s）"] = imgs
    viz.row_grid_figure(rows, "fig32_diffusion_ddim_steps.png", n_cols=8, scale=2,
                        title="DDIM 采样步数对质量与速度的影响（同一固定噪声）")

    saved = torch.load(RUN / "samples_final.pt", map_location="cpu", weights_only=False)
    samples = saved["samples"].float().div_(255.0)
    gsteps = saved.get("ddim_steps", 50)
    viz.grid_figure(samples, "fig33_diffusion_gallery.png", nrow=10, scale=3,
                    title=f"扩散模型生成的 80 张二次元头像（DDIM-{gsteps}，FID={ev['final']['fid']:.1f}）",
                    resize="lanczos")

    traj = sample_ddim_trajectory(model, diffusion, n=1, device=device, steps=50, seed=7)
    idx = [0, 7, 14, 21, 28, 35, 42, 49]
    frames = traj[idx]                       # (8,1,3,64,64)
    fig, axes = plt.subplots(1, 8, figsize=(11, 1.9))
    for ax, im, i in zip(axes, frames, idx):
        ax.imshow(viz.images_to_array(im, nrow=1, padding=0, scale=2))
        ax.axis("off")
        ax.set_title(f"t={int(torch.linspace(999, 0, 51)[i])}", fontsize=8)
    fig.suptitle("DDIM 去噪轨迹：从纯噪声（左）逐步生成图像（右）", fontsize=11)
    fig.tight_layout()
    viz.savefig(fig, "fig34_diffusion_trajectory.png", tight=False)


# ---------------------------------------------------------------------------
# 图 35/36：三模型对比
# ---------------------------------------------------------------------------


def three_way(ev, diff_model, diff_cfg, device):
    images, _, _ = load_clean_images()
    _, ref_idx = split_indices(len(images))
    real8 = images[torch.as_tensor(ref_idx[:8])].float().div_(255.0)

    blob = torch.load(HERE / "checkpoints" / "final_G.pt", map_location=device, weights_only=False)
    gcfg = blob["cfg"]
    G = Generator(z_dim=gcfg["z_dim"], base=gcfg["g_base"], activation=gcfg["g_act"]).to(device)
    G.load_state_dict(blob["g"])
    dcgan8 = sample_gan(G, 8, device, seed=321)[:8]
    dcgan500 = sample_gan(G, 500, device, seed=654)
    t0 = time.time()
    _ = sample_gan(G, 256, device, seed=1)
    dcgan_time_per_1k = (time.time() - t0) / 256 * 1000

    vblob = torch.load(HERE / "checkpoints" / "vae_beta1.pt", map_location=device, weights_only=False)
    vcfg = vblob["cfg"]
    V = VAE(z_dim=vcfg["z_dim"], base=vcfg["base"], beta=vcfg["beta"]).to(device)
    V.load_state_dict(vblob["vae"])
    V.eval()
    vae8 = V.sample(8, device, seed=321)
    vae500 = V.sample(500, device, seed=654)
    t0 = time.time()
    _ = V.sample(256, device, seed=1)
    vae_time_per_1k = (time.time() - t0) / 256 * 1000

    diffusion = GaussianDiffusion(diff_cfg["T"], "cosine", device)
    diff8 = sample_ddim(diff_model, diffusion, 8, device, steps=50, seed=321, batch=8)
    t0 = time.time()
    diff500 = sample_ddim(diff_model, diffusion, 500, device, steps=50, seed=654, batch=128)
    diff_time_per_1k = (time.time() - t0) / 500 * 1000

    viz.row_grid_figure({
        "真实图": real8,
        f"DCGAN（FID 92.7）": dcgan8,
        f"VAE β=1（FID 203.1）": vae8,
        f"DDPM 扩散（FID {ev['final']['fid']:.1f}）": diff8,
    }, "fig35_three_way_samples.png", n_cols=8, scale=2,
        title="真实图 vs DCGAN vs VAE vs DDPM 扩散模型")

    # 指标：FID / C2ST / 每千张采样耗时 / 最近邻比值
    R = json.loads(RESULTS.read_text(encoding="utf-8"))
    train_pool = images[torch.as_tensor(split_indices(len(images))[0])].float().div_(255.0)
    nn_dcgan = metrics.min_distances(dcgan500, train_pool, device)
    nn_vae = metrics.min_distances(vae500, train_pool, device)
    nn_real = metrics.min_distances(images[torch.as_tensor(ref_idx[:500])].float().div_(255.0),
                                    train_pool, device)
    nn_real_mean = float(np.mean(nn_real))
    ddpm_c2st = ev["final"]["c2st"]["accuracy"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    ax = axes[0][0]
    names = ["DCGAN", "VAE", "DDPM"]
    fids = [R["final"]["final_fid"], R["final"]["vae_best"]["fid"], ev["final"]["fid"]]
    bars = ax.bar(names, fids, color=[viz.PALETTE[0], viz.PALETTE[3], viz.PALETTE[2]], alpha=0.88)
    for b, v in zip(bars, fids):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("FID（越低越好）")
    ax.set_title("生成质量")

    ax = axes[0][1]
    c2sts = [R["final"]["c2st"]["dcgan"]["accuracy"], R["final"]["c2st"]["vae"]["accuracy"], ddpm_c2st]
    bars = ax.bar(names, c2sts, color=[viz.PALETTE[0], viz.PALETTE[3], viz.PALETTE[2]], alpha=0.88)
    ax.axhline(0.5, color="gray", ls="--", lw=0.9)
    for b, v in zip(bars, c2sts):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("C2ST 准确率（1.0 = 完全可分）")
    ax.set_title("真实/生成可区分性")

    ax = axes[1][0]
    times = [dcgan_time_per_1k, vae_time_per_1k, diff_time_per_1k]
    bars = ax.bar(names, times, color=[viz.PALETTE[0], viz.PALETTE[3], viz.PALETTE[2]], alpha=0.88)
    for b, v in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("生成 1,000 张图耗时（秒，对数轴）")
    ax.set_yscale("log")
    ax.set_title("采样速度（DDPM 用 DDIM-50）")

    ax = axes[1][1]
    ratios = [float(np.mean(nn_dcgan)) / nn_real_mean, float(np.mean(nn_vae)) / nn_real_mean,
              ev["final"]["nn_ratio"]]
    bars = ax.bar(names, ratios, color=[viz.PALETTE[0], viz.PALETTE[3], viz.PALETTE[2]], alpha=0.88)
    ax.axhline(1.0, color="gray", ls="--", lw=0.9, label="真实留出图基线")
    for b, v in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("最近邻距离 / 真实基线")
    ax.set_title("记忆化程度（越小越像背训练图）")
    ax.legend(fontsize=8)
    fig.suptitle("三种生成模型的定量对比（同一 FID 参考集 / 同一批真实图）", fontsize=12)
    fig.tight_layout()
    viz.savefig(fig, "fig36_three_way_metrics.png", tight=False)

    return {
        "dcgan": {"fid": R["final"]["final_fid"], "c2st": R["final"]["c2st"]["dcgan"]["accuracy"],
                  "sample_seconds_per_1k": round(dcgan_time_per_1k, 2),
                  "nn_ratio_vs_real": round(float(np.mean(nn_dcgan)) / nn_real_mean, 4)},
        "vae": {"fid": R["final"]["vae_best"]["fid"], "c2st": R["final"]["c2st"]["vae"]["accuracy"],
                "sample_seconds_per_1k": round(vae_time_per_1k, 2),
                "nn_ratio_vs_real": round(float(np.mean(nn_vae)) / nn_real_mean, 4)},
        "ddpm": {"fid": ev["final"]["fid"], "c2st": ddpm_c2st,
                 "sample_seconds_per_1k": round(diff_time_per_1k, 2),
                 "nn_ratio_vs_real": ev["final"]["nn_ratio"]},
        "real_nn_mean": round(nn_real_mean, 3),
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main():
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    torch.manual_seed(0)
    viz.set_fig_dir(FIG_DIR)
    FIG_LOG = RUN / "figures.log"
    print("加载最佳扩散模型 ...", flush=True)
    model, cfg, ev = load_best_model(device)
    summary = json.loads((RUN / "train_summary.json").read_text(encoding="utf-8"))
    print("生成结构图 ...", flush=True)
    fig_arch(cfg)
    print("生成训练曲线 ...", flush=True)
    fig_training(ev)
    print("生成样本演化图 ...", flush=True)
    fig_evolution(ev, device)
    print("生成 DDIM/画廊/轨迹图 ...", flush=True)
    fig_ddim_and_gallery(ev, model, cfg, device)
    print("生成三模型对比图 ...", flush=True)
    comparison = three_way(ev, model, cfg, device)

    R = json.loads(RESULTS.read_text(encoding="utf-8"))
    R["diffusion"] = {
        "train": {
            "steps": summary["final_step"],
            "minutes": round(summary["train_seconds"] / 60, 1),
            "batch": cfg["batch"], "base": cfg["base"], "params": cfg["params"],
            "T": cfg["T"], "schedule": cfg["schedule"], "ema_decay": cfg["ema_decay"],
            "dropout": cfg["dropout"], "lr": cfg["lr"], "warmup": cfg["warmup"],
            "max_hours": 9.2,
            "stop_reason": summary["stop_reason"],
        },
        "eval": ev,
        "comparison": comparison,
        "figs": {
            "arch": "fig29_diffusion_arch.png",
            "training": "fig30_diffusion_training.png",
            "evolution": "fig31_diffusion_evolution.png",
            "ddim": "fig32_diffusion_ddim_steps.png",
            "gallery": "fig33_diffusion_gallery.png",
            "trajectory": "fig34_diffusion_trajectory.png",
            "samples_compare": "fig35_three_way_samples.png",
            "metrics_compare": "fig36_three_way_metrics.png",
        },
    }
    tmp = RESULTS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(RESULTS)
    print(f"results.json 已更新 diffusion 字段；评估 FID = {ev['final']['fid']:.2f}", flush=True)


if __name__ == "__main__":
    main()
