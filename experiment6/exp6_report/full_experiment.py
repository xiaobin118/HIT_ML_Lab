#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验六主实验脚本：用 DCGAN（主实验）与卷积 VAE（对照）生成 64×64 动漫头像。

实验共十个部分，可分部分增量运行，结果写入 results.json，插图写入 figures/：

    1. 数据概览：去重统计、样本网格、亮度/饱和度/通道分布
    2. 模型结构：生成器/判别器/VAE 的逐层尺寸与参数量、结构示意图
    3. 正确性验证：形状、梯度、单 batch 过拟合、FID 合理性、插值平滑性
    4. 主实验：DCGAN 全量数据训练 6,000 步，记录样本演化与 FID 随训练变化
    5. 激活函数：生成器用 ReLU / LeakyReLU / ELU / GELU，另测判别器用 ReLU
    6. Dropout：在小数据（5,000 张）上对比判别器 Dropout 0 / 0.2 / 0.5
    7. 数据量：1,000 / 5,000 / 14,029 张，固定优化步数，比较 FID 与记忆化程度
    8. 超参数：潜维、学习率、批大小、Adam β1、单边标签平滑
    9. VAE 对照：β=1 与 β=4 的 VAE，重建/采样/插值，与 DCGAN 比较
   10. 最终模型：按扫描结果选配方、重新训练、生成画廊/插值/C2ST 与 VAE 对比

运行方式：
    cd exp6_report && python full_experiment.py                  # 全部实验（约 1.5 小时）
    cd exp6_report && python full_experiment.py --parts 4        # 只跑第 4 部分
    cd exp6_report && python full_experiment.py --parts 5 6 7    # 跑多个部分
"""

from __future__ import annotations

import argparse
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

import metrics
import viz
from data import (IMAGE_SIZE, REF_SIZE, dataset_stats, load_clean_images,
                  split_indices, subset_indices, to_gan_range)
from engine import (GANConfig, VAEConfig, environment_report, get_device,
                    interpolate_gan, interpolate_vae, reconstruct_vae,
                    sample_gan, set_seed, train_dcgan, train_vae)
from models import Discriminator, Generator, VAE, count_parameters, model_summary

FIG = HERE / "figures"
RUN_LOGS = HERE / "run_logs"
CKPT = HERE / "checkpoints"
RESULTS = HERE / "results.json"
FID_REF = HERE / "fid_reference.npz"
VERIFICATION = HERE / "verification.json"

# 对比实验统一使用的 FID 采样数（最终模型用 5,000）
FID_N_SWEEP = 4000
FID_N_FINAL = 5000
FID_FAKE_SEED = 777
SAMPLE_SEED = 2024

PROTOCOL = {
    "dataset": "Anime Face Dataset（Kaggle，64×64 RGB 动漫头像，CC0）",
    "raw_images": 21551,
    "unique_images": 17029,
    "ref_size": REF_SIZE,
    "train_size": 14029,
    "n_generator_params": None,
    "n_discriminator_params": None,
    "n_vae_params": None,
    "optimizer": "Adam(β1, β2)，G/D 初始学习率 2e-4，β=(0.5, 0.999)",
    "batch_size": 128,
    "gan_loss": "BCEWithLogitsLoss（D 输出 logit）",
    "fid_protocol": f"Inception-v3 池化特征；真实参考集 {REF_SIZE} 张（不参与训练），"
                    f"对比实验生成 {FID_N_SWEEP} 张、最终模型生成 {FID_N_FINAL} 张，"
                    f"固定采样种子 {FID_FAKE_SEED}",
    "steps_control": "对比实验固定优化步数（而非 epoch 数），保证不同数据量/配置下梯度更新次数一致",
    "seeds": "每个配置使用固定随机种子 0；最终模型与 C2ST 使用独立种子",
    "timing": "全部训练在 RTX 4050 Laptop GPU（6 GB）上完成，FP32",
}


# ---------------------------------------------------------------------------
# 结果读写
# ---------------------------------------------------------------------------


def load_results() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text(encoding="utf-8"))
    return {}


def save_results(R: dict) -> None:
    RESULTS.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")


class Ctx:
    """一次运行共享的上下文：设备、数据、FID 评估器（懒加载）。"""

    def __init__(self):
        self.device = get_device()
        self.images, self.names, self.dedup = load_clean_images()
        self.train_idx, self.ref_idx = split_indices(len(self.images))
        viz.set_fig_dir(FIG)
        CKPT.mkdir(parents=True, exist_ok=True)
        RUN_LOGS.mkdir(parents=True, exist_ok=True)
        self._fid = None
        self._real_stats = None
        print(f"[ctx] device={self.device}  图片 {tuple(self.images.shape)}  "
              f"训练 {len(self.train_idx)} / 参考 {len(self.ref_idx)}")

    @property
    def fid(self) -> metrics.FIDEvaluator:
        if self._fid is None:
            print("[ctx] 加载 Inception-v3 计算 FID ...")
            self._fid = metrics.FIDEvaluator(self.device)
        return self._fid

    @property
    def real_stats(self) -> dict:
        if self._real_stats is None:
            st = metrics.reference_stats(self.fid, self.images, self.ref_idx, FID_REF)
            self._real_stats = st
            print(f"[ctx] FID 实图参考统计：n={st['n']}")
        return self._real_stats

    def gan_fid(self, G, n: int = FID_N_SWEEP, seed: int = FID_FAKE_SEED) -> float:
        fake = sample_gan(G, n, self.device, seed=seed)
        return self.fid.score(self.real_stats, fake)

    def vae_fid(self, V, n: int = FID_N_SWEEP, seed: int = FID_FAKE_SEED) -> float:
        fake = V.sample(n, self.device, seed=seed)
        return self.fid.score(self.real_stats, fake)


@torch.no_grad()
def eval_discriminator(D, real: torch.Tensor, fake: torch.Tensor) -> dict:
    """在 eval 模式下评估判别器：真实图判对率、生成图判对率（都基于 logits>0）。"""
    was_training = D.training
    D.eval()
    real = real.to(next(D.parameters()).device)
    fake = fake.to(real.device)
    acc_real = (D(real) > 0).float().mean().item()
    fake_as_real = (D(fake) > 0).float().mean().item()
    D.train(was_training)
    return {
        "acc_real": round(acc_real, 4),
        "acc_fake": round(1.0 - fake_as_real, 4),
    }


def holdout_real(ctx: Ctx, n: int = 500) -> torch.Tensor:
    """取参考集中 n 张真实图（[0,1]）用于评估判别器的泛化。"""
    return ctx.images[torch.as_tensor(ctx.ref_idx[:n])].float().div_(255.0)


# ---------------------------------------------------------------------------
# 1. 数据概览
# ---------------------------------------------------------------------------


def part1_data(ctx: Ctx, R: dict) -> None:
    print("\n=== 第 1 部分：数据概览 ===")
    stats = dataset_stats(ctx.images, ctx.train_idx)
    sample_idx = ctx.train_idx[:64]
    samples = ctx.images[torch.as_tensor(sample_idx)].float().div_(255.0)
    viz.grid_figure(samples, "fig01_dataset_samples.png", nrow=8, scale=2,
                    title="Anime Face Dataset 训练集样本（64 张）")
    viz.plot_data_stats(ctx.images, ctx.train_idx, stats, "fig02_data_stats.png")

    info = {
        "name": "Anime Face Dataset",
        "source": "https://www.kaggle.com/soumikrakshit/anime-faces（HuggingFace 镜像 huggan/anime-faces）",
        "license": "CC0-1.0",
        "image_size": IMAGE_SIZE,
        "n_raw": ctx.dedup["n_raw"],
        "n_unique": ctx.dedup["n_unique"],
        "n_duplicate": ctx.dedup["n_duplicate"],
        "duplicate_ratio": ctx.dedup["duplicate_ratio"],
        "n_train": int(len(ctx.train_idx)),
        "n_ref": int(len(ctx.ref_idx)),
        "stats": stats,
        "fig_samples": "fig01_dataset_samples.png",
        "fig_stats": "fig02_data_stats.png",
    }
    R["data"] = info
    print(f"  原始 {info['n_raw']} 张，去重后 {info['n_unique']} 张"
          f"（重复 {info['n_duplicate']} 张，{info['duplicate_ratio']:.1%}）")
    print(f"  像素均值 RGB={stats['mean_rgb']}  标准差={stats['std_rgb']}  "
          f"平均亮度={stats['mean_brightness']}  平均饱和度={stats['mean_saturation']}")
    save_results(R)


# ---------------------------------------------------------------------------
# 2. 模型结构
# ---------------------------------------------------------------------------


def _simplify_summary(records: list[dict]) -> list[dict]:
    """只保留有输出尺寸的层，并去掉连续的激活函数记录（报告中表格更紧凑）。"""
    out = []
    for r in records:
        if r["out_shape"] is None:
            continue
        out.append({"name": r["name"], "type": r["type"],
                    "out_shape": r["out_shape"], "params": r["params"]})
    return out


def part2_architecture(ctx: Ctx, R: dict) -> None:
    print("\n=== 第 2 部分：模型结构 ===")
    G = Generator()
    D = Discriminator()
    V = VAE()
    g_layers = _simplify_summary(model_summary(G, (100,)))
    d_layers = _simplify_summary(model_summary(D, (3, IMAGE_SIZE, IMAGE_SIZE)))
    v_layers = _simplify_summary(model_summary(V, (3, IMAGE_SIZE, IMAGE_SIZE)))
    viz.plot_arch_diagram(g_layers, d_layers, "fig03_dcgan_arch.png")

    # VAE 的编码器/解码器分别取名字以 enc. / dec. 开头的层画结构图
    enc = [r for r in v_layers if r["name"].startswith("enc.")]
    dec = [r for r in v_layers if r["name"].startswith("dec.")]
    viz.plot_vae_arch(enc, dec, "fig04_vae_arch.png")

    info = {
        "generator": {"params": count_parameters(G), "layers": g_layers},
        "discriminator": {"params": count_parameters(D), "layers": d_layers},
        "vae": {"params": count_parameters(V), "layers": v_layers},
        "fig_dcgan": "fig03_dcgan_arch.png",
        "fig_vae": "fig04_vae_arch.png",
    }
    R["architecture"] = info
    PROTOCOL["n_generator_params"] = info["generator"]["params"]
    PROTOCOL["n_discriminator_params"] = info["discriminator"]["params"]
    PROTOCOL["n_vae_params"] = info["vae"]["params"]
    print(f"  G={info['generator']['params']:,} 参数  "
          f"D={info['discriminator']['params']:,} 参数  "
          f"VAE={info['vae']['params']:,} 参数")
    save_results(R)


# ---------------------------------------------------------------------------
# 3. 正确性验证
# ---------------------------------------------------------------------------


def part3_verification(ctx: Ctx, R: dict) -> None:
    print("\n=== 第 3 部分：正确性验证 ===")
    import verify

    result = verify.run_verification(ctx.device)
    result["device"] = str(ctx.device)
    VERIFICATION.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    R["verification"] = result
    print(f"  通过 {result['n_passed']}/{result['n_total']} 项检查，"
          f"耗时 {result['runtime_s']}s")
    save_results(R)


# ---------------------------------------------------------------------------
# 4. 主实验：DCGAN 全量训练
# ---------------------------------------------------------------------------

MAIN_STEPS = 6000
MAIN_SNAP_STEPS = (500, 1000, 2000, 3000, 4000, 6000)
MAIN_FID_STEPS = (1000, 2000, 3000, 4000, 6000)


def train_one_gan(ctx: Ctx, cfg: GANConfig, train_idx: np.ndarray,
                  with_snapshots: bool = False, fid_steps: tuple[int, ...] = (),
                  tag: str = "", progress: bool = True):
    """训练一个新的 DCGAN，返回 (G, D, history, snapshots, fid_points, log_lines)。"""
    set_seed(cfg.seed)
    G = Generator(z_dim=cfg.z_dim, base=cfg.g_base, activation=cfg.g_act).to(ctx.device)
    D = Discriminator(base=cfg.d_base, activation=cfg.d_act,
                      dropout=cfg.d_dropout).to(ctx.device)
    snapshots: dict[int, torch.Tensor] = {}
    fid_points: list[tuple[int, float]] = []
    log_lines: list[str] = []

    snapshot_cb_arg = None
    score_cb_arg = None
    if with_snapshots:
        def snapshot_cb(step, model):
            snapshots[step] = sample_gan(model, 64, ctx.device, seed=SAMPLE_SEED)

        snapshot_cb_arg = snapshot_cb

        def score_cb(step, model):
            f = ctx.gan_fid(model)
            fid_points.append((step, f))
            line = f"[{tag}] step {step:5d}  FID {f:7.2f}"
            print("  " + line)
            log_lines.append(line)

        score_cb_arg = score_cb

    cfg_snapshot_steps = tuple(sorted(
        (set(MAIN_SNAP_STEPS) if with_snapshots else set()) | set(fid_steps)))

    t0 = time.time()
    history = train_dcgan(G, D, ctx.images, train_idx, cfg, ctx.device,
                          snapshot_steps=cfg_snapshot_steps,
                          snapshot_cb=snapshot_cb_arg,
                          score_cb=score_cb_arg,
                          progress=progress)
    elapsed = time.time() - t0
    log_lines.append(f"[{tag}] 配置 {json.dumps(cfg.to_json(), ensure_ascii=False)}")
    log_lines.append(f"[{tag}] {cfg.max_steps} 步训练完成，耗时 {elapsed:.1f}s "
                     f"（{history['steps_per_s']:.2f} 步/s，约 {history['epochs_equivalent']:.1f} 个 epoch）")
    return G, D, history, snapshots, fid_points, log_lines


def part4_main(ctx: Ctx, R: dict) -> None:
    print("\n=== 第 4 部分：主实验（DCGAN 全量 14,029 张）===")
    cfg = GANConfig(max_steps=MAIN_STEPS, batch_size=128, z_dim=100,
                    g_act="relu", d_act="leaky_relu", d_dropout=0.0,
                    lr_g=2e-4, lr_d=2e-4, betas=(0.5, 0.999), seed=0)
    print(f"  配置：{cfg.to_json()}")
    G, D, history, snapshots, fid_points, log_lines = train_one_gan(
        ctx, cfg, ctx.train_idx, with_snapshots=True, fid_steps=MAIN_FID_STEPS, tag="main")

    final_fid = ctx.gan_fid(G, n=FID_N_SWEEP)
    if not fid_points or fid_points[-1][0] != cfg.max_steps:
        fid_points.append((cfg.max_steps, final_fid))
    samples = sample_gan(G, 64, ctx.device, seed=SAMPLE_SEED)
    viz.grid_figure(samples, "fig07_main_samples.png", nrow=8, scale=2,
                    title=f"主实验 DCGAN 生成样本（训练 {cfg.max_steps} 步，FID={final_fid:.1f}）")
    viz.evolution_figure(snapshots, "fig06_main_evolution.png", n_cols=8,
                         title="训练过程中固定噪声样本的演化")
    viz.plot_gan_training(history, "fig05_main_training.png",
                          title="主实验 DCGAN 训练曲线",
                          fid_points=fid_points)

    torch.save({"g": G.state_dict(), "d": D.state_dict(), "cfg": cfg.to_json()},
               CKPT / "main_G.pt")
    color = metrics.color_diversity(samples, n=64)
    R["main"] = {
        "config": cfg.to_json(),
        "history": history,
        "snapshots": sorted(snapshots),
        "fid_points": [[int(s), float(f)] for s, f in fid_points],
        "final_fid": final_fid,
        "train_time_s": history["train_time_s"],
        "steps_per_s": history["steps_per_s"],
        "epochs_equivalent": history["epochs_equivalent"],
        "sample_figs": {
            "samples": "fig07_main_samples.png",
            "evolution": "fig06_main_evolution.png",
            "training": "fig05_main_training.png",
        },
        "color_diversity": color,
        "log_lines": log_lines,
        "checkpoint": "checkpoints/main_G.pt",
    }
    print(f"  最终 FID = {final_fid:.2f}，耗时 {history['train_time_s']:.0f}s")
    save_results(R)


# ---------------------------------------------------------------------------
# 5. 激活函数
# ---------------------------------------------------------------------------

ACT_SWEEP = [
    ("relu", "relu", "leaky_relu", "ReLU（DCGAN 默认）"),
    ("leaky_relu", "leaky_relu", "leaky_relu", "LeakyReLU(0.2)"),
    ("elu", "elu", "leaky_relu", "ELU"),
    ("gelu", "gelu", "leaky_relu", "GELU"),
]
SWEEP_STEPS = 1500


def part5_activation(ctx: Ctx, R: dict) -> None:
    print("\n=== 第 5 部分：激活函数对比（1,500 步）===")
    runs = []
    sample_rows = {}
    histories = {}
    for key, g_act, d_act, label in ACT_SWEEP:
        print(f"  -- 生成器激活函数：{label}")
        cfg = GANConfig(max_steps=SWEEP_STEPS, batch_size=128, z_dim=100,
                        g_act=g_act, d_act=d_act, d_dropout=0.0,
                        lr_g=2e-4, lr_d=2e-4, betas=(0.5, 0.999), seed=0)
        G, D, history, _, _, log_lines = train_one_gan(ctx, cfg, ctx.train_idx, tag=f"act_{key}")
        fid = ctx.gan_fid(G)
        samples = sample_gan(G, 64, ctx.device, seed=SAMPLE_SEED)
        sample_rows[label] = samples[:8]
        histories[label] = history
        runs.append({
            "key": key, "label": label, "g_act": g_act, "d_act": d_act,
            "fid": fid, "history": history, "log_lines": log_lines,
            "color_diversity": metrics.color_diversity(samples, n=64),
        })
        print(f"     {label}: FID = {fid:.2f}")

    # 额外一组：判别器用 ReLU（G 保持 ReLU）
    print("  -- 判别器激活函数：ReLU（对照组）")
    cfg = GANConfig(max_steps=SWEEP_STEPS, batch_size=128, z_dim=100,
                    g_act="relu", d_act="relu", d_dropout=0.0,
                    lr_g=2e-4, lr_d=2e-4, betas=(0.5, 0.999), seed=0)
    G, D, history, _, _, log_lines = train_one_gan(ctx, cfg, ctx.train_idx, tag="act_d_relu")
    fid = ctx.gan_fid(G)
    samples = sample_gan(G, 64, ctx.device, seed=SAMPLE_SEED)
    sample_rows["D 用 ReLU"] = samples[:8]
    histories["D 用 ReLU"] = history
    d_relu = {"key": "d_relu", "label": "D 用 ReLU", "g_act": "relu", "d_act": "relu",
              "fid": fid, "history": history, "log_lines": log_lines,
              "color_diversity": metrics.color_diversity(samples, n=64)}
    print(f"     D 用 ReLU: FID = {fid:.2f}")

    viz.row_grid_figure({r["label"]: sample_rows[r["label"]] for r in runs},
                        "fig08_activation_samples.png", n_cols=8, scale=2,
                        title="不同生成器激活函数的生成样本（同为 1,500 步）")
    viz.plot_fid_bars([r["label"] for r in runs] + ["D 用 ReLU"],
                      [r["fid"] for r in runs] + [d_relu["fid"]],
                      "fig09_activation_fid.png", "激活函数对 FID 的影响（1,500 步）",
                      colors=viz.PALETTE[:4] + [viz.PALETTE[7]])
    viz.plot_config_curves(histories, "fig10_activation_curves.png",
                           "不同激活函数的训练曲线", metric="loss_d", metric_label="判别器损失")

    R["activation"] = {
        "steps": SWEEP_STEPS,
        "runs": runs,
        "d_relu": d_relu,
        "figs": {"samples": "fig08_activation_samples.png",
                 "fid_bars": "fig09_activation_fid.png",
                 "curves": "fig10_activation_curves.png"},
    }
    save_results(R)


# ---------------------------------------------------------------------------
# 6. Dropout
# ---------------------------------------------------------------------------

DROPOUT_RATES = (0.0, 0.2, 0.5)
DROPOUT_STEPS = 2000
DROPOUT_DATA = 5000


def part6_dropout(ctx: Ctx, R: dict) -> None:
    print(f"\n=== 第 6 部分：Dropout（{DROPOUT_DATA} 张训练图，2,000 步）===")
    sub_idx = subset_indices(ctx.train_idx, DROPOUT_DATA, seed=3)
    real_hold = holdout_real(ctx, 500)                       # [0,1] CPU
    real_train_batch = ctx.images[torch.as_tensor(sub_idx[:500])].float().div_(255.0)

    runs = []
    sample_rows = {}
    histories = {}
    for rate in DROPOUT_RATES:
        print(f"  -- 判别器 Dropout = {rate}")
        cfg = GANConfig(max_steps=DROPOUT_STEPS, batch_size=128, z_dim=100,
                        g_act="relu", d_act="leaky_relu", d_dropout=rate,
                        lr_g=2e-4, lr_d=2e-4, betas=(0.5, 0.999), seed=0)
        G, D, history, _, _, log_lines = train_one_gan(ctx, cfg, sub_idx, tag=f"drop{rate}")
        fid = ctx.gan_fid(G)
        samples = sample_gan(G, 64, ctx.device, seed=SAMPLE_SEED)
        sample_rows[f"Dropout {rate}"] = samples[:8]
        histories[f"Dropout {rate}"] = history
        fake = sample_gan(G, 500, ctx.device, seed=999)
        eval_hold = eval_discriminator(D, to_gan_range(real_hold), to_gan_range(fake))
        eval_train = eval_discriminator(D, to_gan_range(real_train_batch), to_gan_range(fake))
        runs.append({
            "dropout": rate,
            "label": f"Dropout {rate}",
            "fid": fid,
            "d_holdout": eval_hold,
            "d_train": eval_train,
            "history": history,
            "log_lines": log_lines,
            "color_diversity": metrics.color_diversity(samples, n=64),
        })
        print(f"     FID={fid:.2f}  D 留出真实图判对率={eval_hold['acc_real']:.3f}  "
              f"D 训练真实图判对率={eval_train['acc_real']:.3f}")

    viz.plot_config_curves(histories, "fig11_dropout_curves.png",
                           "不同判别器 Dropout 的训练曲线",
                           metric="d_fake", metric_label="D(生成图) 输出概率")
    viz.plot_fid_bars([r["label"] for r in runs], [r["fid"] for r in runs],
                      "fig12_dropout_fid.png", "判别器 Dropout 对 FID 的影响")
    # Dropout 对判别器泛化的影响：留出真实图判对率 vs 训练真实图判对率
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xs = np.arange(len(runs))
    train_acc = [r["d_train"]["acc_real"] for r in runs]
    hold_acc = [r["d_holdout"]["acc_real"] for r in runs]
    ax.bar(xs - 0.18, train_acc, width=0.36, color=viz.PALETTE[0], label="训练真实图判对率")
    ax.bar(xs + 0.18, hold_acc, width=0.36, color=viz.PALETTE[1], label="留出真实图判对率")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in runs])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("判别器真实图判对率")
    ax.set_title("Dropout 对判别器泛化的影响（训练 vs 留出）")
    ax.legend(fontsize=8)
    for i in range(len(runs)):
        gap = train_acc[i] - hold_acc[i]
        ax.text(xs[i], max(train_acc[i], hold_acc[i]) + 0.02, f"差距 {gap:+.3f}",
                ha="center", fontsize=8)
    viz.savefig(fig, "fig13_dropout_gap.png")
    viz.row_grid_figure(sample_rows, "fig14_dropout_samples.png", n_cols=8, scale=2,
                        title="不同 Dropout 的生成样本")

    R["dropout"] = {
        "steps": DROPOUT_STEPS,
        "n_train_images": DROPOUT_DATA,
        "runs": runs,
        "figs": {"curves": "fig11_dropout_curves.png",
                 "fid": "fig12_dropout_fid.png",
                 "gap": "fig13_dropout_gap.png",
                 "samples": "fig14_dropout_samples.png"},
    }
    save_results(R)


# ---------------------------------------------------------------------------
# 7. 数据量
# ---------------------------------------------------------------------------

DATA_SIZES = (1000, 5000, 14029)
DATA_SIZE_STEPS = 3000


def part7_data_size(ctx: Ctx, R: dict) -> None:
    print(f"\n=== 第 7 部分：数据量（{DATA_SIZES}，各 3,000 步）===")
    train_full = ctx.train_idx
    gen_pool = ctx.images[torch.as_tensor(train_full)].float().div_(255.0)
    real_hold = holdout_real(ctx, 500)

    runs = []
    sample_rows = {}
    nn_gen: dict[int, dict] = {}
    for size in DATA_SIZES:
        sub = subset_indices(train_full, size, seed=11)
        print(f"  -- 训练图片数 {size}（约 {DATA_SIZE_STEPS * 128 / size:.0f} 个 epoch）")
        cfg = GANConfig(max_steps=DATA_SIZE_STEPS, batch_size=128, z_dim=100,
                        g_act="relu", d_act="leaky_relu", d_dropout=0.0,
                        lr_g=2e-4, lr_d=2e-4, betas=(0.5, 0.999), seed=0)
        G, D, history, _, _, log_lines = train_one_gan(ctx, cfg, sub, tag=f"size{size}")
        fid = ctx.gan_fid(G)
        samples = sample_gan(G, 500, ctx.device, seed=555)
        own_pool = ctx.images[torch.as_tensor(sub)].float().div_(255.0)
        d_gen = metrics.min_distances(samples, own_pool, ctx.device)
        d_real = metrics.min_distances(real_hold, own_pool, ctx.device)
        stats_gen = metrics.nn_summary(d_gen)
        stats_real = metrics.nn_summary(d_real)
        nn_gen[size] = stats_gen
        sample_rows[f"{size} 张"] = sample_gan(G, 64, ctx.device, seed=SAMPLE_SEED)[:8]
        runs.append({
            "size": size,
            "fid": fid,
            "nn_gen": stats_gen,
            "nn_real": stats_real,
            "nn_ratio": round(stats_gen["mean"] / stats_real["mean"], 4),
            "epochs_equivalent": history["epochs_equivalent"],
            "history": history,
            "log_lines": log_lines,
            "color_diversity": metrics.color_diversity(
                sample_gan(G, 64, ctx.device, seed=SAMPLE_SEED), n=64),
        })
        print(f"     FID={fid:.2f}  生成图最近邻均值={stats_gen['mean']:.2f}  "
              f"真实留出图最近邻均值={stats_real['mean']:.2f}  比值={runs[-1]['nn_ratio']:.3f}")

    viz.plot_data_size_results(list(DATA_SIZES), [r["fid"] for r in runs],
                               [r["nn_real"] for r in runs], nn_gen, "fig15_data_size.png")
    viz.row_grid_figure(sample_rows, "fig16_data_size_samples.png", n_cols=8, scale=2,
                        title="不同训练数据量的生成样本（各 3,000 步）")

    R["data_size"] = {
        "sizes": list(DATA_SIZES),
        "steps": DATA_SIZE_STEPS,
        "runs": runs,
        "figs": {"summary": "fig15_data_size.png", "samples": "fig16_data_size_samples.png"},
    }
    save_results(R)


# ---------------------------------------------------------------------------
# 8. 超参数
# ---------------------------------------------------------------------------


def _hyper_cfg(**kwargs) -> GANConfig:
    base = dict(max_steps=SWEEP_STEPS, batch_size=128, z_dim=100, g_act="relu",
                d_act="leaky_relu", d_dropout=0.0, lr_g=2e-4, lr_d=2e-4,
                betas=(0.5, 0.999), seed=0)
    base.update(kwargs)
    return GANConfig(**base)


def _run_hyper(ctx: Ctx, cfg: GANConfig, tag: str) -> dict:
    G, D, history, _, _, log_lines = train_one_gan(ctx, cfg, ctx.train_idx, tag=tag)
    fid = ctx.gan_fid(G)
    samples = sample_gan(G, 64, ctx.device, seed=SAMPLE_SEED)
    print(f"     FID={fid:.2f}")
    return {
        "cfg": cfg.to_json(), "fid": fid, "history": history,
        "log_lines": log_lines,
        "color_diversity": metrics.color_diversity(samples, n=64),
    }


def part8_hyperparams(ctx: Ctx, R: dict) -> None:
    print(f"\n=== 第 8 部分：超参数（各 {SWEEP_STEPS} 步）===")
    # 参考配置：来自激活函数实验的 ReLU 基线（同 1,500 步），直接复用其 FID
    base_run = None
    for r in R.get("activation", {}).get("runs", []):
        if r["key"] == "relu":
            base_run = r
            break
    base_fid = base_run["fid"] if base_run else None
    print(f"  参考配置 FID（复用激活实验 ReLU 组）= {base_fid}")

    hyper = {"steps": SWEEP_STEPS, "base_fid": base_fid, "sweeps": {}}

    # 潜维
    latent_runs = []
    for z in (32, 100, 256):
        if z == 100 and base_run is not None:
            latent_runs.append({"z_dim": z, "label": "潜维 100（参考）", "fid": base_fid,
                                "history": base_run["history"], "cfg": "继承激活实验 ReLU 组"})
            continue
        print(f"  -- 潜维 z_dim={z}")
        run = _run_hyper(ctx, _hyper_cfg(z_dim=z), tag=f"z{z}")
        run.update({"z_dim": z, "label": f"潜维 {z}"})
        latent_runs.append(run)
    hyper["sweeps"]["latent"] = latent_runs

    # 学习率
    lr_runs = []
    for lr in (1e-4, 2e-4, 5e-4):
        if lr == 2e-4 and base_run is not None:
            lr_runs.append({"lr": lr, "label": "lr 2e-4（参考）", "fid": base_fid,
                            "history": base_run["history"], "cfg": "继承激活实验 ReLU 组"})
            continue
        print(f"  -- 学习率 {lr:g}")
        run = _run_hyper(ctx, _hyper_cfg(lr_g=lr, lr_d=lr), tag=f"lr{lr:g}")
        run.update({"lr": lr, "label": f"lr {lr:g}"})
        lr_runs.append(run)
    hyper["sweeps"]["lr"] = lr_runs

    # 批大小
    batch_runs = []
    for bs in (64, 128, 256):
        if bs == 128 and base_run is not None:
            batch_runs.append({"batch_size": bs, "label": "batch 128（参考）", "fid": base_fid,
                               "history": base_run["history"], "cfg": "继承激活实验 ReLU 组"})
            continue
        print(f"  -- 批大小 {bs}")
        run = _run_hyper(ctx, _hyper_cfg(batch_size=bs), tag=f"bs{bs}")
        run.update({"batch_size": bs, "label": f"batch {bs}"})
        batch_runs.append(run)
    hyper["sweeps"]["batch"] = batch_runs

    # Adam β1 与单边标签平滑
    other_runs = []
    if base_run is not None:
        other_runs.append({"label": "β1=0.5（参考）", "fid": base_fid,
                           "history": base_run["history"], "cfg": "继承激活实验 ReLU 组"})
    print("  -- Adam β1=0.9")
    run = _run_hyper(ctx, _hyper_cfg(betas=(0.9, 0.999)), tag="beta09")
    run.update({"label": "β1=0.9"})
    other_runs.append(run)
    print("  -- 单边标签平滑 0.1")
    run = _run_hyper(ctx, _hyper_cfg(real_smooth=0.1), tag="smooth01")
    run.update({"label": "标签平滑 0.1"})
    other_runs.append(run)
    hyper["sweeps"]["other"] = other_runs

    # 四联图汇总
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    panels = [("latent", "潜维 z_dim"), ("lr", "学习率"), ("batch", "批大小"), ("other", "Adam β1 / 标签平滑")]
    for ax, (key, title) in zip(axes.flat, panels):
        runs = hyper["sweeps"][key]
        labels = [r["label"] for r in runs]
        vals = [r["fid"] for r in runs]
        colors = [viz.PALETTE[0] if i != (len(runs) - 1) else viz.PALETTE[3]
                  for i in range(len(runs))]
        bars = ax.bar(labels, vals, color=colors, alpha=0.88)
        ax.set_title(title)
        ax.set_ylabel("FID（越低越好）")
        if vals:
            ax.set_ylim(0, max(vals) * 1.18)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.suptitle(f"超参数对 FID 的影响（各 {SWEEP_STEPS} 步）", fontsize=12)
    fig.tight_layout()
    viz.savefig(fig, "fig17_hyper_summary.png")

    # 学习率对训练稳定性的影响单独画曲线
    lr_hist = {r["label"]: r["history"] for r in hyper["sweeps"]["lr"]}
    viz.plot_config_curves(lr_hist, "fig18_hyper_lr_curves.png",
                           "学习率对训练曲线的影响", metric="loss_g",
                           metric_label="生成器损失")
    R["hyper"] = hyper
    save_results(R)


# ---------------------------------------------------------------------------
# 9. VAE 对照
# ---------------------------------------------------------------------------

VAE_STEPS = 4000


def part9_vae(ctx: Ctx, R: dict) -> None:
    print(f"\n=== 第 9 部分：VAE 对照（{VAE_STEPS} 步）===")
    runs = []
    histories = {}
    sample_rows = {}
    for beta, label in ((1.0, "VAE β=1"), (4.0, "β-VAE β=4")):
        print(f"  -- {label}")
        cfg = VAEConfig(max_steps=VAE_STEPS, batch_size=128, z_dim=128, base=64,
                        beta=beta, lr=1e-3, seed=0)
        set_seed(cfg.seed)
        V = VAE(z_dim=cfg.z_dim, base=cfg.base, beta=cfg.beta).to(ctx.device)
        history = train_vae(V, ctx.images, ctx.train_idx, cfg, ctx.device)
        fid = ctx.vae_fid(V)
        real64, recon64 = reconstruct_vae(V, ctx.images, ctx.ref_idx, ctx.device, n=8, seed=1)
        recon_metrics = metrics.reconstruction_metrics(real64, recon64)
        samples = V.sample(64, ctx.device, seed=SAMPLE_SEED)
        interp = interpolate_vae(V, ctx.device, n_steps=10, seed=0)
        histories[label] = history
        sample_rows[label] = samples[:8]
        runs.append({
            "label": label,
            "beta": beta,
            "z_dim": cfg.z_dim,
            "fid": fid,
            "history": history,
            "recon": recon_metrics,
            "color_diversity": metrics.color_diversity(samples, n=64),
            "epochs_equivalent": history["epochs_equivalent"],
        })
        torch.save({"vae": V.state_dict(), "cfg": cfg.to_json()}, CKPT / f"vae_beta{beta:g}.pt")
        # 每个 beta 的重建/插值图
        viz.plot_reconstruction(real64, recon64, f"fig20_vae_recon_beta{beta:g}.png",
                                title=f"{label} 重建（MSE={recon_metrics['mse']:.4f}, "
                                      f"PSNR={recon_metrics['psnr']:.2f} dB）")
        viz.plot_interpolation(interp, f"fig21_vae_interp_beta{beta:g}.png",
                               title=f"{label} 潜空间插值")
        print(f"     FID={fid:.2f}  重建 MSE={recon_metrics['mse']:.4f}  "
              f"PSNR={recon_metrics['psnr']:.2f} dB")

    viz.plot_vae_training(histories["VAE β=1"], "fig19_vae_training.png",
                          title="VAE 训练曲线（β=1）")
    viz.row_grid_figure(sample_rows, "fig22_vae_samples.png", n_cols=8, scale=2,
                        title="VAE 从先验采样的样本")
    R["vae"] = {
        "steps": VAE_STEPS,
        "runs": runs,
        "figs": {"training": "fig19_vae_training.png",
                 "recon_beta1": "fig20_vae_recon_beta1.png",
                 "recon_beta4": "fig20_vae_recon_beta4.png",
                 "interp_beta1": "fig21_vae_interp_beta1.png",
                 "interp_beta4": "fig21_vae_interp_beta4.png",
                 "samples": "fig22_vae_samples.png"},
    }
    save_results(R)


# ---------------------------------------------------------------------------
# 10. 最终模型与对比
# ---------------------------------------------------------------------------


def _best_activation(R: dict) -> str:
    runs = R.get("activation", {}).get("runs", [])
    if not runs:
        return "relu"
    best = min(runs, key=lambda r: r["fid"])
    return best["g_act"]


def _best_latent(R: dict) -> int:
    runs = R.get("hyper", {}).get("sweeps", {}).get("latent", [])
    if not runs:
        return 100
    best = min(runs, key=lambda r: r["fid"])
    return best["z_dim"]


def part10_final(ctx: Ctx, R: dict) -> None:
    print("\n=== 第 10 部分：最终模型与对比 ===")
    act = _best_activation(R)
    z_dim = _best_latent(R)
    cfg = GANConfig(max_steps=MAIN_STEPS, batch_size=128, z_dim=z_dim,
                    g_act=act, d_act="leaky_relu", d_dropout=0.0,
                    lr_g=2e-4, lr_d=2e-4, betas=(0.5, 0.999), seed=0)
    same_as_main = (act == "relu" and z_dim == 100)
    print(f"  选择的配方：g_act={act}, z_dim={z_dim}；"
          f"{'与主实验一致，直接复用主实验模型' if same_as_main else '重新训练 6,000 步'}")

    final_ckpt = CKPT / "final_G.pt"
    reused_final = False
    if final_ckpt.exists():
        blob = torch.load(final_ckpt, map_location=ctx.device, weights_only=False)
        saved = blob.get("cfg", {})
        if (saved.get("g_act") == act and saved.get("z_dim") == z_dim
                and saved.get("max_steps") == MAIN_STEPS):
            G = Generator(z_dim=z_dim, base=64, activation=act).to(ctx.device)
            G.load_state_dict(blob["g"])
            history = blob.get("history", {})
            log_lines = blob.get("log_lines", [])
            reused_final = True
            print("  发现已训练的最终模型检查点，直接复用")
    if not reused_final and same_as_main and (CKPT / "main_G.pt").exists():
        blob = torch.load(CKPT / "main_G.pt", map_location=ctx.device, weights_only=False)
        G = Generator(z_dim=z_dim, base=64, activation=act).to(ctx.device)
        G.load_state_dict(blob["g"])
        history = R["main"]["history"]
        log_lines = list(R["main"].get("log_lines", []))
    elif not reused_final:
        G, D, history, snapshots, fid_points, log_lines = train_one_gan(
            ctx, cfg, ctx.train_idx, with_snapshots=True,
            fid_steps=(1000, 3000, 6000), tag="final")
        torch.save({"g": G.state_dict(), "cfg": cfg.to_json(),
                    "history": history, "log_lines": log_lines}, final_ckpt)

    final_fid = ctx.gan_fid(G, n=FID_N_FINAL, seed=FID_FAKE_SEED)
    gallery = sample_gan(G, 80, ctx.device, seed=2025)
    viz.grid_figure(gallery, "fig23_final_gallery.png", nrow=10, scale=3,
                    title=f"最终 DCGAN 生成的动漫头像（80 张，FID={final_fid:.1f}）",
                    resize="lanczos")
    interp = interpolate_gan(G, ctx.device, n_steps=10, seed=7)
    viz.plot_interpolation(interp, "fig24_final_interp.png",
                           title="最终模型潜空间线性插值（10 帧）")

    # 与 VAE 对比：FID（复用第 9 部分）+ C2ST
    vae_runs = R.get("vae", {}).get("runs", [])
    vae_best = min(vae_runs, key=lambda r: r["fid"]) if vae_runs else None
    real_2000 = holdout_real(ctx, 2000)
    dcgan_fake = sample_gan(G, 2000, ctx.device, seed=321)
    c2st_dcgan = metrics.c2st_accuracy(real_2000, dcgan_fake, ctx.device, seed=0,
                                       n_train=2000, n_test=800, epochs=4)
    C2ST = {"dcgan": c2st_dcgan}
    if vae_best is not None:
        blob = torch.load(CKPT / f"vae_beta{vae_best['beta']:g}.pt",
                          map_location=ctx.device, weights_only=False)
        V = VAE(z_dim=blob["cfg"]["z_dim"], base=blob["cfg"]["base"],
                beta=blob["cfg"]["beta"]).to(ctx.device)
        V.load_state_dict(blob["vae"])
        vae_fake = V.sample(2000, ctx.device, seed=321)
        c2st_vae = metrics.c2st_accuracy(real_2000, vae_fake, ctx.device, seed=1,
                                         n_train=2000, n_test=800, epochs=4)
        C2ST["vae"] = c2st_vae
        # DCGAN / VAE / 真实图三行对比
        rows = {"真实图": real_2000[:8],
                f"DCGAN（FID {final_fid:.1f}）": dcgan_fake[:8],
                f"{vae_best['label']}（FID {vae_best['fid']:.1f}）": vae_fake[:8]}
        viz.row_grid_figure(rows, "fig26_final_vs_vae_samples.png", n_cols=8, scale=2,
                            title="真实图 vs DCGAN vs VAE 生成样本")
        viz.plot_gan_vae_compare(final_fid, vae_best["fid"],
                                 c2st_dcgan["accuracy"], c2st_vae["accuracy"],
                                 "fig27_final_vs_vae_metrics.png")
    else:
        vae_fake = None

    # 三种随机种子采样，检查多样性/模式崩溃
    seed_rows = {f"种子 {s}": sample_gan(G, 8, ctx.device, seed=s) for s in (11, 22, 33)}
    viz.row_grid_figure(seed_rows, "fig25_final_seeds.png", n_cols=8, scale=2,
                        title="最终模型不同随机种子的采样（检查模式崩溃）")

    # 运行日志截图
    viz.render_terminal_log(log_lines + [
        f"[final] 最终 FID = {final_fid:.2f}",
        f"[final] C2ST(DCGAN) = {c2st_dcgan['accuracy']:.4f}",
    ], "fig28_terminal_log.png", title="DCGAN 训练日志（节选）")

    R["final"] = {
        "config": cfg.to_json(),
        "reused_main": same_as_main,
        "final_fid": final_fid,
        "history": history,
        "c2st": C2ST,
        "vae_best": vae_best,
        "color_diversity": metrics.color_diversity(gallery, n=80),
        "figs": {"gallery": "fig23_final_gallery.png",
                 "interp": "fig24_final_interp.png",
                 "seeds": "fig25_final_seeds.png",
                 "vs_vae_samples": "fig26_final_vs_vae_samples.png",
                 "vs_vae_metrics": "fig27_final_vs_vae_metrics.png",
                 "terminal_log": "fig28_terminal_log.png"},
        "log_lines": log_lines,
    }
    print(f"  最终模型 FID={final_fid:.2f}  "
          f"C2ST: DCGAN={c2st_dcgan['accuracy']:.4f}"
          + (f"  VAE={C2ST['vae']['accuracy']:.4f}" if "vae" in C2ST else ""))
    save_results(R)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

PARTS = {
    1: ("data", part1_data),
    2: ("architecture", part2_architecture),
    3: ("verification", part3_verification),
    4: ("main", part4_main),
    5: ("activation", part5_activation),
    6: ("dropout", part6_dropout),
    7: ("data_size", part7_data_size),
    8: ("hyper", part8_hyperparams),
    9: ("vae", part9_vae),
    10: ("final", part10_final),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="实验六：DCGAN + VAE 生成动漫头像")
    parser.add_argument("--parts", nargs="*", type=int, default=None,
                        help="要运行的实验部分（默认全部）；例如 --parts 4 5 6")
    parser.add_argument("--device", default=None, help="cuda / cpu（默认自动）")
    parser.add_argument("--fresh", action="store_true", help="清空 results.json 后重新开始")
    args = parser.parse_args()

    if args.fresh and RESULTS.exists():
        RESULTS.unlink()
    RUN_LOGS.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOGS / "run_all.log"
    log_fh = log_path.open("a", encoding="utf-8")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = Tee(sys.__stdout__, log_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh)

    t_start = time.time()
    if args.device:
        import engine
        orig = engine.get_device
        engine.get_device = lambda prefer_cuda=True: torch.device(args.device)
        globals()["get_device"] = engine.get_device
    ctx = Ctx()
    R = load_results()
    R.setdefault("meta", {})
    R["meta"].update({
        "environment": environment_report(),
        "protocol": PROTOCOL,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_file": str(Path(__file__).name),
    })
    save_results(R)

    parts = args.parts if args.parts else sorted(PARTS)
    print(f"\n>>> 计划运行部分：{parts}")
    for p in parts:
        if p not in PARTS:
            print(f"未知部分 {p}，跳过")
            continue
        name, fn = PARTS[p]
        t0 = time.time()
        fn(ctx, R)
        R["meta"].setdefault("part_times_s", {})[f"{p}_{name}"] = round(time.time() - t0, 1)
        save_results(R)
        print(f"<<< 第 {p} 部分（{name}）完成，耗时 {time.time() - t0:.0f}s")

    R["meta"]["total_time_s"] = round(time.time() - t_start, 1)
    R["meta"]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_results(R)
    print(f"\n全部完成，总耗时 {R['meta']['total_time_s'] / 60:.1f} 分钟，结果已写入 {RESULTS.name}")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_fh.close()


if __name__ == "__main__":
    main()
