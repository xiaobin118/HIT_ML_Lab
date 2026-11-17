#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩散模型评估脚本：在训练结束后挑选最佳 EMA checkpoint，并计算全套指标。

流程：
    1. 从 diffusion_run/ema/ 里挑选若干个 EMA checkpoint（约 55%/75%/90%/100% 训练进度）；
    2. 用 DDIM-50 各生成 2,048 张图，与 3,000 张真实参考图的统计量算 FID，挑出最佳 checkpoint；
    3. 对最佳 checkpoint 做完整评估：
       - FID（4,000 张，与 GAN/VAE 完全相同的真实参考集与 Inception 特征）；
       - C2ST（真实 2,000 vs 生成 2,000 的小 CNN 判别测试）；
       - 最近邻记忆化分析、颜色多样性、特征多样性；
       - 采样速度：DDIM-20/50/100 与 DDPM-1000 的耗时；
       - DDIM 步数-FID 权衡（20/50/100 步，各 1,024 张）；
    4. 结果写入 diffusion_run/eval_results.json，生成样本保存为 .pt 供报告出图。

运行：cd exp6_report && python diffusion_eval.py
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

import metrics  # noqa: E402
from data import load_clean_images, split_indices  # noqa: E402
from diffusion import GaussianDiffusion, UNet, sample_ddim, sample_ddpm  # noqa: E402

RUN = HERE / "diffusion_run"
EMA_DIR = RUN / "ema"
OUT = RUN / "eval_results.json"
FID_REF = HERE / "fid_reference.npz"
MEM_FRACTION = 0.72


def load_model(ckpt_path: Path, device: torch.device) -> tuple[UNet, dict]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob["cfg"]
    model = UNet(base=cfg["base"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(blob["ema"])
    model.eval()
    return model, {"step": int(blob["step"]), "ckpt": ckpt_path.name, **cfg}


def timed(fn):
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t0


def main():
    RUN.mkdir(exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    torch.manual_seed(0)

    # ---------- 真实参考集统计量 ----------
    ref = np.load(FID_REF)
    real_stats = {"mu": ref["mu"], "sigma": ref["sigma"], "n": int(ref["n"])}
    print(f"[eval] 真实参考集 n={real_stats['n']}", flush=True)

    images, _, _ = load_clean_images()
    train_idx, ref_idx = split_indices(len(images))
    real_ref_uint8 = images[torch.as_tensor(ref_idx)]
    train_pool = images[torch.as_tensor(train_idx)].float().div_(255.0)

    # ---------- 挑选 checkpoint ----------
    ema_files = sorted(EMA_DIR.glob("ema_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ema_files:
        raise FileNotFoundError(f"{EMA_DIR} 中没有 EMA checkpoint")
    steps = [int(p.stem.split("_")[1]) for p in ema_files]
    max_step = steps[-1]
    targets = [0.55, 0.75, 0.9, 1.0]
    selected = []
    for t in targets:
        target = max_step * t
        idx = min(range(len(steps)), key=lambda i: abs(steps[i] - target))
        if ema_files[idx] not in selected:
            selected.append(ema_files[idx])
    print(f"[eval] 所有 checkpoint: {steps}", flush=True)
    print(f"[eval] 选中的 checkpoint: {[p.name for p in selected]}", flush=True)

    fid_eval = metrics.FIDEvaluator(device)
    screening = []
    for path in selected:
        model, cfg = load_model(path, device)
        t0 = time.time()
        imgs = sample_ddim(model, GaussianDiffusion(cfg["T"], "cosine", device), 2048,
                           device, steps=50, seed=777, batch=128)
        dt = time.time() - t0
        feats = fid_eval.features(imgs)
        fake_stats = {"mu": feats.mean(0), "sigma": np.cov(feats, rowvar=False)}
        fid = metrics.fid_from_stats(real_stats, fake_stats)
        screening.append({"step": cfg["step"], "ckpt": path.name, "fid": fid,
                          "sample_seconds": round(dt, 1)})
        print(f"  [screen] {path.name}  step={cfg['step']}  FID={fid:.2f}  "
              f"采样 {dt:.1f}s", flush=True)
        del model, imgs, feats
        torch.cuda.empty_cache()

    best_ckpt = min(screening, key=lambda r: r["fid"])
    print(f"[eval] 最佳 checkpoint: {best_ckpt['ckpt']} (step={best_ckpt['step']}, "
          f"FID={best_ckpt['fid']:.2f})", flush=True)

    # ---------- 最佳 checkpoint 完整评估 ----------
    model, cfg = load_model(EMA_DIR / best_ckpt["ckpt"], device)
    diffusion = GaussianDiffusion(cfg["T"], "cosine", device)

    # 1) FID：4000 张（与 GAN/VAE 的对比实验一致）
    t0 = time.time()
    fake_4000 = sample_ddim(model, diffusion, 4000, device, steps=50, seed=2024, batch=128)
    ddim50_seconds = time.time() - t0
    feats_4000 = fid_eval.features(fake_4000)
    fake_stats = {"mu": feats_4000.mean(0), "sigma": np.cov(feats_4000, rowvar=False)}
    fid_final = metrics.fid_from_stats(real_stats, fake_stats)
    print(f"[eval] 最终 FID(4000, DDIM-50) = {fid_final:.2f}  采样 {ddim50_seconds:.0f}s", flush=True)

    # 2) C2ST：真实 2000 vs 生成 2000
    real_2000 = real_ref_uint8[:2000].float().div_(255.0)
    c2st = metrics.c2st_accuracy(real_2000, fake_4000[:2000], device, seed=0,
                                 n_train=2000, n_test=800, epochs=4)
    print(f"[eval] C2ST = {c2st}", flush=True)

    # 3) 最近邻（记忆化）与多样性
    gen_500 = fake_4000[:500]
    nn = metrics.min_distances(gen_500, train_pool, device)
    real_500 = images[torch.as_tensor(ref_idx[:500])].float().div_(255.0)
    nn_real = metrics.min_distances(real_500, train_pool, device)
    color = metrics.color_diversity(fake_4000[:500], n=500)
    color_real = metrics.color_diversity(real_500, n=500)
    feat_div = metrics.feature_diversity(feats_4000[:500], n=500)
    real_feats = fid_eval.features(real_500)
    feat_div_real = metrics.feature_diversity(real_feats, n=500)
    print(f"[eval] NN 生成 {metrics.nn_summary(nn)}  真实 {metrics.nn_summary(nn_real)}", flush=True)

    # 4) 采样速度
    speed = {}
    for steps in (20, 50, 100):
        _, dt = timed(lambda s=steps: sample_ddim(model, diffusion, 64, device, steps=s,
                                                 seed=1, batch=64))
        speed[f"ddim{steps}"] = round(dt, 2)
    _, dt = timed(lambda: sample_ddpm(model, diffusion, 16, device, seed=1, batch=16))
    speed["ddpm1000"] = round(dt, 2)
    print(f"[eval] 采样耗时: {speed}", flush=True)

    # 5) DDIM 步数 - FID 权衡（各 1024 张）
    ddim_sweep = []
    for steps in (20, 50, 100):
        imgs = sample_ddim(model, diffusion, 1024, device, steps=steps, seed=777, batch=128)
        feats = fid_eval.features(imgs)
        st = {"mu": feats.mean(0), "sigma": np.cov(feats, rowvar=False)}
        ddim_sweep.append({"steps": steps, "fid": metrics.fid_from_stats(real_stats, st),
                           "seconds": speed[f"ddim{steps}"]})
        print(f"  [ddim] {steps} 步 FID={ddim_sweep[-1]['fid']:.2f}", flush=True)
        del imgs, feats
        torch.cuda.empty_cache()

    # 6) 保存样本供出图：用 FID 最低的 DDIM 步数生成 80 张画廊
    best_sweep = min(ddim_sweep, key=lambda r: r["fid"])
    gallery_steps = best_sweep["steps"]
    gallery = sample_ddim(model, diffusion, 80, device, steps=gallery_steps, seed=2024, batch=80)
    torch.save({"samples": (gallery * 255).round().clamp(0, 255).to(torch.uint8),
                "step": best_ckpt["step"], "ddim_steps": gallery_steps},
               RUN / "samples_final.pt")

    result = {
        "eval_config": {
            "device": torch.cuda.get_device_name(0),
            "ddim_steps": 50, "n_fid": 4000, "n_ref": real_stats["n"],
            "fid_reference": FID_REF.name,
            "memory_fraction": MEM_FRACTION,
            "model_params": cfg["params"],
            "best_step": best_ckpt["step"],
        },
        "screening": screening,
        "best": best_ckpt,
        "final": {
            "step": best_ckpt["step"],
            "ckpt": best_ckpt["ckpt"],
            "fid": fid_final,
            "c2st": c2st,
            "nn_gen": metrics.nn_summary(nn),
            "nn_real": metrics.nn_summary(nn_real),
            "nn_ratio": round(float(np.mean(nn) / max(np.mean(nn_real), 1e-9)), 4),
            "color": color,
            "color_real": color_real,
            "feature_diversity": feat_div,
            "feature_diversity_real": feat_div_real,
            "sampling_seconds": speed,
            "ddim50_sampling_seconds_4000": round(ddim50_seconds, 1),
            "gallery_ddim_steps": gallery_steps,
        },
        "ddim_sweep": ddim_sweep,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] 结果已写入 {OUT}", flush=True)


if __name__ == "__main__":
    main()
