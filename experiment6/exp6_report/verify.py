#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正确性验证：在跑大规模训练之前，先用小实验确认数据、模型、梯度和评估指标都正常。

检查项：
    1. 数据形状/范围、去重统计、训练/参考集不重叠
    2. 生成器/判别器/VAE 的输出形状、参数量、输出范围
    3. 生成器能通过判别器得到非零梯度（反向传播通路正常）
    4. 判别器能在单个 batch 上过拟合到接近 100% 准确率（说明卷积和损失都工作）
    5. VAE 在单个 batch 上过拟合，重建损失显著下降
    6. FID 指标的合理性：真实图对真实参考集 FID 很小，对噪声/未训练生成器 FID 很大
    7. 潜空间插值：相邻帧平滑变化（不是乱跳）
    8. 图片数据都在 [0,1] 内

运行方式：
    cd exp6_report && python verify.py
输出：
    verification.json（报告的"正确性验证"一节直接引用其中的数字）
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code"))

import metrics
from data import (IMAGE_SIZE, load_clean_images, split_indices, subset_indices,
                  to_gan_range, to_unit_range)
from engine import (GANConfig, VAEConfig, get_device, interpolate_gan,
                    sample_gan, set_seed, train_dcgan, train_vae)
from models import Discriminator, Generator, VAE, count_parameters

VER_PATH = HERE / "verification.json"


def _check(records: list[dict], name: str, passed: bool, value, expected: str,
           detail: str = "") -> None:
    records.append({"name": name, "passed": bool(passed), "value": value,
                    "expected": expected, "detail": detail})
    flag = "PASS" if passed else "FAIL"
    print(f"  [{flag}] {name}: {value} | 期望 {expected}")


def run_verification(device: torch.device, verbose: bool = True) -> dict:
    t0 = time.time()
    set_seed(0)
    images, names, dedup = load_clean_images()
    train_idx, ref_idx = split_indices(len(images))
    records: list[dict] = []

    # ---------------- 1. 数据 ----------------
    _check(records, "数据张量形状", tuple(images.shape) == (17029, 3, 64, 64),
           list(images.shape), "(17029, 3, 64, 64)")
    _check(records, "数据 dtype/范围", images.dtype == torch.uint8,
           str(images.dtype), "uint8（0~255）")
    overlap = len(set(train_idx.tolist()) & set(ref_idx.tolist()))
    _check(records, "训练集/参考集不重叠", overlap == 0, overlap, "0")
    _check(records, "去重统计", dedup["n_unique"] == len(images),
           dedup, "重复率约 21%")
    x_sample = images[torch.as_tensor(ref_idx[:256])].float().div_(255.0)
    _check(records, "图片值域", bool(x_sample.min() >= 0 and x_sample.max() <= 1),
           [round(float(x_sample.min()), 3), round(float(x_sample.max()), 3)], "[0, 1]")

    # ---------------- 2. 模型形状与参数量 ----------------
    G = Generator().to(device)
    D = Discriminator().to(device)
    V = VAE().to(device)
    z = torch.randn(8, 100, device=device)
    g_out = G(z)
    _check(records, "生成器输出形状", tuple(g_out.shape) == (8, 3, 64, 64),
           list(g_out.shape), "(8, 3, 64, 64)")
    _check(records, "生成器输出范围", bool(g_out.min() >= -1.001 and g_out.max() <= 1.001),
           [round(float(g_out.min()), 3), round(float(g_out.max()), 3)], "[-1, 1]（tanh）")
    d_out = D(g_out.detach())
    _check(records, "判别器输出形状", tuple(d_out.shape) == (8,), list(d_out.shape), "(8,) logits")
    x = x_sample[:8].to(device)
    recon_logits, mu, logvar = V(x)
    _check(records, "VAE 重建形状", tuple(recon_logits.shape) == (8, 3, 64, 64),
           list(recon_logits.shape), "(8, 3, 64, 64)")
    _check(records, "VAE μ/logσ² 形状", tuple(mu.shape) == (8, 128) and tuple(logvar.shape) == (8, 128),
           [list(mu.shape), list(logvar.shape)], "(8, 128)")
    recon_prob = torch.sigmoid(recon_logits)
    _check(records, "VAE 重建值域", bool(recon_prob.min() >= 0 and recon_prob.max() <= 1),
           [round(float(recon_prob.min()), 3), round(float(recon_prob.max()), 3)], "[0, 1]（sigmoid）")

    params = {"G": count_parameters(G), "D": count_parameters(D), "VAE": count_parameters(V)}
    _check(records, "参数量", params["G"] > 1e6 and params["D"] > 1e6 and params["VAE"] > 1e6,
           params, "三者均 > 1e6，与结构匹配")

    # ---------------- 3. 生成器梯度通路 ----------------
    G.zero_grad(set_to_none=True)
    D.zero_grad(set_to_none=True)
    fake = G(torch.randn(16, 100, device=device))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(D(fake), torch.ones(16, device=device))
    loss.backward()
    n_params = n_nonzero = 0
    for p in G.parameters():
        if p.requires_grad:
            n_params += 1
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                n_nonzero += 1
    _check(records, "生成器梯度非零", n_nonzero == n_params, f"{n_nonzero}/{n_params}", "全部参数都有非零梯度")
    G.zero_grad(set_to_none=True)
    D.zero_grad(set_to_none=True)

    # ---------------- 4. 判别器单 batch 过拟合 ----------------
    set_seed(1)
    D2 = Discriminator().to(device)
    fake_pool = sample_gan(G, 128, device, seed=7)
    fake_pool = to_gan_range(fake_pool).to(device)
    real_pool = to_gan_range(x_sample[:128].to(device))
    opt = torch.optim.Adam(D2.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    ones = torch.ones(128, device=device)
    zeros = torch.zeros(128, device=device)

    def d_acc() -> float:
        with torch.no_grad():
            pr = torch.sigmoid(D2(real_pool)).mean().item()
            pf = torch.sigmoid(D2(fake_pool)).mean().item()
        return (pr + (1 - pf)) / 2

    acc0 = d_acc()
    for _ in range(150):
        opt.zero_grad(set_to_none=True)
        loss = bce(D2(real_pool), ones) + bce(D2(fake_pool), zeros)
        loss.backward()
        opt.step()
    acc1 = d_acc()
    _check(records, "判别器单 batch 过拟合", acc1 > 0.95,
           {"初始": round(acc0, 4), "150 步后": round(acc1, 4)}, "> 0.95（能学会区分固定真/假样本）")

    # ---------------- 5. VAE 单 batch 过拟合 ----------------
    set_seed(2)
    V2 = VAE().to(device)
    xb = x_sample[:64].to(device)
    opt = torch.optim.Adam(V2.parameters(), lr=1e-3)
    initial = final = None
    for step in range(200):
        recon_logits, mu, logvar = V2(xb)
        loss, recon, kl = V2.elbo_loss(recon_logits, xb, mu, logvar)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            initial = float(recon.item())
        final = float(recon.item())
    ratio = final / max(initial, 1e-9)
    _check(records, "VAE 单 batch 过拟合", ratio < 0.5,
           {"初始重建损失": round(initial, 2), "200 步后": round(final, 2),
            "剩余比例": round(ratio, 4)}, "重建损失下降超过一半")

    # ---------------- 6. FID 合理性 ----------------
    print("  计算 FID 合理性检查（首次会加载 Inception-v3）...")
    fid = metrics.FIDEvaluator(device)
    real_512 = images[torch.as_tensor(ref_idx[:512])].float().div_(255.0)
    stats_full = fid.stats(images[torch.as_tensor(ref_idx)].float().div_(255.0))
    fid_real = fid.score(stats_full, real_512)
    noise = torch.rand(512, 3, 64, 64)
    fid_noise = fid.score(stats_full, noise)
    fake_untrained = sample_gan(Generator().to(device), 512, device, seed=3)
    fid_untrained = fid.score(stats_full, fake_untrained)
    _check(records, "FID 合理性", fid_real < 60 and fid_noise > 100 and fid_untrained > 100,
           {"真实图(512张)": round(fid_real, 2), "均匀噪声": round(fid_noise, 2),
            "未训练生成器": round(fid_untrained, 2)}, "真实图 < 60，噪声/未训练 > 100")
    del fid
    torch.cuda.empty_cache()

    # ---------------- 7. 潜空间插值平滑性 ----------------
    set_seed(3)
    G3 = Generator().to(device)
    interp = interpolate_gan(G3, device, n_steps=10, seed=0)
    diffs = (interp[1:] - interp[:-1]).abs().flatten(1).mean(dim=1)
    _check(records, "潜空间插值平滑", bool(diffs.mean() > 0 and diffs.max() < 0.5),
           {"相邻帧平均差": round(float(diffs.mean()), 4),
            "最大帧间差": round(float(diffs.max()), 4)}, "相邻帧有变化但差异小于 0.5")

    # ---------------- 8. 最近邻指标合理性 ----------------
    gen64 = sample_gan(G3, 64, device, seed=4)
    train_pool = images[torch.as_tensor(subset_indices(train_idx, 2000, seed=0))].float().div_(255.0)
    d_gen = metrics.min_distances(gen64, train_pool, device)
    d_real = metrics.min_distances(images[torch.as_tensor(ref_idx[:64])].float().div_(255.0),
                                   train_pool, device)
    _check(records, "最近邻距离指标", bool(d_gen.mean() > 10 and d_real.mean() > 10),
           {"未训练生成器": round(float(d_gen.mean()), 2),
            "真实留出图": round(float(d_real.mean()), 2)}, "均为正且量级合理")

    result = {
        "checks": records,
        "n_passed": sum(r["passed"] for r in records),
        "n_total": len(records),
        "runtime_s": round(time.time() - t0, 2),
        "device": str(device),
        "params": params,
        "fid_sanity": {"real_512": round(fid_real, 3), "uniform_noise": round(fid_noise, 3),
                       "untrained_generator": round(fid_untrained, 3)},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return result


def main() -> dict:
    device = get_device()
    print(f"=== 正确性验证（device={device}）===")
    result = run_verification(device)
    VER_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n通过 {result['n_passed']}/{result['n_total']} 项检查，耗时 {result['runtime_s']}s，"
          f"已写入 {VER_PATH.name}")
    if result["n_passed"] != result["n_total"]:
        failed = [r["name"] for r in result["checks"] if not r["passed"]]
        print("未通过：", failed)
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
