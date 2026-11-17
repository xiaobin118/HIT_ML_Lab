#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DDPM 训练脚本：在 Anime Face 64×64 数据上训练扩散模型（U-Net + EMA + 余弦调度）。

安全性（针对 6GB 笔记本显卡）：
    - 进程启动前设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，减少显存碎片；
    - torch.cuda.set_per_process_memory_fraction(0.72) 硬性限制显存（约 4.3GB），
      超出只会报 CUDA OOM 而不会把整机拖垮；
    - 默认 batch=32、base=64（18.6M 参数），配合 bf16 AMP + torch.compile 加速。

运行方式：
    cd exp6_report
    python diffusion_train.py --steps 200000 --max-hours 9.2
断点续训：
    python diffusion_train.py --steps 200000 --max-hours 9.2 --resume
输出（默认 exp6_report/diffusion_run/）：
    ckpt_last.pt       最近一次完整状态（模型/EMA/优化器/步数），用于断点续训
    ema/ema_XXXXX.pt   每 10k 步保存的 EMA 权重，供评估脚本挑选最佳 checkpoint
    samples/step_XXXXX.png  每 2.5k 步的 16 张 DDIM-50 采样（监控质量）
    train_log.jsonl    每 50 步的 loss / lr / 吞吐
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code"))

from data import load_clean_images, split_indices  # noqa: E402
from diffusion import (EMA, GaussianDiffusion, UNet, count_parameters,  # noqa: E402
                       sample_ddim)


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------


def make_loader(images: torch.Tensor, indices: np.ndarray, batch_size: int, seed: int):
    ds = torch.utils.data.TensorDataset(images[torch.as_tensor(indices)])
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=True, generator=g,
    )


def batch_to_x0(batch_uint8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 [0,255] → float [-1,1]，并做随机水平翻转（动漫头像近似左右对称）。"""
    x = batch_uint8.to(device, non_blocking=True).float().div_(127.5).sub_(1.0)
    flip = torch.rand(x.size(0), device=device) < 0.5
    if flip.any():
        x[flip] = torch.flip(x[flip], dims=[3])
    return x


# ---------------------------------------------------------------------------
# 学习率调度
# ---------------------------------------------------------------------------


def lr_at(step: int, total: int, base_lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return base_lr * (min_ratio + (1 - min_ratio) * cos)


# ---------------------------------------------------------------------------
# 保存 / 加载
# ---------------------------------------------------------------------------


def save_checkpoint(path: Path, model, ema, opt, step: int, cfg: dict, ema_only: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ema": ema.state_dict(), "step": step, "cfg": cfg}
    if not ema_only:
        payload.update({"model": model.state_dict(), "opt": opt.state_dict()})
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


@torch.no_grad()
def save_sample_grid(ema_model, diffusion, path: Path, n: int = 16, ddim_steps: int = 50,
                     seed: int = 1234):
    from torchvision.utils import save_image
    imgs = sample_ddim(ema_model, diffusion, n=n, device=next(ema_model.parameters()).device,
                       steps=ddim_steps, seed=seed, batch=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(imgs, path, nrow=4, padding=2)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--schedule", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(HERE / "diffusion_run"))
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--sample-every", type=int, default=2500)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--ema-every", type=int, default=10000)
    ap.add_argument("--max-hours", type=float, default=9.2)
    ap.add_argument("--mem-fraction", type=float, default=0.72)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "samples").mkdir(parents=True, exist_ok=True)
    (outdir / "ema").mkdir(parents=True, exist_ok=True)

    # ---- 随机种子 & 显存保护 ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 数据 ----
    images, _, _ = load_clean_images()
    train_idx, _ = split_indices(len(images))
    print(f"[data] 训练图片 {len(train_idx)} 张，batch={args.batch}", flush=True)

    # ---- 模型 ----
    model = UNet(base=args.base, dropout=args.dropout).to(device)
    ema = EMA(model, decay=args.ema_decay)
    diffusion = GaussianDiffusion(T=args.T, schedule=args.schedule, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    print(f"[model] U-Net base={args.base} params={count_parameters(model):,}", flush=True)

    cfg = {
        "steps": args.steps, "batch": args.batch, "base": args.base, "lr": args.lr,
        "warmup": args.warmup, "ema_decay": args.ema_decay, "dropout": args.dropout,
        "T": args.T, "schedule": args.schedule, "seed": args.seed,
        "params": count_parameters(model),
        "mem_fraction": args.mem_fraction,
    }
    (outdir / "train_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                              encoding="utf-8")

    # ---- 断点续训 ----
    start_step = 0
    ckpt = outdir / "ckpt_last.pt"
    if args.resume and ckpt.exists():
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        ema.load_state_dict(blob["ema"])
        opt.load_state_dict(blob["opt"])
        start_step = int(blob["step"])
        print(f"[resume] 从第 {start_step} 步继续", flush=True)

    # ---- torch.compile（训练用的模型） ----
    train_model = model
    if not args.no_compile:
        print("[compile] 编译 U-Net（首次前向约需 1 分钟）...", flush=True)
        train_model = torch.compile(model)

    loader = make_loader(images, train_idx, args.batch, args.seed)
    log_path = outdir / "train_log.jsonl"
    log_fh = log_path.open("a", encoding="utf-8")

    def log(obj):
        log_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        log_fh.flush()

    # ---- 训练循环 ----
    t_start = time.time()
    step = start_step
    data_iter = None
    loss_acc, n_acc = 0.0, 0
    t_last = time.time()
    stop_reason = "max_steps"
    print(f"[train] 目标 {args.steps} 步，最长 {args.max_hours} 小时", flush=True)
    try:
        while step < args.steps:
            if data_iter is None:
                data_iter = iter(loader)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            x0 = batch_to_x0(batch[0], device)
            lr = lr_at(step + 1, args.steps, args.lr, args.warmup)
            for pg in opt.param_groups:
                pg["lr"] = lr

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diffusion.training_loss(train_model, x0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            step += 1
            loss_acc += float(loss.detach())
            n_acc += 1

            if step % args.log_every == 0:
                now = time.time()
                sps = args.log_every / max(now - t_last, 1e-9)
                t_last = now
                rec = {"step": step, "loss": round(loss_acc / n_acc, 5), "lr": round(lr, 7),
                       "steps_per_s": round(sps, 3),
                       "elapsed_s": round(now - t_start, 1),
                       "mem_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)}
                log(rec)
                if step % (args.log_every * 20) == 0:
                    print(f"  step {step:6d}/{args.steps} loss {rec['loss']:.4f} "
                          f"lr {lr:.2e} {sps:.2f} it/s mem {rec['mem_gb']:.2f}GB "
                          f"elapsed {(now - t_start)/60:.1f}min", flush=True)
                loss_acc, n_acc = 0.0, 0

            if step % args.sample_every == 0:
                t_sample = time.time()
                imgs = save_sample_grid(ema.ema, diffusion,
                                        outdir / "samples" / f"step_{step:06d}.png",
                                        n=16, ddim_steps=50)
                print(f"  [sample] step {step} 已保存 16 张 DDIM-50 样本"
                      f"（{time.time() - t_sample:.1f}s）", flush=True)

            if step % args.ckpt_every == 0:
                save_checkpoint(ckpt, model, ema, opt, step, cfg)

            if step % args.ema_every == 0:
                save_checkpoint(outdir / "ema" / f"ema_{step:06d}.pt", model, ema, opt,
                                step, cfg, ema_only=True)

            if (time.time() - t_start) > args.max_hours * 3600:
                stop_reason = f"max_hours({args.max_hours})"
                print(f"[stop] 达到时间上限 {args.max_hours} 小时，停止训练", flush=True)
                break
    finally:
        save_checkpoint(ckpt, model, ema, opt, step, cfg)
        save_checkpoint(outdir / "ema" / f"ema_{step:06d}.pt", model, ema, opt, step, cfg,
                        ema_only=True)
        log_fh.close()
        total = time.time() - t_start
        (outdir / "train_summary.json").write_text(json.dumps({
            "final_step": step, "stop_reason": stop_reason,
            "train_seconds": round(total, 1), "cfg": cfg,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] 训练结束：step={step}，耗时 {total/3600:.2f} 小时，"
              f"平均 {step / max(total, 1e-9):.2f} 步/秒", flush=True)


if __name__ == "__main__":
    main()
